"""
backend/persistence/callback_store.py

Phase 12: SQLite Persistence Layer for Human-Callback Bookings (Spec §38–42, §88).

Requirements:
  - Lightweight SQLite wrapper using python's built-in `sqlite3`.
  - Schema: slot_id, date, start_time, end_time, booked, booking_reference, idempotency_key, language.
  - Seed database with mock available upcoming slots.
  - Atomicity: UPDATE callback_slots SET booked = 1, booking_reference = ?, idempotency_key = ? ...
  - Idempotency: Handling duplicate/retry booking attempts gracefully with existing booking reference.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default SQLite database path (can be :memory: or local file)
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "callbacks.db"


class CallbackStore:
    """
    SQLite-backed store for scheduling human agent callbacks.
    Thread-safe connection handling per method or shared connection.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path = str(_DEFAULT_DB_PATH)
        else:
            self.db_path = str(db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the schema and seed default mock slots if empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS callback_slots (
                    slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    booked INTEGER DEFAULT 0,
                    booking_reference TEXT,
                    idempotency_key TEXT,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_callback_date ON callback_slots(date);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_callback_idempotency ON callback_slots(idempotency_key);"
            )
            conn.commit()

            # Check if seeded
            cursor.execute("SELECT COUNT(*) FROM callback_slots")
            count = cursor.fetchone()[0]
            if count == 0:
                self._seed_default_slots(conn)

    def _seed_default_slots(self, conn: sqlite3.Connection) -> None:
        """Seed available slots for today, tomorrow, and the next 5 days."""
        today = date.today()
        slots_to_seed = []
        standard_times = [
            ("10:00 AM", "10:30 AM"),
            ("11:30 AM", "12:00 PM"),
            ("02:00 PM", "02:30 PM"),
            ("03:30 PM", "04:00 PM"),
            ("05:00 PM", "05:30 PM"),
        ]

        for day_offset in range(7):
            slot_date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for start, end in standard_times:
                slots_to_seed.append((slot_date, start, end, 0))

        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO callback_slots (date, start_time, end_time, booked)
            VALUES (?, ?, ?, ?);
            """,
            slots_to_seed,
        )
        conn.commit()
        logger.info("[CallbackStore] Seeded %d mock callback slots.", len(slots_to_seed))

    def get_available_slots(self, date_str: str | None = None) -> list[dict[str, Any]]:
        """
        Return unbooked slots for a given date (YYYY-MM-DD), or default upcoming dates.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if date_str:
                cursor.execute(
                    """
                    SELECT slot_id, date, start_time, end_time, booked
                    FROM callback_slots
                    WHERE date = ? AND booked = 0
                    ORDER BY slot_id ASC;
                    """,
                    (date_str,),
                )
            else:
                # If no date specified, return upcoming unbooked slots
                cursor.execute(
                    """
                    SELECT slot_id, date, start_time, end_time, booked
                    FROM callback_slots
                    WHERE booked = 0
                    ORDER BY date ASC, slot_id ASC
                    LIMIT 10;
                    """
                )
            rows = cursor.fetchall()
            return [
                {
                    "slot_id": row["slot_id"],
                    "date": row["date"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "time_slot": f"{row['start_time']} - {row['end_time']}",
                }
                for row in rows
            ]

    def book_slot(
        self,
        date_str: str,
        time_slot: str,
        idempotency_key: str,
        language: str = "en-IN",
    ) -> dict[str, Any]:
        """
        Atomically book a slot (Spec §40, §88).

        Handling:
          1. Parse time_slot (supports '10:00 AM' or '10:00 AM - 10:30 AM').
          2. First check if this exact idempotency_key already booked a slot (Idempotency check).
          3. Atomic UPDATE WHERE date = ? AND start_time = ? AND booked = 0.
          4. If rowcount == 1: Success.
          5. If rowcount == 0: Check if booked by another user (Failure/Unavailable).
        """
        # Clean / parse time slot start time
        start_time = time_slot.split("-")[0].strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Idempotency Check: Did this key already book this slot?
            if idempotency_key:
                cursor.execute(
                    """
                    SELECT slot_id, date, start_time, end_time, booking_reference, language
                    FROM callback_slots
                    WHERE idempotency_key = ?;
                    """,
                    (idempotency_key,),
                )
                existing = cursor.fetchone()
                if existing:
                    logger.info(
                        "[CallbackStore] Idempotency hit: key %s already has booking %s",
                        idempotency_key, existing["booking_reference"],
                    )
                    return {
                        "success": True,
                        "status": "CONFIRMED",
                        "booking_reference": existing["booking_reference"],
                        "date": existing["date"],
                        "time_slot": f"{existing['start_time']} - {existing['end_time']}",
                        "language": existing["language"],
                        "idempotent_replay": True,
                        "message": (
                            f"Your callback is confirmed with booking reference {existing['booking_reference']} "
                            f"for {existing['date']} at {existing['start_time']}."
                        ),
                    }

            # 2. Generate a unique human-friendly booking reference
            booking_reference = f"CBK-{uuid.uuid4().hex[:6].upper()}"

            # 3. Atomic Booking Transaction
            cursor.execute(
                """
                UPDATE callback_slots
                SET booked = 1,
                    booking_reference = ?,
                    idempotency_key = ?,
                    language = ?
                WHERE date = ? AND (start_time = ? OR start_time LIKE ?) AND booked = 0;
                """,
                (
                    booking_reference,
                    idempotency_key,
                    language,
                    date_str,
                    start_time,
                    f"{start_time}%",
                ),
            )
            conn.commit()

            if cursor.rowcount == 1:
                # Query full row for details
                cursor.execute(
                    "SELECT slot_id, date, start_time, end_time FROM callback_slots WHERE booking_reference = ?",
                    (booking_reference,),
                )
                row = cursor.fetchone()
                time_range = f"{row['start_time']} - {row['end_time']}" if row else time_slot
                logger.info(
                    "[CallbackStore] Successfully booked slot: %s %s (Ref: %s)",
                    date_str, time_range, booking_reference,
                )
                return {
                    "success": True,
                    "status": "CONFIRMED",
                    "booking_reference": booking_reference,
                    "date": date_str,
                    "time_slot": time_range,
                    "language": language,
                    "idempotent_replay": False,
                    "message": (
                        f"Your callback has been successfully booked for {date_str} at {time_range}. "
                        f"Your booking reference is {booking_reference}."
                    ),
                }

            # 4. Failed: Check why rowcount == 0
            cursor.execute(
                """
                SELECT slot_id, booked, booking_reference
                FROM callback_slots
                WHERE date = ? AND (start_time = ? OR start_time LIKE ?);
                """,
                (date_str, start_time, f"{start_time}%"),
            )
            slot_info = cursor.fetchone()

            if slot_info is None:
                return {
                    "success": False,
                    "status": "NOT_FOUND",
                    "booking_reference": None,
                    "message": f"No callback slot found for date '{date_str}' at '{time_slot}'.",
                    "reason": "Invalid date or time slot requested.",
                }
            elif slot_info["booked"] == 1:
                return {
                    "success": False,
                    "status": "ALREADY_BOOKED",
                    "booking_reference": None,
                    "message": f"The requested slot on {date_str} at {time_slot} is no longer available.",
                    "reason": "Slot already taken. Please choose another available time slot.",
                }
            else:
                return {
                    "success": False,
                    "status": "FAILED",
                    "booking_reference": None,
                    "message": "Failed to book the requested slot.",
                    "reason": "Slot could not be reserved.",
                }


# Global singleton instance for the app
callback_store = CallbackStore()
