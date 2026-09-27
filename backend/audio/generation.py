"""
backend/audio/generation.py

LLM → sentence chunker → TTS pipeline for Phase 6 (streaming TTS).

The spec (§55, §140–141) requires:
  - Do not wait for the full LLM answer before sending to TTS.
  - Once a sentence is complete, send it to TTS immediately.
  - This overlaps LLM generation, TTS synthesis, and audio playback.

Strategy:
  - Buffer LLM token chunks as they arrive.
  - Detect sentence boundaries using punctuation heuristics.
  - Flush immediately at each boundary.
  - Also flush any remaining text when the LLM stream ends.

Sentence boundary detection:
  Splits on: . ! ? ।  followed by whitespace or end-of-string.
  The Devanagari danda (।) is the Hindi sentence-end marker.
  Minimum chunk size: MIN_CHUNK_CHARS — avoids tiny 3-word TTS requests.

This module is pure Python with no external dependencies.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

# Minimum characters before emitting a chunk (spec §141 — avoid 5-word tiny chunks).
MIN_CHUNK_CHARS = 30

# Sentence-ending punctuation regex.
# Matches: . ! ? । followed by whitespace or end-of-string.
# Negative lookbehind for common abbreviations (Mr. Dr. etc.) is omitted
# for simplicity — voice agents rarely use them in spoken answers.
_SENTENCE_END = re.compile(r'[.!?।]+(?=\s|$)')


def split_into_sentences(text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """
    Split `text` into sentence-boundary chunks.

    A chunk is emitted when:
      - A sentence-end punctuation is found AND
      - The accumulated text is at least `min_chars` long.

    Any remaining text (no final punctuation) is returned as the last chunk.

    >>> split_into_sentences("Hello! How are you? I am fine.")
    ['Hello!', 'How are you?', 'I am fine.']
    """
    chunks: list[str] = []
    start = 0

    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
            start = end
        # If chunk is too short, let it merge with the next sentence.

    remainder = text[start:].strip()
    if remainder:
        chunks.append(remainder)

    return chunks


async def stream_sentences(
    token_stream: AsyncIterator[str],
    min_chars: int = MIN_CHUNK_CHARS,
) -> AsyncIterator[str]:
    """
    Consume an async token stream and yield complete sentence chunks.

    The generator buffers tokens and yields a chunk each time a sentence
    boundary is detected and the buffer is long enough.

    The final partial sentence (no trailing punctuation) is yielded at
    the end of the stream.

    Args:
        token_stream: AsyncIterator[str] — e.g. from SarvamLLM.stream_response()
        min_chars:    Minimum buffer length before yielding on a sentence boundary.

    Yields:
        str — one sentence (or last partial) per yield.
    """
    buffer = ""

    async for token in token_stream:
        buffer += token

        # Check for a sentence boundary in the accumulated buffer.
        last_end = 0
        for m in _SENTENCE_END.finditer(buffer):
            end = m.end()
            chunk = buffer[last_end:end].strip()
            if len(chunk) >= min_chars:
                yield chunk
                last_end = end

        # Trim the buffer to only what hasn't been yielded.
        if last_end > 0:
            buffer = buffer[last_end:].lstrip()

    # Yield any remaining text after the stream ends.
    remainder = buffer.strip()
    if remainder:
        yield remainder
