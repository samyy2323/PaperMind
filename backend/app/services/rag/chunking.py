import re


def chunk_text(text: str, target_chars: int = 900) -> list[str]:
    """Split text into chunks of roughly one to two paragraphs.

    Paragraphs (blank-line separated) are merged until ~target_chars is reached; a single paragraph
    longer than 1.6x target is split on sentence boundaries.
    """
    text = text.replace("\r\n", "\n")
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if len(p) > 20]

    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= target_chars * 1.6:
            pieces.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) > target_chars:
                pieces.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) > target_chars:
            chunks.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks
