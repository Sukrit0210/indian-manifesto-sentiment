import re
import unicodedata

import pymupdf
import pysbd

SKIP = {
    "AIADMK-2019": range(1, 6),
    "INC-2014": [1, 2],
    "INC-2019": [1, 2, 3],
    "INC-2009": [1, 2],
    "INC-2024": [1, 2, 3],
    "BJP-2019": [1, 2],
    "BJP-2014": [1, 2, 3, 4, 5, 7, 8],
    "BJP-2024": [1, 2, 3],
    "TMC-2024": [1, 2, 3, 9],
    "DMK-2024": [1, 2, 3],
}
segmenter = pysbd.Segmenter(language="en", clean=False)
promise = re.compile(
    r"\b(will|shall|pledge[sd]?|promise[sd]?|commit(?:ted|s)? to|undertake to|guarantee to)\b",
    re.I,
)


def chunks(raw, doc_id):
    if doc_id == "BJP-2019":
        raw = raw.translate(
            str.maketrans(
                {
                    "Ē": "rt",
                    "Ĕ": "tt",
                    "Ą": "ff",
                    "ď": "ft",
                    "ć": "ffi",
                    "ē": "tf",
                    "ª": "fi",
                    "Ÿ": "•",
                }
            )
        )
    raw = unicodedata.normalize("NFKC", raw).replace("\xad", "").replace("\x01", "•")
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    raw = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", raw)
    raw = re.sub(r"(?m)^\s*\d{1,3}[.)]\s+", "\n• ", raw)
    for unit in re.split(r"[•●▪►»\uf0b7]", raw):
        joined = " ".join(unit.split())
        if joined:
            yield from (s.strip() for s in segmenter.segment(joined) if s.strip())


def page_blocks(page, key):
    blocks = page.get_text("blocks")
    if key not in {"AIADMK-2019", "DMK-2024"}:
        return blocks
    # These single-column PDFs store wrapped lines as separate blocks.
    result, group = [], []

    def flush():
        if group:
            box = pymupdf.Rect(group[0][:4])
            for b in group[1:]:
                box |= pymupdf.Rect(b[:4])
            result.append(
                (*box, "\n".join(b[4].strip() for b in group), group[0][5], 0)
            )
            group.clear()

    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        line = b[4].strip()
        if key == "AIADMK-2019" and (b[1] < 75 or b[1] > 715):
            continue
        if not line or re.fullmatch(r"[-\s\d]+", line):
            flush()
            continue
        if line.isupper() or re.match(r"^(?:\d+[.)]|[•●])\s", line):
            flush()
        group.append(b)
        if line.isupper() or line.endswith((".", "!", "?")):
            flush()
    flush()
    return result
