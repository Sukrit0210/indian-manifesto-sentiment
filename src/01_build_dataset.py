"""Build the master passage table from the inventoried English PDF editions."""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import pymupdf
from boundaries import ACTION_START, WORDS
from common import CACHE, DATA, REGISTRY, ROOT, read, write
from extraction import SKIP, chunks, page_blocks

# Inclusive, one-based page ranges established by inspecting the scans and contents pages.
SPECIAL = {
    "AIADMK-2009": (5, 40),
    "AIADMK-2014": (5, 110),
    "DMK-2014": (7, 64),
    "DMK-2019": (8, 70),
    "TMC-2014": (2, 13),
}
KEYWORDS = json.loads((REGISTRY.parent / "keywords.json").read_text())
TOPICS = {
    t: [re.compile(r"\b" + re.escape(w) + r"s?\b", re.I) for w in words]
    for t, words in KEYWORDS.items()
}


def scan_chunks(words, key, height):
    """Rebuild lines geometrically since OCR can fragment a single baseline into many boxes."""
    lines = []
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        if key.startswith("DMK-") and (w[1] + w[3]) / 2 < 0.137 * height:
            continue
        center = (w[1] + w[3]) / 2
        if lines and abs(center - lines[-1][0]) < max(2.5, (w[3] - w[1]) * 0.58):
            lines[-1][1].append(w)
        else:
            lines.append((center, [w]))
    spacing = (
        statistics.median(
            [b[0] - a[0] for a, b in zip(lines, lines[1:]) if b[0] > a[0]]
        )
        if len(lines) > 1
        else 20
    )
    buffer = []
    previous = None
    for center, items in lines:
        line = " ".join(w[4] for w in sorted(items, key=lambda w: w[0])).strip()
        if re.fullmatch(r"[\W\d_]+", line):
            continue
        if key == "TMC-2014":
            line = re.sub(r"^a (?=[A-Z])", "• ", line)
        new = re.match(r"^(?:\d{1,3}[.)]|[•*+«])\s+", line)
        heading = line.isupper()
        if buffer and (
            new
            or heading
            or center - previous > spacing * 1.48
            or (re.search(r"[.!?]$", buffer[-1]) and re.match(r"[A-Z]", line))
        ):
            yield " ".join(buffer)
            buffer = []
        if not heading:
            buffer.append(line[new.end() :] if new else line)
        previous = center
    if buffer:
        yield " ".join(buffer)


def native_items(doc, key):
    pages = [page_blocks(p, key) for p in doc]
    freq = Counter()
    for blocks in pages:
        freq.update(set(" ".join(b[4].split()).casefold() for b in blocks if b[6] == 0))
    for pg, blocks in enumerate(pages, 1):
        if pg in SKIP.get(key, []):
            continue
        for b in blocks:
            if b[6] != 0:
                continue
            norm = " ".join(b[4].split()).casefold()
            if len(norm.split()) < 15 and freq[norm] >= 4:
                continue
            for idx, item in enumerate(
                re.split(
                    r"(?:[•●▪►»\uf0b7\uf074\uf06c\uf06e\uf0d8\x01]|(?m:^\s*\d{1,3}[.)]\s+))",
                    b[4],
                ),
                1,
            ):
                yield (
                    pg,
                    f"b{b[5]:03}-i{idx:03}",
                    " ".join(chunks(item, key)),
                    "pdf_text",
                )


def scanned_items(doc, key, cache, tessdata):
    for n in range(SPECIAL[key][0], SPECIAL[key][1] + 1):
        p = doc[n - 1]
        dest = cache / key / f"{n:03}.json"
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8"))
        else:
            words = p.get_text("words", sort=True)
            ocr = len(words) < 12
            if ocr:
                tp = p.get_textpage_ocr(
                    language="eng", dpi=240, full=True, tessdata=tessdata
                )
                words = p.get_text("words", textpage=tp, sort=True)
            data = {"words": words, "height": p.rect.height, "ocr": ocr}
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for i, t in enumerate(scan_chunks(data["words"], key, data["height"]), 1):
            yield n, f"g{i:03}", t, ("english_ocr" if data["ocr"] else "pdf_text")


def build(raw, cache, tessdata):
    registry = read(REGISTRY)
    master = []
    for src in registry:
        key = src["document_id"]
        if src["status"] != "included":
            continue
        pdf = raw / (key + ".pdf")
        if not pdf.is_file():
            raw.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                urllib.request.Request(
                    src["source_url"], headers={"User-Agent": "Mozilla/5.0"}
                ),
                timeout=90,
            ) as r:
                payload = r.read()
            if hashlib.sha256(payload).hexdigest() != src["sha256"]:
                raise ValueError(f"Edition changed for {key}")
            pdf.write_bytes(payload)
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != src["sha256"]:
            raise ValueError(f"PDF mismatch: {key}")
        seen = set()
        with pymupdf.open(pdf) as doc:
            items = (
                scanned_items(doc, key, cache, tessdata)
                if key in SPECIAL
                else native_items(doc, key)
            )
            for page, index, text, method in items:
                text = " ".join(unicodedata.normalize("NFKC", text).split())
                token_count = len(WORDS.findall(text))
                if (
                    token_count < 2
                    or not re.search("[a-z]{2}", text)
                    or re.search(r"\.{4,}|�{3,}", text)
                ):
                    continue
                if (
                    text.istitle()
                    and token_count <= 12
                    and not ACTION_START.match(text)
                    and not re.search(r"[.!?]$", text)
                ):
                    continue
                norm = " ".join(re.findall(r"\w+", text.casefold()))
                if norm in seen:
                    continue
                seen.add(norm)
                pid = f"{key}-p{page:03}-{index}"
                meta = dict(
                    document_id=key, party=src["party"], year=src["year"], page=page
                )
                scores = {
                    t: sum(bool(rx.search(text)) for rx in expr)
                    for t, expr in TOPICS.items()
                }
                area = (
                    max(scores, key=scores.get) if any(scores.values()) else "general"
                )
                master.append(
                    dict(
                        passage_id=pid,
                        doc_id=pid,
                        **meta,
                        language="en",
                        text=text,
                        word_count=token_count,
                        policy_area=area,
                        extraction=method,
                        source_url=src["source_url"],
                    )
                )
    if len({row["passage_id"] for row in master}) != len(master):
        raise ValueError("Duplicate passage IDs")
    for s in registry:
        k = s["document_id"]
        s["passages"] = sum(r["document_id"] == k for r in master)
        if s["status"] == "included":
            assert s["passages"] > 0
    write(DATA / "source_counts.csv", registry)
    write(DATA / "master_passages.csv", master)
    print(f"Saved {len(master):,} passages to {DATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--cache", type=Path, default=CACHE / "ocr")
    parser.add_argument("--tessdata", default=str(REGISTRY.parent / "ocr"))
    a = parser.parse_args()
    build(a.pdf_dir, a.cache, a.tessdata)
