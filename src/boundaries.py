import re

WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*")
ACT = r"(?:establish|introduce|create|raise|reduce|abolish|provide|ensure|expand|increase|implement|enact|set up|strengthen|build|launch|restore|review|protect|amend|extend|promote|repeal|support|allocate)\b"
SUBJ = r"(?:we|congress|the government|our government|the party|the bjp|the cpi\(m\)|the dmk|the tmc)"
PRED = re.compile(r"\b" + SUBJ + r"\s+(?:will|shall)\s+" + ACT, re.I)
BOUNDARY = re.compile(
    r"(?:;\s*(?:(?:and|but)\s+)?|,?\s+(?:and|but)\s+)(?="
    + SUBJ
    + r"\s+(?:will|shall)\s+"
    + ACT
    + r")",
    re.I,
)
ACTION_START = re.compile(
    r"^(?:(?:and|also)\s+)*(?:(?:we|congress|bjp)\s+(?:will|shall|would)\s+)?(?:"
    + ACT
    + r"|encourage|halt(?:ing)?|repeal(?:ing)?|enforc(?:e|ing)|"
    r"amend(?:ing)?|restor(?:e|ing)|extend(?:ing)?|implement(?:ing)?|"
    r"establish(?:ing)?|introduc(?:e|ing)|ensur(?:e|ing)|expand(?:ing)?|"
    r"build(?:ing)?|creat(?:e|ing)|scrap(?:ping)?|stop(?:ping)?|"
    r"giv(?:e|ing)|protect(?:ing)?|increas(?:e|ing))\b",
    re.I,
)


def spans(text):
    start = 0
    separators = {m.start(): m for m in re.finditer(r";\s*(?=\S)", text)}
    separators.update({m.start(): m for m in BOUNDARY.finditer(text)})
    for _, m in sorted(separators.items()):
        if m.start() < start:
            continue
        left, right = text[start : m.start()], text[m.end() :]
        if re.search(
            r"\b(?:in this endeavou?r|to this end|to achieve this|do (?:so|that))\b",
            right,
            re.I,
        ):
            continue
        if (PRED.search(left) or ACTION_START.match(left)) and ACTION_START.match(
            right
        ):
            yield start, m.start()
            start = m.end()
    yield start, len(text)
