"""Check deliverable text for forbidden dashes, italics and backtick highlighting."""
import re, sys, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parent
TARGETS = ["README.md", "report/main.tex", "notebooks/ablation_study.ipynb"]

ALLOW = [
    r"distilbert-base-uncased", r"bert-base-uncased", r"nyu-mll", r"fancyzhx",
    r"t-SNE", r"--[a-z-]+",            # CLI flags in code blocks
    r"^\s*#", r"pre-?trained",
]

def lines_of(path):
    p = ROOT / path
    if path.endswith(".ipynb"):
        nb = json.load(open(p))
        out = []
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "markdown":
                for ln in "".join(c["source"]).split("\n"):
                    out.append((f"cell {i}", ln))
        return out
    return [(str(n + 1), ln) for n, ln in enumerate(p.read_text().split("\n"))]

problems = []
in_code = {}
for path in TARGETS:
    fenced = False
    for loc, ln in lines_of(path):
        if ln.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        stripped = ln
        for pat in ALLOW:
            stripped = re.sub(pat, "", stripped)
        # em dash, en dash, or a hyphen used between word characters or as punctuation
        for m in re.finditer(r"[—–]|(?<=\w)-(?=\w)|\s-\s|--", stripped):
            problems.append((path, loc, m.group(0), ln.strip()[:100]))
        # inline backticks used on a single word
        for m in re.finditer(r"`[^`\n]{1,30}`", stripped):
            problems.append((path, loc, "backtick", m.group(0)))
        # markdown italics
        for m in re.finditer(r"(?<![\*_\w])[\*_][^\*_\n]{2,40}[\*_](?![\*_\w])", stripped):
            problems.append((path, loc, "italic", m.group(0)))

if not problems:
    print("STYLE CLEAN: no dashes, inline backticks or italics found in deliverable prose")
else:
    print(f"{len(problems)} issues")
    for p, loc, kind, ctx in problems[:40]:
        print(f"  {p}:{loc}  [{kind}]  {ctx}")
