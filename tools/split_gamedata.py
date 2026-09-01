"""Split each decrypted group into its constituent files.

The payload is `path\n<body>\n` records; the game splits on newline and treats
any line ending in a known extension as the next filename. CSV bodies are stored
on one line with literal \n escapes (the game calls RegexParser.Unescape).
"""
import json, pathlib, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = pathlib.Path("data/extracted/gamedata")
OUT = pathlib.Path("data/extracted/json")
EXT = re.compile(r"\.(json|csv)$", re.I)

BS = chr(92)   # spelled this way so the source survives shell heredocs intact

def unescape(s):
    """Turn the stored two-character escapes into real control characters."""
    return (s.replace(BS + "r" + BS + "n", "\n")
             .replace(BS + "n", "\n")
             .replace(BS + "t", "\t")
             .replace(BS + '"', '"'))

total = 0
for f in sorted(SRC.glob("*.txt")):
    lines = f.read_text(encoding="utf-8").split("\n")
    group, i, n = OUT / f.stem, 0, 0
    while i < len(lines):
        name = lines[i].strip()
        if EXT.search(name) and i + 1 < len(lines):
            body, tgt = lines[i + 1], group / name
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if name.lower().endswith(".csv"):
                tgt.write_text(unescape(body), encoding="utf-8")
            else:
                try:
                    tgt.write_text(json.dumps(json.loads(body), indent=1, ensure_ascii=False), encoding="utf-8")
                except json.JSONDecodeError:
                    tgt.write_text(body, encoding="utf-8")
            n += 1
            i += 2
        else:
            i += 1
    print(f"{f.stem}: {n} files")
    total += n
print(f"\ntotal {total} -> {OUT}")
