"""Find every rip-relative reference to a string literal, and name the method.

`tools/xrefs.py` finds `call rel32` sites, which answers "who calls this method".
It cannot answer "who reads this JSON key", because a key is a string literal
loaded through a `lea reg, [rip + disp]` rather than called. That question comes
up constantly here: the balance tables are JSON, the game reads them by name,
and the only way to find the code behind a column is to find who mentions it.

    python tools/data_xrefs.py FightInFirst outcomes
    python tools/data_xrefs.py --addr 0x2ba60d0

Scans `.text` and `il2cpp`, the same two sections `xrefs.py` learned to scan
after reporting "0 call sites" for methods that plainly had callers.

The scan is arithmetic rather than disassembly. For a rip-relative operand whose
4-byte displacement sits at file offset `o`, the target is
`rva(o + 4) + disp`, and `rva` is linear inside a section, so the displacement a
hit would need is itself linear in `o`. That makes the whole thing one vectorised
comparison per section instead of disassembling 35 MB. It over-reports by
construction: any four bytes that happen to hold the right number look like a
reference, so every hit is reported with the instruction bytes before it, and a
real `lea` shows up as `48 8d` with a ModRM naming rip.
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import pickle
import sys

import numpy as np
import pefile

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DLL = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\GameAssembly.dll")
SCRIPT = pathlib.Path("data/extracted/il2cpp/script.json")
SYMMAP = pathlib.Path("data/extracted/symmap.pkl")
SECTIONS = (b".text", b"il2cpp")


def load():
    raw = DLL.read_bytes()
    pe = pefile.PE(str(DLL), fast_load=True)
    sym = pickle.loads(SYMMAP.read_bytes())
    order = sorted(sym)
    return raw, pe, sym, order


def strings() -> dict[str, int]:
    d = json.loads(SCRIPT.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for e in d["ScriptString"]:
        out.setdefault(e["Value"], int(e["Address"]))
    return out


def owner(rva: int, sym: dict, order: list) -> str:
    """The symbol whose body contains this address, by nearest one below it."""
    i = bisect.bisect_right(order, rva) - 1
    if i < 0:
        return "?"
    return f"{sym[order[i]]}+{rva - order[i]:#x}"


def find(raw: bytes, pe, target_rva: int) -> list[int]:
    """File offsets whose 4-byte displacement points at `target_rva`."""
    hits: list[int] = []
    for s in pe.sections:
        if s.Name.rstrip(b"\x00") not in SECTIONS:
            continue
        off0 = s.PointerToRawData
        size = min(s.SizeOfRawData, len(raw) - off0)
        rva0 = s.VirtualAddress
        b = np.frombuffer(raw, dtype=np.uint8, count=size, offset=off0)
        if size < 8:
            continue
        disp = (b[0:size - 3].astype(np.uint32)
                | (b[1:size - 2].astype(np.uint32) << 8)
                | (b[2:size - 1].astype(np.uint32) << 16)
                | (b[3:size].astype(np.uint32) << 24)).view(np.int32)
        # rva(o + 4) + disp == target, and rva(o) = rva0 + (o - off0)
        o = np.arange(disp.size, dtype=np.int64)
        need = target_rva - rva0 - o - 4
        hits += (off0 + o[disp == need.astype(np.int32)]).tolist()
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="string literals to look up")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=None,
                    help="an RVA to look up directly")
    ap.add_argument("--max", type=int, default=12, help="hits to print each")
    args = ap.parse_args()

    raw, pe, sym, order = load()
    table = strings() if args.names else {}
    targets = [(f"{n!r}", table[n]) for n in args.names if n in table]
    for n in args.names:
        if n not in table:
            print(f"{n!r}: not in the string literal table")
    if args.addr is not None:
        targets.append((f"{args.addr:#x}", args.addr))

    for label, rva in targets:
        hits = find(raw, pe, rva)
        print(f"\n{label} at rva {rva:#x}: {len(hits)} candidate references")
        for off in hits[:args.max]:
            here = pe.get_rva_from_offset(off)
            before = raw[off - 3:off].hex(" ")
            print(f"  off={off:#x} rva={here:#x}  prefix={before:11s}  "
                  f"{owner(here, sym, order)}")
        if len(hits) > args.max:
            print(f"  ... {len(hits) - args.max} more")


if __name__ == "__main__":
    main()
