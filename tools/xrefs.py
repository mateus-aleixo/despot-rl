"""Find every `call rel32` site that targets a given VA, and name the caller.

Scans `.text` **and** `il2cpp`. This used to scan `.text` only, which holds
almost nothing: Il2CppDumper puts every generated game method in the `il2cpp`
section (0x2e1000, 32 MB of it), so the tool reported "0 call sites" for real
methods and quietly supported the wrong conclusion that IL2CPP never emits a
direct call. It does: `C_Team.GetExperience` has five callers, and one of them
(`C_Unit.Die`) is how the player is paid for a kill.

    python tools/xrefs.py 0x180877980 [more VAs...]
"""
import bisect
import pathlib
import pickle
import sys

import numpy as np
import pefile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DLL = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\GameAssembly.dll")
CODE_SECTIONS = (b".text", b"il2cpp")

raw = DLL.read_bytes()
pe = pefile.PE(str(DLL), fast_load=True)
BASE = pe.OPTIONAL_HEADER.ImageBase
SYM = pickle.loads(pathlib.Path("data/extracted/symmap.pkl").read_bytes())
RVAS = sorted(SYM)

targets = [int(a, 16) for a in sys.argv[1:]]
if not targets:
    sys.exit(__doc__)
found = {t: [] for t in targets}

for sec in pe.sections:
    if sec.Name.rstrip(b"\0") not in CODE_SECTIONS:
        continue
    start, size = sec.PointerToRawData, sec.SizeOfRawData
    va0 = BASE + sec.VirtualAddress
    buf = np.frombuffer(raw[start:start + size], dtype=np.uint8)
    for i in np.flatnonzero(buf[:-5] == 0xE8):
        i = int(i)
        rel = int.from_bytes(raw[start + i + 1:start + i + 5], "little", signed=True)
        site = va0 + i + 5 + rel
        if site in found:
            found[site].append(va0 + i)


def owner(va: int) -> str:
    """The symbol whose body contains `va`."""
    j = bisect.bisect_right(RVAS, va - BASE) - 1
    return SYM.get(RVAS[j], "?") if j >= 0 else "?"


for t in targets:
    print(f"{t:#x}  {len(found[t])} call sites  ({SYM.get(t - BASE, '?')})")
    for va in found[t]:
        print(f"  {owner(va)}  at {va:#x}")
