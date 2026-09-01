"""Annotated disassembler: resolves call targets and rip-relative data refs
to Il2CppDumper symbol names and string literals."""
import json, pathlib, pickle, sys
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DLL = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\GameAssembly.dll")
raw = DLL.read_bytes()
pe = pefile.PE(str(DLL), fast_load=True)
BASE = pe.OPTIONAL_HEADER.ImageBase
SYM = pickle.loads(pathlib.Path("data/extracted/symmap.pkl").read_bytes())
_d = json.loads(pathlib.Path("data/extracted/il2cpp/script.json").read_text(encoding="utf-8"))
STR = {int(e["Address"]): e["Value"] for e in _d["ScriptString"]}
del _d

def rva2off(rva):
    for s in pe.sections:
        if s.VirtualAddress <= rva < s.VirtualAddress + max(s.Misc_VirtualSize, s.SizeOfRawData):
            return s.PointerToRawData + (rva - s.VirtualAddress)

def off2rva(off):
    for s in pe.sections:
        if s.PointerToRawData <= off < s.PointerToRawData + s.SizeOfRawData:
            return s.VirtualAddress + (off - s.PointerToRawData)

def name(va):
    """script.json addresses are RVAs, so strip the image base before lookup."""
    rva = va - BASE
    if rva in SYM: return SYM[rva]
    if rva in STR: return f'STR "{STR[rva][:70]}"'
    # metadata slots hold a pointer; follow one level
    off = rva2off(rva)
    if off:
        p = int.from_bytes(raw[off:off + 8], "little") - BASE
        if p in SYM: return f"[{SYM[p]}]"
        if p in STR: return f'[STR "{STR[p][:70]}"]'
    return None

_SYM_RVAS = None

def func_end(rva):
    """Next symbol start above `rva`, so disassembly stops at the function end
    instead of running on into the neighbouring function."""
    global _SYM_RVAS
    if _SYM_RVAS is None:
        import bisect
        _SYM_RVAS = sorted(SYM)
    import bisect
    i = bisect.bisect_right(_SYM_RVAS, rva)
    return _SYM_RVAS[i] if i < len(_SYM_RVAS) else rva + 4096

def go(off, n=200, label=""):
    rva = off2rva(off)
    end = func_end(rva)
    md = Cs(CS_ARCH_X86, CS_MODE_64); md.detail = True
    print(f"\n===== {label}  off={off:#x} va={BASE+rva:#x} size={end-rva} =====")
    for i, ins in enumerate(md.disasm(raw[off:off + min(n * 15, end - rva)], BASE + rva)):
        ann = ""
        if ins.mnemonic in ("call", "jmp") and ins.op_str.startswith("0x"):
            t = int(ins.op_str, 16); ann = f"   ; {name(t) or ''}"
        elif "rip +" in ins.op_str:
            try:
                disp = int(ins.op_str.split("rip +")[1].split("]")[0].strip(), 16)
                t = ins.address + ins.size + disp
                nm = name(t)
                ann = f"   ; {t:#x} {nm or ''}"
            except Exception:
                pass
        print(f"  {ins.address:#x}  {ins.mnemonic:9s} {ins.op_str}{ann}")
        if i >= n: break

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, off = spec.rsplit("=", 1)
        go(int(off, 16), 700, label)
