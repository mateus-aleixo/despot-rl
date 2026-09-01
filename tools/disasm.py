"""Disassemble a function in GameAssembly.dll by its Il2CppDumper file offset."""
import sys, pathlib
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

DLL = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\GameAssembly.dll")
raw = DLL.read_bytes()
pe = pefile.PE(str(DLL), fast_load=True)
IMAGE_BASE = pe.OPTIONAL_HEADER.ImageBase

def off_to_rva(off):
    for s in pe.sections:
        if s.PointerToRawData <= off < s.PointerToRawData + s.SizeOfRawData:
            return s.VirtualAddress + (off - s.PointerToRawData)
    return None

def rva_to_off(rva):
    for s in pe.sections:
        if s.VirtualAddress <= rva < s.VirtualAddress + max(s.Misc_VirtualSize, s.SizeOfRawData):
            return s.PointerToRawData + (rva - s.VirtualAddress)
    return None

def disasm(off, n=120, label=""):
    rva = off_to_rva(off)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    print(f"\n===== {label} file_off={off:#x} rva={rva:#x} =====")
    code = raw[off:off + n * 15]
    for i, ins in enumerate(md.disasm(code, IMAGE_BASE + rva)):
        extra = ""
        if "rip" in ins.op_str:
            # resolve rip-relative target
            try:
                tgt = ins.address + ins.size + int(ins.op_str.split("rip +")[1].split("]")[0].strip(), 16)
                toff = rva_to_off(tgt - IMAGE_BASE)
                if toff:
                    extra = f"   ; -> {tgt:#x} off={toff:#x} bytes={raw[toff:toff+16].hex()}"
            except Exception:
                pass
        print(f"  {ins.address:#x}  {ins.mnemonic:8s} {ins.op_str}{extra}")
        if i >= n or ins.mnemonic == "ret":
            break

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, off = spec.split("=")
        disasm(int(off, 16), 90, label)
