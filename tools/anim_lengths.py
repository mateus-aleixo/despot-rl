"""Extract animation clip lengths and each unit's animator overrides.

Attack pacing in the behaviour trees is driven by WaitForAnimation, so the real
timings are these clip lengths, not a constant.
"""
import json, pathlib, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

A = pathlib.Path("data/extracted/ripped/ExportedProject/Assets")
OUT = pathlib.Path("data/extracted")

clips = {}
guid_to_clip = {}
for f in sorted((A / "AnimationClip").glob("*.anim")):
    t = f.read_text(encoding="utf-8", errors="replace")
    name = re.search(r"^\s*m_Name:\s*(.+)$", t, re.M)
    stop = re.search(r"m_StopTime:\s*([0-9.eE+-]+)", t)
    if name and stop:
        clips[name.group(1).strip()] = float(stop.group(1))
    meta = f.with_suffix(".anim.meta")
    if meta.exists():
        g = re.search(r"guid:\s*([a-f0-9]{32})", meta.read_text(encoding="utf-8", errors="replace"))
        if g and name:
            guid_to_clip[g.group(1)] = name.group(1).strip()

print(f"{len(clips)} clips")
(OUT / "anim_lengths.json").write_text(json.dumps(clips, indent=1, sort_keys=True), encoding="utf-8")

pat = re.compile(r"attack|recovery|shoot|cast|swing|hit", re.I)
sample = {k: v for k, v in sorted(clips.items()) if pat.search(k)}
print(f"{len(sample)} attack/recovery-ish clips; sample:")
for k, v in list(sample.items())[:18]:
    print(f"  {v:6.3f}s  {k}")

# Per-unit animator overrides: base clip -> unit clip
overrides = {}
for f in sorted((A / "AnimatorOverrideController").glob("*.overrideController")):
    t = f.read_text(encoding="utf-8", errors="replace")
    pairs = re.findall(r"m_OriginalClip:\s*\{fileID:\s*[0-9-]+,\s*guid:\s*([a-f0-9]{32}).*?"
                       r"m_OverrideClip:\s*\{fileID:\s*[0-9-]+,\s*guid:\s*([a-f0-9]{32})", t, re.S)
    if pairs:
        overrides[f.stem] = {guid_to_clip.get(a, a): guid_to_clip.get(b, b) for a, b in pairs}
print(f"\n{len(overrides)} animator override controllers")
for name, m in list(overrides.items())[:3]:
    print(f"  {name}: {len(m)} overrides, e.g. {list(m.items())[:3]}")
(OUT / "anim_overrides.json").write_text(json.dumps(overrides, indent=1, sort_keys=True), encoding="utf-8")
