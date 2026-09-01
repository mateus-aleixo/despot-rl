"""List every unit prefab (GameObject carrying a V_Unit component) from the bundles."""
import pathlib, sys, collections
import UnityPy
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

B = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data\StreamingAssets\aa\StandaloneWindows64")
env = UnityPy.load(str(B / "7443a426cc273843ed5743fb31c72054.bundle"))

by_pathid = {o.path_id: o for o in env.objects}
units, projectiles = [], []
for obj in env.objects:
    if obj.type.name != "MonoBehaviour":
        continue
    try:
        mb = obj.read(check_read=False)
        s = getattr(mb, "m_Script", None)
        if not s: continue
        cls = s.read().m_ClassName
        if cls not in ("V_Unit", "V_Projectile"): continue
        go = mb.m_GameObject.read()
        (units if cls == "V_Unit" else projectiles).append(go.m_Name)
    except Exception:
        continue
print(f"== unit prefabs ({len(units)}) ==")
for n in sorted(units): print("  ", n)
print(f"\n== projectile prefabs ({len(projectiles)}) ==")
for n in sorted(projectiles): print("  ", n)
