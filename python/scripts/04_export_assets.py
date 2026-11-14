"""Step 4 - export URDF + USD articulation of the arm, then sanity-check them."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rehab_arm.assets import export_all  # noqa: E402

out = export_all()
print("assets written to", out)

try:
    import xml.etree.ElementTree as ET
    ET.parse(out / "rehab_arm.urdf")
    print("URDF parses OK")
    from pxr import Usd, UsdPhysics
    st = Usd.Stage.Open(str(out / "rehab_arm.usda"))
    joints = [p for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    bodies = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    mass = sum(UsdPhysics.MassAPI(b).GetMassAttr().Get() for b in bodies)
    print(f"USD OK: {len(bodies)} rigid bodies, {len(joints)} revolute joints, total mass {mass:.3f} kg")
    for j in joints:
        rj = UsdPhysics.RevoluteJoint(j)
        print(f"  {j.GetName():20s} axis={rj.GetAxisAttr().Get()} limits=[{rj.GetLowerLimitAttr().Get():.0f}, "
              f"{rj.GetUpperLimitAttr().Get():.0f}] deg")
except ImportError:
    pass
_ = np
