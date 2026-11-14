"""Step 2 - derive Lagrangian dynamics symbolically and generate Python + MATLAB code."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rehab_arm.dynamics_derive import main  # noqa: E402

if __name__ == "__main__":
    main()
