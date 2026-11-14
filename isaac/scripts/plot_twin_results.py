"""Plot the Isaac Sim digital-twin safety log (runs with any Python + matplotlib).

    python isaac/scripts/plot_twin_results.py [path/to/outputs/safety]
"""
import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = pathlib.Path(__file__).resolve()
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here.parents[1] / "outputs" / "safety"
cfg = json.loads((here.parents[2] / "config" / "arm_params.json").read_text())["safety"]

with open(OUT / "twin_log.csv") as f:
    rows = list(csv.DictReader(f))
t = np.array([float(r["t"]) for r in rows])
d = np.array([float(r["d_min"]) for r in rows])
a = np.array([float(r["alpha"]) for r in rows])
st = np.array([r["state"] for r in rows])
q = np.array([[float(r[f"q{i}"]) for i in (1, 2, 3)] for r in rows])
qr = np.array([[float(r[f"q{i}_ref"]) for i in (1, 2, 3)] for r in rows])
err = np.array([float(r["ee_twin_err_m"]) for r in rows]) * 1000

fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
colors = {"RUN": "#c8f0c8", "SLOW": "#fbe7a1", "STOP": "#f6b3b3"}
for ax in axs:
    start = 0
    for k in range(1, len(t) + 1):
        if k == len(t) or st[k] != st[start]:
            ax.axvspan(t[start], t[min(k, len(t) - 1)], color=colors[st[start]], lw=0)
            start = k
axs[0].plot(t, d * 100, "k")
for key, ls in [("d_slow_m", "--"), ("d_resume_m", ":"), ("d_stop_m", "-.")]:
    axs[0].axhline(cfg[key] * 100, color="r", ls=ls, label=key)
axs[0].set_ylabel("min separation [cm]"); axs[0].legend(loc="upper right"); axs[0].set_ylim(bottom=0)
axs[0].set_title("Isaac Sim digital twin - speed & separation monitoring (green RUN / yellow SLOW / red STOP)")
axs[1].plot(t, a, "b"); axs[1].set_ylabel("speed scale alpha")
for j, c in enumerate("rgb"):
    axs[2].plot(t, np.rad2deg(q[:, j]), c, label=f"q{j + 1}")
    axs[2].plot(t, np.rad2deg(qr[:, j]), c + "--", lw=0.8)
axs[2].set_ylabel("joint angle [deg]"); axs[2].legend(loc="upper right")
axs[3].plot(t, err, "m"); axs[3].set_ylabel("|p_isaac - p_FK| [mm]"); axs[3].set_xlabel("t [s]")
fig.tight_layout()
fig.savefig(OUT / "safety_timeline.png", dpi=150)
print("saved", OUT / "safety_timeline.png")
