"""Generate isaac/notebooks/isaac_sim_kaggle.ipynb (keeps the notebook source reviewable)."""
import json
import pathlib

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 3-DOF Rehab Arm - Isaac Sim Digital Twin (free cloud GPU)

Runs **headless Isaac Sim** on a free **Kaggle T4 GPU** (also works on Google Colab with minor path changes).

**Before running** (Kaggle notebook right panel → *Settings*):
1. **Accelerator → GPU T4** (x1 or x2)
2. **Internet → On** (requires a phone-verified Kaggle account)
3. Set `REPO_URL` below to your GitHub repo, then *Run All*.

First run downloads Isaac Sim (~15 GB) and compiles shaders: expect **20-40 min**. Outputs (video, CSV, plots,
synthetic dataset) are zipped into `/kaggle/working/twin_outputs.zip` (download from the *Output* tab).

> Setting `OMNI_KIT_ACCEPT_EULA=YES` means you accept the NVIDIA Omniverse / Isaac Sim license.
""")

code(r"""
# ---------------- user settings ----------------
REPO_URL      = "https://github.com/<your-username>/rehab-arm-3dof.git"   # <-- change me
ISAAC_VERSION = "5.1.0"          # pip version of isaacsim (Python 3.11)
SDG_FRAMES    = 300              # Replicator frames
TWIN_DURATION = 26.0             # seconds of safety demo
import os, pathlib
IS_KAGGLE = os.path.exists("/kaggle")
WORK = pathlib.Path("/kaggle/working" if IS_KAGGLE else "/content")
REPO = pathlib.Path("/root/rehab_arm_3dof")      # outside /kaggle/working (20 GB limit)
VENV = pathlib.Path("/root/isaac_env")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["ACCEPT_EULA"] = "Y"
print(WORK, REPO, VENV)
""")

md("## 1. System & network pre-flight check (GPU, Internet, glibc, RAM, disk)")
code(r"""
import shutil, subprocess, urllib.request

if not shutil.which("nvidia-smi"):
    raise RuntimeError(
        "No NVIDIA GPU detected! In Kaggle, open the right-hand 'Session options' panel -> "
        "Accelerator -> select 'GPU T4 x1' (or 'T4 x2')."
    )
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True
)
if gpu.returncode != 0 or not gpu.stdout.strip():
    raise RuntimeError(f"nvidia-smi failed ({gpu.stderr.strip()}). Enable GPU T4 in Session options.")
print("GPU:", gpu.stdout.strip())

try:
    urllib.request.urlopen("https://pypi.org", timeout=8).close()
    print("Internet: OK (reachable)")
except Exception as exc:
    raise RuntimeError(
        "No Internet access detected! In Kaggle:\n"
        "  1. Verify your phone number under Account Settings.\n"
        "  2. Open the right-hand 'Session options' panel -> toggle 'Internet' ON.\n"
        f"Details: {exc}"
    ) from exc

!ldd --version | head -1
!python3 --version
!free -h | head -2
!df -h / | tail -1
""")

md(r"""
## 2. Vulkan / RTX user-space driver
Isaac Sim's RTX renderer needs the NVIDIA **Vulkan ICD** and **OptiX** libraries. Cloud notebook containers
usually ship only the CUDA part of the driver, so we install `libvulkan1` + `libnvidia-gl-<major>` via `apt`
and extract the full matching user-space libraries from the official NVIDIA driver package (same version as the
host kernel driver; no kernel module is installed).
""")
code(r"""
import subprocess

bash_script = r'''
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
if [ -z "$DRV" ]; then
  echo "ERROR: Could not read NVIDIA driver version from nvidia-smi." >&2
  exit 1
fi
DRV_MAJOR=${DRV%%.*}
echo "Host driver: $DRV (major: $DRV_MAJOR)"

export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
apt-get -qq install -y libvulkan1 vulkan-tools libglu1-mesa libxt6 libegl1 curl
apt-get -qq install -y "libnvidia-gl-${DRV_MAJOR}" || echo "Note: libnvidia-gl-${DRV_MAJOR} unavailable via apt; will rely on .run extraction"

LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -e "$LIBDIR/libGLX_nvidia.so.$DRV" ] && [ -e "$LIBDIR/libnvoptix.so.$DRV" ] && ls /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  echo "Vulkan/OptiX user-space driver already present"
else
  cd /tmp
  RUN="NVIDIA-Linux-x86_64-${DRV}.run"
  if [ ! -s "$RUN" ]; then
    for URL in \
      "https://download.nvidia.com/XFree86/Linux-x86_64/${DRV}/${RUN}" \
      "https://us.download.nvidia.com/tesla/${DRV}/${RUN}" \
      "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRV}/${RUN}"; do
      echo "Trying $URL ..."
      if curl -fSL --retry 3 --connect-timeout 15 -o "$RUN" "$URL" && [ -s "$RUN" ]; then
        echo "Downloaded $RUN"
        break
      fi
      rm -f "$RUN"
    done
  fi

  if [ -s "$RUN" ]; then
    rm -rf "NVIDIA-Linux-x86_64-${DRV}"
    sh "$RUN" -x >/dev/null
    cd "NVIDIA-Linux-x86_64-${DRV}"
    for lib in libGLX_nvidia libEGL_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-glvkspirv \
               libnvidia-tls libnvidia-rtcore libnvoptix libnvidia-gpucomp libnvidia-allocator libnvidia-cbl \
               libnvidia-ptxjitcompiler libnvidia-nvvm; do
      for f in ${lib}.so.${DRV}; do
        [ -f "$f" ] && [ ! -e "$LIBDIR/$f" ] && cp "$f" "$LIBDIR/" && echo "installed $f"
      done
    done
    ln -sf "libGLX_nvidia.so.${DRV}" "$LIBDIR/libGLX_nvidia.so.0"
    ln -sf "libEGL_nvidia.so.${DRV}" "$LIBDIR/libEGL_nvidia.so.0"
    [ -e "$LIBDIR/libnvoptix.so.${DRV}" ] && ln -sf "libnvoptix.so.${DRV}" "$LIBDIR/libnvoptix.so.1" || true
    [ -f "libnvidia-nvvm.so.${DRV}" ] && ln -sf "libnvidia-nvvm.so.${DRV}" "$LIBDIR/libnvidia-nvvm.so.4" || true
    mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
    ICD=$(ls nvidia_icd.json* 2>/dev/null | head -1)
    if [ -n "$ICD" ]; then
      sed 's/__NV_VK_ICD__/libGLX_nvidia.so.0/' "$ICD" > /usr/share/vulkan/icd.d/nvidia_icd.json
    fi
    [ -f 10_nvidia.json ] && cp 10_nvidia.json /usr/share/glvnd/egl_vendor.d/ || true
    ldconfig
  else
    echo "WARNING: Could not download $RUN from NVIDIA mirrors; using apt-installed Vulkan libraries if present."
    ldconfig
  fi
fi

vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion" || {
  echo "ERROR: vulkaninfo still sees no GPU. Check /usr/share/vulkan/icd.d and driver version $DRV." >&2
  exit 1
}
'''
subprocess.run(["bash", "-c", bash_script], check=True)
""")

md("## 3. Get the project code")
code(r"""
import subprocess
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull"], check=False)
!ls {REPO}
""")

md("## 4. Install Isaac Sim (pip, isolated Python 3.11 venv via `uv`)")
code(r"""
%%bash -s "$VENV" "$ISAAC_VERSION"
set -e
VENV=$1; VER=$2
command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null)
export PATH=$HOME/.local/bin:$PATH
[ -x $VENV/bin/python ] || uv venv $VENV --python 3.11 --seed
uv pip install --python $VENV/bin/python --index-strategy unsafe-best-match \
   --extra-index-url https://pypi.nvidia.com "isaacsim[all,extscache]==$VER" 2>&1 | tail -3
uv pip install --python $VENV/bin/python "imageio[ffmpeg]" pillow matplotlib 2>&1 | tail -1
$VENV/bin/python -c "import isaacsim, numpy; print('isaacsim OK, numpy', numpy.__version__)"
""")

code(r"""
# helper: run a script in the Isaac venv, stream the important lines, keep the full log
import subprocess, sys, time
def run_isaac(script, *args, log_name=None, keys=("[twin]", "[sdg]", "[smoke]", "Error", "error:", "Traceback", "Exception")):
    log_path = WORK / (log_name or (pathlib.Path(script).stem + ".log"))
    cmd = [str(VENV / "bin/python"), str(REPO / script), *map(str, args)]
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            if any(k in line for k in keys):
                print(line.rstrip()[:300])
        p.wait()
    print(f"--> exit code {p.returncode} after {time.time()-t0:.0f}s, full log: {log_path}")
    return p.returncode
""")

md("## 5. Smoke test (first launch compiles shaders - be patient)")
code(r"""
smoke = REPO / "isaac/scripts/_smoke.py"
smoke.write_text('''
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb
print("[smoke] Isaac Sim started, renderer:", carb.settings.get_settings().get("/renderer/active"))
app.close()
''')
run_isaac("isaac/scripts/_smoke.py")
""")

md("## 6. Digital twin: USD import, twin validation, proximity-based safety")
code(r"""
rc = run_isaac("isaac/scripts/safety_twin.py", "--duration", TWIN_DURATION)
!{sys.executable} {REPO}/isaac/scripts/plot_twin_results.py {REPO}/isaac/outputs/safety
!cat {REPO}/isaac/outputs/safety/summary.json
""")
code(r"""
from IPython.display import Image, Video, display
display(Image(str(REPO / "isaac/outputs/safety/safety_timeline.png"), width=900))
mp4 = REPO / "isaac/outputs/safety/safety_demo.mp4"
if mp4.exists():
    import shutil; shutil.copy(mp4, WORK / "safety_demo.mp4")
    display(Video(str(WORK / "safety_demo.mp4"), embed=True, width=800))
""")

md("## 7. Replicator synthetic data generation")
code(r"""
rc = run_isaac("isaac/scripts/replicator_sdg.py", "--frames", SDG_FRAMES)
!cat {REPO}/isaac/outputs/sdg/dataset_info.json
!ls {REPO}/isaac/outputs/sdg | head -20
""")
code(r"""
import glob, matplotlib.pyplot as plt, matplotlib.image as mpimg
sdg = REPO / "isaac/outputs/sdg"
rgbs = sorted(glob.glob(str(sdg / "**/rgb_*.png"), recursive=True))[:4]
segs = sorted(glob.glob(str(sdg / "**/semantic_segmentation_*.png"), recursive=True))[:4]
fig, axs = plt.subplots(2, max(1, len(rgbs)), figsize=(4 * max(1, len(rgbs)), 6), squeeze=False)
for i, (a, b) in enumerate(zip(rgbs, segs)):
    axs[0, i].imshow(mpimg.imread(a)); axs[0, i].axis("off")
    axs[1, i].imshow(mpimg.imread(b)); axs[1, i].axis("off")
plt.tight_layout(); plt.savefig(sdg / "preview.png", dpi=100); plt.show()
""")

md("## 8. Package outputs for download")
code(r"""
import shutil
for f in WORK.glob("*.log"):
    shutil.copy(f, REPO / "isaac/outputs")
shutil.make_archive(str(WORK / "twin_outputs"), "zip", REPO / "isaac/outputs")
print("Download:", WORK / "twin_outputs.zip")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "isaac_sim_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
