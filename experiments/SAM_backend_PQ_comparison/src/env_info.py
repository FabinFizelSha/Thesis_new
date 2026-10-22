import json, subprocess, datetime, platform, sys, re
import torch, tensorrt
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e: return str(e)
info = {
    "l4t": sh("head -1 /etc/nv_tegra_release"),
    "jetpack_pkg": sh("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null"),
    "cuda_torch": torch.version.cuda, "cuda_nvcc": sh("nvcc --version 2>/dev/null | tail -1"),
    "tensorrt": tensorrt.__version__, "torch": torch.__version__, "python": sys.version.split()[0],
    "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
    "power_mode": sh("nvpmodel -q 2>/dev/null | tr '\\n' ' '"),
    "jetson_clocks": sh("sudo -n jetson_clocks --show 2>&1 | head -3"),
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_commit": sh("git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null"),
}
json.dump(info, open(sys.argv[1], "w"), indent=2); print(json.dumps(info, indent=2))
