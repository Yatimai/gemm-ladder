"""One GPU session of a rung, on Modal: compute-sanitizer and ncu for each step (the judge runs apart,
through judge/modal_judge.py).

For each step <s> of the rung's folder: verif.cu + <s>.cu are built; compute-sanitizer memcheck, racecheck
and synccheck run ./verif on small shapes with edges in M (an error makes the tool's exit code 9); ncu (--set full,
clocks not locked, demangled names) profiles one call of each variant of the step (g_var, when the step has some) and
one of cuBLASLt at the main shape, in the same
process, on inputs drawn like the judge's; cuBLASLt runs the best configuration the judge found at that shape for the
EXACT card line of --variant (rank 0 of judge/reference/<card>.txt, newest pass first); if the cache has none, the session
refuses before asking for a GPU. All the steps of a session share one container, so their clock-dependent counters
compare. Results land in <rung>/results/<date>-<time>_<steps>/ (<date>-<time>-2_<steps> when two sessions end in the
same second).

Usage (from the repository's root):
  modal run modal/session.py --rung turing --steps step0,ref_simple --variant "<card line>" [--tries N]
--variant (required): as in the judge, the GPU container reads its card line (Tesla_T4|70W|lt130201|cuda13030, for
instance) before any work and stops if it differs; at most N tries, each in a fresh container. 00_run.txt gives the md5 of
this file and of each file sent, the cuBLASLt configuration taken from the judge's cache, and the variant and the try.
"""
import datetime
import hashlib
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CARD = {"turing": ("T4", "sm_75", "2048 2560 2048", "ladder", "t4.txt", "Tesla_T4|70W|"),
        # Ampere: the A100-SXM4-80GB capped at 400 W only; under "A100-80GB" Modal also serves 500 W SXM4 and 300 W PCIe cards:
        # the container checks its card (name and power cap) and stops at once otherwise; main() tries again (--tries).
        "ampere": ("A100", "sm_80", "3456 4096 2048", "ladder", "a100.txt", "NVIDIA_A100-SXM4-80GB|400W|"),
        # Blackwell: the B200 at its 1000 W cap (the judge's card line); sm_100a needs the -gencode form (see GEN below)
        "blackwell": ("B200", "sm_100a", "8192 8192 8192", "ladder", "b200.txt", "NVIDIA_B200|1000W|"),
        # Hopper: the H100 SXM at its 700 W cap only; under "H100" Modal also serves H100 NVL cards at 400 W (checked the same way)
        "hopper": ("H100", "sm_90a", "4608 5632 8192", "ladder", "h100.txt", "NVIDIA_H100_80GB_HBM3|700W|")}
EXPECTED_GPU = {"A100": ("A100-SXM4-80GB", 400.0), "B200": ("B200", 1000.0), "H100": ("H100 80GB HBM3", 700.0)}   # name as nvidia-smi gives it, power cap (W)
# Time limits (s) of the plain run, of each sanitizer tool and of ncu: short on the B200, where a hung kernel costs most.
# The rule, on every card (verif.cu says it too): each run of verif gets GEMM_LADDER_BUDGET = its limit - BUDGET_MARGIN, so
# that verif's HANG diagnostic (shape, variant) always comes before the limit cuts the run, under every tool. The limits of
# the plain run and of the sanitizer tools are the time a correct run had before the budget, plus BUDGET_MARGIN (the
# budget equals that time).
BUDGET_MARGIN = 60
LIMITS = {"T4": (1260, 1860, 2400), "A100": (1260, 1860, 2400), "B200": (360, 360, 1200), "H100": (360, 660, 1200)}
CACHE = REPO / "judge" / "reference"

image = modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
app = modal.App("gemm-ladder-session", image=image)


def card_line() -> str:
    """The judge's card line (as judge/modal_judge.py reads it: the card's name through NVML, the NVML power cap enforced
    in W, the version of cuBLASLt, the driver's CUDA version), read without building anything."""
    import ctypes
    try:
        return _card_line(ctypes)
    except (OSError, AttributeError) as e:
        return f"?({type(e).__name__}: {e})"


def _card_line(ctypes) -> str:
    nv = ctypes.CDLL("libnvidia-ml.so.1")
    h, name, mw = ctypes.c_void_p(), ctypes.create_string_buffer(96), ctypes.c_uint()
    if nv.nvmlInit_v2() or nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) or nv.nvmlDeviceGetName(h, name, 96) \
            or nv.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(mw)):
        return "?"
    try:
        lt = ctypes.CDLL("libcublasLt.so.13")
    except OSError:
        lt = ctypes.CDLL("/usr/local/cuda/lib64/libcublasLt.so.13")
    lt.cublasLtGetVersion.restype = ctypes.c_size_t
    driver = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(driver))
    name = re.sub(r"[ ,|]", "_", name.value.decode())
    return f"{name}|{mw.value // 1000}W|lt{lt.cublasLtGetVersion()}|cuda{driver.value}"


def _session(files: dict, steps: list, arch: str, shape: str, algo: str, expected=None, limits=LIMITS["T4"],
             variant: str = "") -> dict:
    import os
    import subprocess
    if variant:   # the card line, read before any work, as in the judge
        here = card_line()
        if here != variant:
            return {"#variant": f"this container is {here}, expected {variant}; stopped before the session\n"}
    os.makedirs("/tmp/s", exist_ok=True)
    for name, text in files.items():
        (pathlib.Path("/tmp/s") / name).write_text(text)
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    def sh(cmd, t=1200):
        """Runs without raising on a timeout: the output written so far comes back, flagged."""
        try:
            return subprocess.run(cmd, shell=True, cwd="/tmp/s", env=env, capture_output=True, text=True, timeout=t)
        except subprocess.TimeoutExpired as e:
            dec = lambda x: (x.decode(errors="replace") if isinstance(x, bytes) else (x or ""))
            return subprocess.CompletedProcess(cmd, -9, dec(e.stdout), dec(e.stderr) + f"\nTIMEOUT ({t} s)\n")
    out = {"00_machine.txt": sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader; "
                                "nvcc --version | tail -1; ncu --version | tail -1; date -u").stdout}
    if expected:                                                         # card imposed: name and power cap, or nothing is measured
        gpu_name, cap = expected
        q = sh("nvidia-smi --query-gpu=name,power.limit --format=csv,noheader").stdout.strip().split(",")
        ok = len(q) == 2 and gpu_name in q[0] and abs(float(q[1].split()[0]) - cap) < 1.0
        if not ok:
            out["00_card_refused.txt"] = f"card served: {q}; expected: {gpu_name}, {cap} W\n"
            return out
    for s in steps:
        gen = f"-gencode arch=compute_{arch[3:]},code={arch}" if arch.endswith("a") else f"-arch={arch}"   # "a" targets: -gencode
        comp = sh(f"nvcc -O3 -std=c++17 {gen} -lineinfo -Xptxas -v verif.cu {s}.cu -o verif_{s} -lcublasLt")
        log = f"build: rc {comp.returncode}\n{comp.stdout}{comp.stderr}\n"
        if comp.returncode:
            out[f"{s}_01_sanitizer.txt"] = log
            continue
        r = sh(f"GEMM_LADDER_BUDGET={limits[0] - BUDGET_MARGIN} ./verif_{s}", limits[0])
        log += f"== plain run: rc {r.returncode}\n{r.stdout}{r.stderr[-1500:]}"
        for tool in ("memcheck", "racecheck", "synccheck"):   # small shapes: only the step's kernel runs
            r = sh(f"GEMM_LADDER_BUDGET={limits[1] - BUDGET_MARGIN} compute-sanitizer --tool {tool} --error-exitcode 9 ./verif_{s}", limits[1])
            log += f"== {tool}: rc {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-1500:]}\n"
        out[f"{s}_01_sanitizer.txt"] = log
        # --launch-skip 2: the two fill kernels that draw the inputs are not profiled
        r = sh(f"ncu --clock-control none --kernel-name-base demangled --set full --import-source yes --launch-skip 2 "
               f"-o {s}_ncu -f ./verif_{s} {shape} {algo}", limits[2])
        out[f"{s}_02_ncu_run.txt"] = (f"kernels profiled: {r.stdout.count('==PROF== Profiling')}\n" + r.stdout[-4000:] + r.stderr[-2000:])
        if os.path.exists(f"/tmp/s/{s}_ncu.ncu-rep"):
            out[f"{s}_02_ncu.ncu-rep"] = open(f"/tmp/s/{s}_ncu.ncu-rep", "rb").read()
            out[f"{s}_02_ncu_raw.csv"] = sh(f"ncu --import {s}_ncu.ncu-rep --page raw --csv").stdout
    return out


# single_use_containers: each call takes a fresh container (for --variant / --tries, as in the judge)
@app.function(gpu="T4", cpu=4.0, memory=16384, timeout=5400, single_use_containers=True)
def session_t4(files: dict, steps: list, arch: str, shape: str, algo: str, variant: str = "") -> dict:
    return _session(files, steps, arch, shape, algo, None, LIMITS["T4"], variant)


@app.function(gpu="A100-80GB", cpu=4.0, memory=32768, timeout=5400, single_use_containers=True)
def session_a100(files: dict, steps: list, arch: str, shape: str, algo: str, variant: str = "") -> dict:
    return _session(files, steps, arch, shape, algo, EXPECTED_GPU["A100"], LIMITS["A100"], variant)


@app.function(gpu="B200", cpu=4.0, memory=32768, timeout=5400, single_use_containers=True)
def session_b200(files: dict, steps: list, arch: str, shape: str, algo: str, variant: str = "") -> dict:
    return _session(files, steps, arch, shape, algo, EXPECTED_GPU["B200"], LIMITS["B200"], variant)


@app.function(gpu="H100!", cpu=4.0, memory=32768, timeout=5400, single_use_containers=True)
def session_h100(files: dict, steps: list, arch: str, shape: str, algo: str, variant: str = "") -> dict:
    return _session(files, steps, arch, shape, algo, EXPECTED_GPU["H100"], LIMITS["H100"], variant)


def fresh_dir(parent: pathlib.Path, stamp: str, tail: str) -> pathlib.Path:
    """<stamp>_<tail>, or <stamp>-2_<tail>, <stamp>-3_<tail>...: never a folder that exists (two outputs in one second)."""
    k = 1
    while True:
        d = parent / (f"{stamp}_{tail}" if k == 1 else f"{stamp}-{k}_{tail}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            k += 1


@app.local_entrypoint()
def main(rung: str, steps: str, variant: str = "", tries: int = 1):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\|\d+W\|lt\d+\|cuda\d+", variant):
        raise SystemExit("--variant (required): the exact card line of the rung, for instance Tesla_T4|70W|lt130201|cuda13030")
    if not 1 <= tries <= 10:
        raise SystemExit("--tries: an integer from 1 to 10")
    card, arch, shape, shape_name, cache, card_prefix = CARD[rung]
    if not variant.startswith(card_prefix):
        raise SystemExit(f"--variant {variant}: not a card line of the {rung} rung ({card_prefix}...)")
    algo = ""
    for line in (CACHE / cache).read_text().splitlines():   # newest pass first: the first rank-0 line of this card line wins
        m = line.split()
        if len(m) == 9 and m[0] == "REFERENCE" and m[1] == shape_name and m[5] == variant and m[6] == "0":
            algo = m[8]
            algo_line = f"cuBLASLt at {shape_name}: best configuration of {m[5]} (judge's cache): {algo}"
            print(f"cuBLASLt at {shape_name}: best configuration of {m[5]} (judge's cache)")
            break
    if not algo:   # no silent fallback on cuBLASLt's heuristic
        raise SystemExit(f"no rank-0 configuration of {variant} at {shape_name} in judge/reference/{cache}: run the judge's "
                         f"full sort on this card line first; nothing is profiled")
    d = REPO / rung
    files = {p.name: p.read_text() for p in d.iterdir() if p.suffix in (".cu", ".cuh", ".h")}
    names = steps.split(",")
    fn = {"T4": session_t4, "A100": session_a100, "B200": session_b200, "H100": session_h100}[card]
    for attempt in range(1, tries + 1):
        res = fn.remote(files, names, arch, shape, algo, variant)
        if "#variant" in res:
            print(f"try {attempt}/{tries}: ANOTHER VARIANT: {res['#variant'].splitlines()[0]}")
            continue
        if "00_card_refused.txt" not in res:
            break
        print(f"try {attempt}/{tries}: {res['00_card_refused.txt'].strip()}")
    if "#variant" in res:
        raise SystemExit(f"no container of the variant {variant} in {tries} try(ies): nothing is stored")
    res["00_run.txt"] = (f"{pathlib.Path(__file__).name} md5 {hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()[:8]}"
                         f"; variant {variant} (try {attempt}/{tries})\n{algo_line}\n"
                         + "".join(f"{k} md5 {hashlib.md5(v.encode()).hexdigest()}\n" for k, v in sorted(files.items())))
    dest = fresh_dir(d / "results", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), '-'.join(names))
    for name, content in res.items():
        (dest / name).write_bytes(content if isinstance(content, bytes) else content.encode())
    print(f"=== {dest}")
    for name in sorted(res):
        if name.endswith("_01_sanitizer.txt"):
            lines = res[name].splitlines()
            print(name, "|", " ; ".join(l for l in lines if l.startswith(("build", "== ", "OK", "FAILED", "========= ERROR SUMMARY", "========= RACECHECK SUMMARY"))))
