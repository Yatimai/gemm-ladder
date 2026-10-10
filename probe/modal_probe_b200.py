"""modal_probe_b200.py: the interleaved probe of the Blackwell rung on a Modal B200 (probe_b200.cu, next to it), outside the
judge: the correctness, time, clock and ENERGY per GEMM (NVML) of each arm, and ncu counters.
probe_b200.cu is built with a candidate (as the judge builds it) in a container WITHOUT a GPU, of the same image; the GPU
container only measures. Then:
  - energy: ./probe cache.txt M N K arms seconds   for each shape of --shapes;
  - ncu (option --ncu "M,N,K;M,N,K"): one ncu report per arm and per shape (the arm's 2nd call), --set full, or --metrics "a,b".
--guard builds the guard version (-DGEMM_LADDER_GUARD: every mbarrier wait bounded, a trap past it), for a new kernel's first
contact. --containers N runs the same probe in N fresh containers in parallel (probe_c1.txt ... probe_cN.txt), to see the spread
from one container to the next.
Usage (from the repository's root):
  modal run probe/modal_probe_b200.py --candidate blackwell/step1.cu --shapes "128,8192,8192;8192,8192,8192" \\
      --arms "ref0,cand" [--seconds 1.0] [--ncu "8192,8192,8192"] [--limit 120] [--guard] [--containers 3] \\
      --out blackwell/results/<date>_<name>_probe
  --limit: the time limit per shape in s (ncu: 5 x); a hung kernel is cut, the next shape goes on; 0 (default) = automatic,
    60 s + 2 x arms x 4 x seconds.
  --variant "<card line>" [--tries N]: as in the judge, each GPU container reads its card line (NVIDIA_B200|1000W|...)
    before any work and stops if it differs; at most N tries per container, each in a fresh container.
The header of each probe file gives the candidate's path (from the repository's root), the md5 of each file sent, the md5
of probe_b200.cu and of this launcher, and the variant and the try.
"""
import hashlib
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parent   # probe/
REPO = HERE.parent
image = modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
app = modal.App("gemm-ladder-probe-b200", image=image)
GEN = "-gencode arch=compute_100a,code=sm_100a"


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



def _write(files: dict):
    for fname, content in files.items():
        p = pathlib.Path("/tmp/s") / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def fresh_out(o: pathlib.Path) -> pathlib.Path:
    """The output folder, created; if it exists already, <folder>-2, <folder>-3...: an output never overwrites another."""
    k, d = 1, o
    while True:
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            k += 1
            d = o.with_name(f"{o.name}-{k}")


@app.function(cpu=8.0, memory=16384, timeout=1800)
def compiler(files: dict, cand: str, guard: bool) -> dict:
    import os
    import subprocess
    _write(files)
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    g = " -DGEMM_LADDER_GUARD" if guard else ""
    c = subprocess.run(f"nvcc -O3 -std=c++17 {GEN}{g} -Icand probe_b200.cu 'cand/{cand}' -o probe -lcublasLt -lcublas -ldl",
                       shell=True, cwd="/tmp/s", env=env, capture_output=True, text=True, timeout=1500)
    r = {"log": f"build{' (guard version)' if guard else ''} rc {c.returncode}\n{c.stdout[-2000:]}{c.stderr[-3000:]}\n"}
    if c.returncode == 0:
        r["binary"] = open("/tmp/s/probe", "rb").read()
    return r


@app.function(gpu="B200", cpu=4.0, memory=32768, timeout=3600, single_use_containers=True)
def probe(binary: bytes, cache: str, shapes: list, arms: str, seconds: float, ncu: list, metrics: str, limit: float,
          variant: str = "") -> dict:
    import os
    import subprocess
    if variant:   # the card line, read before any work, as in the judge
        here = card_line()
        if here != variant:
            return {"#variant": f"this container is {here}, expected {variant}; stopped before the probe\n"}
    os.makedirs("/tmp/s", exist_ok=True)
    open("/tmp/s/probe", "wb").write(binary)
    os.chmod("/tmp/s/probe", 0o755)
    open("/tmp/s/cache.txt", "w").write(cache)
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    sh = lambda cmd, t: subprocess.run(cmd, shell=True, cwd="/tmp/s", env=env, capture_output=True, text=True, timeout=t)

    def bounded(cmd, t):   # a hung kernel is cut at t s (timeout -k 5), the next shape goes on
        try:
            r = sh(f"timeout -k 5 {t:.0f} {cmd}", t + 30)
            return r.stdout, r.stderr, r.returncode, r.returncode in (124, 137)
        except subprocess.TimeoutExpired as e:
            return (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "", -1, True
    out = {"log": sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv,noheader; date -u", 60).stdout}
    if limit <= 0:
        limit = 60 + 2 * len(arms.split(",")) * 4 * seconds
    out["log"] += f"limit per shape: {limit:.0f} s\n"
    for (m, n, k) in shapes:
        so, se, rc, hung = bounded(f"./probe cache.txt {m} {n} {k} {arms} {seconds}", limit)
        out["log"] += f"== shape {m} {n} {k}\n{so}{se[-1500:]}"
        if hung:
            out["log"] += f"HANG: shape {m} {n} {k} cut at {limit:.0f} s (rc {rc})\n"
    out["reports"] = {}
    for (m, n, k) in ncu:
        for b in arms.split(","):
            report = f"/tmp/s/ncu_{b}_{m}x{n}x{k}"
            what = f"--metrics {metrics}" if metrics else "--set full"
            so, se, rc, hung = bounded(f"ncu --clock-control none {what} --launch-skip 6 --launch-count 6 -f -o {report} "
                                       f"./probe cache.txt {m} {n} {k} {b} 0.05", 5 * limit)
            out["log"] += f"== ncu {b} {m}x{n}x{k} : rc {rc}{' HANG' if hung else ''}\n{so[-600:]}{se[-800:]}"
            if os.path.exists(report + ".ncu-rep"):
                out["reports"][f"{b}_{m}x{n}x{k}"] = open(report + ".ncu-rep", "rb").read()
                out["log"] += sh(f"ncu --import {report}.ncu-rep --page raw --csv", 300).stdout[-30000:]
    out["log"] += sh("date -u", 30).stdout
    return out


@app.local_entrypoint()
def main(candidate: str, shapes: str = "", arms: str = "ref0,cand", seconds: float = 1.0, ncu: str = "", out: str = "", metrics: str = "",
         limit: float = 0.0, guard: bool = False, containers: int = 1, variant: str = "", tries: int = 1):
    if variant and not re.fullmatch(r"[A-Za-z0-9_.-]+\|\d+W\|lt\d+\|cuda\d+", variant):
        raise SystemExit("--variant: the exact card line, for instance NVIDIA_B200|1000W|lt130201|cuda13010")
    if not 1 <= tries <= 10:
        raise SystemExit("--tries: an integer from 1 to 10")
    if tries > 1 and not variant:
        raise SystemExit("--tries goes with --variant only")
    p = pathlib.Path(candidate).resolve()
    files = {"probe_b200.cu": (HERE / "probe_b200.cu").read_text()}
    for v in p.parent.iterdir():
        if v.suffix in (".cuh", ".h", ".hpp") or v == p:
            files[f"cand/{v.name}"] = v.read_text()
    cache = (REPO / "judge" / "reference" / "b200.txt").read_text()
    fl = [[int(x) for x in f.split(",")] for f in shapes.split(";") if f.strip()]
    nc = [[int(x) for x in f.split(",")] for f in ncu.split(";") if f.strip()]
    c = compiler.remote(files, p.name, guard)
    rel = p.relative_to(REPO) if p.is_relative_to(REPO) else p.name
    digests = "; ".join(f"{k[5:]} md5 {hashlib.md5(v.encode()).hexdigest()}" for k, v in files.items() if k.startswith("cand/"))
    tools = (f"probe_b200.cu md5 {hashlib.md5(files['probe_b200.cu'].encode()).hexdigest()[:8]}; "
             f"{pathlib.Path(__file__).name} md5 {hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()[:8]}")
    o = pathlib.Path(out) if out else REPO / "blackwell" / "results" / f"probe_{p.stem}"
    if "binary" not in c:
        o = fresh_out(o)
        (o / "probe.txt").write_text(f"candidate {rel}; {digests}\n{tools}\n" + c["log"])
        print(c["log"][-6000:])
        raise SystemExit("build refused: nothing runs on the B200")
    hs = [probe.spawn(c["binary"], cache, fl, arms, seconds, nc, metrics, limit, variant) for _ in range(max(1, containers))]
    stored = 0
    for i, h in enumerate(hs):
        attempt = 1
        while True:   # without --variant: one pass of this loop, as before
            try:
                r = h.get()
                log = r.get("log", "")
            except Exception as e:   # a lost container does not lose the others
                r, log = {}, f"CONTAINER LOST: {e!r}\n"
            if "#variant" not in r or attempt >= tries:
                break
            print(f"container {i + 1}, try {attempt}/{tries}: ANOTHER VARIANT: {r['#variant'].splitlines()[0]}")
            attempt += 1
            h = probe.spawn(c["binary"], cache, fl, arms, seconds, nc, metrics, limit, variant)
        if "#variant" in r:
            print(f"container {i + 1}: no container of the variant {variant} in {tries} try(ies): nothing is stored for it")
            continue
        if not stored:   # the folder is made by the first output only
            o = fresh_out(o)
        stored += 1
        name = "probe.txt" if containers <= 1 else f"probe_c{i + 1}.txt"
        (o / name).write_text(f"candidate {rel}; {digests}\n{tools}{f'; variant {variant} (try {attempt}/{tries})' if variant else ''}\n"
                              + c["log"] + log)
        for b, raw in r.get("reports", {}).items():
            (o / f"ncu_{b}{'' if containers <= 1 else f'_c{i + 1}'}.ncu-rep").write_bytes(raw)
        print(f"=== container {i + 1}/{max(1, containers)}\n" + log[-4000:])
    if not stored:
        raise SystemExit(f"no container of the variant {variant} in {tries} try(ies), for any of the {len(hs)} container(s): "
                         "nothing is stored")
    print(f"=== -> {o}")
    if stored < len(hs):
        raise SystemExit(f"{len(hs) - stored} of {len(hs)} container(s) without the variant {variant}: their outputs are missing")
