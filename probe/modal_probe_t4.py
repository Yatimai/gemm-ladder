"""modal_probe_t4.py: the interleaved probe on a Modal T4 (probe_t4.cu, next to it), outside the judge: the time, the clock
and the ENERGY per GEMM (NVML) of each arm, and ncu counters.
Builds probe_t4.cu with a candidate (as the judge builds it), then:
  - energy: ./probe M N K energy arms,... seconds rounds   for each shape of --shapes;
  - ncu (option --ncu "M,N,K;M,N,K"): one ncu report per arm and per shape (the arm's 2nd call); --set full, or
    --metrics "a,b" (a list for ncu --metrics); the raw CSV of each report is appended to probe.txt.
Usage (from the repository's root):
  modal run probe/modal_probe_t4.py --candidate turing/step6.cu --shapes "2048,6144,4096;2048,2560,2048" \
      --arms "lt,v0,v1" [--seconds 1.0] [--rounds 5] [--ncu "2048,6144,4096"] [--limit 120] --out turing/results/<date>_<name>_probe
  --limit: the time limit per shape in s (ncu: 5 x); a hung kernel is cut, the next shape goes on;
    0 (default) = automatic, 60 s + 2 x arms x (rounds + 1) x seconds ('lt' counts as 4 arms).
  --variant "<card line>" [--tries N]: as in the judge, the GPU container reads its card line (Tesla_T4|70W|lt...|cuda...)
    before any work and stops if it differs; at most N tries, each in a fresh container.
The header of probe.txt gives the candidate's path (from the repository's root), the md5 of each file sent, the md5 of
probe_t4.cu and of this launcher, and the variant and the try.
"""
import hashlib
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parent   # probe/
REPO = HERE.parent
image = modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
app = modal.App("gemm-ladder-probe-t4", image=image)


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


# single_use_containers: each call takes a fresh container (for --variant / --tries, as in the judge)
@app.function(gpu="T4", cpu=8.0, memory=32768, timeout=3600, single_use_containers=True)
def probe(files: dict, cand: str, shapes: list, arms: str, seconds: float, ncu: list, metrics: str = "", rounds: int = 3,
           limit: float = 0.0, variant: str = "") -> dict:
    import os
    import subprocess
    if variant:   # the card line, read before any work, as in the judge
        here = card_line()
        if here != variant:
            return {"#variant": f"this container is {here}, expected {variant}; stopped before the build\n"}
    os.makedirs("/tmp/s", exist_ok=True)
    for name, content in files.items():
        p = pathlib.Path("/tmp/s") / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    sh = lambda cmd, t: subprocess.run(cmd, shell=True, cwd="/tmp/s", env=env, capture_output=True, text=True, timeout=t)

    def bounded(cmd, t):   # a hung kernel is cut at t s (timeout -k 5), the next shape goes on
        try:
            r = sh(f"timeout -k 5 {t:.0f} {cmd}", t + 30)
            hung = r.returncode in (124, 137)
            return r.stdout, r.stderr, r.returncode, hung
        except subprocess.TimeoutExpired as e:
            return (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "", -1, True
    out = {"log": sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv,noheader", 60).stdout}
    c = sh(f"nvcc -O3 -std=c++17 -arch=sm_75 probe_t4.cu 'cand/{cand}' -o probe -lcublasLt -lcublas -ldl", 1200)
    out["log"] += f"build rc {c.returncode}\n{c.stdout[-2000:]}{c.stderr[-2000:]}\n"
    if c.returncode:
        return out
    if limit <= 0:   # automatic: 60 s + 2 x (arms, 'lt' = 4) x (rounds + 1) x seconds
        nb = sum(4 if b == "lt" else 1 for b in arms.split(","))
        limit = 60 + 2 * nb * (rounds + 1) * seconds
    out["log"] += f"limit per shape: {limit:.0f} s\n"
    for (m, n, k) in shapes:
        so, se, rc, hung = bounded(f"./probe {m} {n} {k} energy {arms} {seconds} {rounds}", limit)
        out["log"] += f"== energy {m} {n} {k}\n{so}{se[-1000:]}"
        if hung:
            out["log"] += f"HANG: shape {m} {n} {k} cut at {limit:.0f} s (rc {rc})\n"
    out["reports"] = {}
    if ncu:
        for (m, n, k) in ncu:
            for b in arms.split(","):
                report = f"/tmp/s/ncu_{b}_{m}x{n}x{k}"
                what = f"--metrics {metrics}" if metrics else "--set full"
                so, se, rc, hung = bounded(f"ncu --clock-control none {what} --kernel-name regex:'gemm|Kernel|cutlass|probe_read' --launch-skip 1 "
                                           f"--launch-count 1 -f -o {report} ./probe {m} {n} {k} ncu {b}", 5 * limit)   # ncu replays: 5 x
                out["log"] += f"== ncu {b} {m}x{n}x{k} : rc {rc}{' HANG' if hung else ''}\n{so[-800:]}{se[-800:]}"
                if os.path.exists(report + ".ncu-rep"):
                    out["reports"][f"{b}_{m}x{n}x{k}"] = open(report + ".ncu-rep", "rb").read()
                    q = sh(f"ncu --import {report}.ncu-rep --page raw --csv", 300)
                    out["log"] += q.stdout[-6000:]
    return out


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


@app.local_entrypoint()
def main(candidate: str, shapes: str = "", arms: str = "lt,cand", seconds: float = 1.0, ncu: str = "", out: str = "", metrics: str = "", rounds: int = 3,
         limit: float = 0.0, variant: str = "", tries: int = 1):
    if variant and not re.fullmatch(r"[A-Za-z0-9_.-]+\|\d+W\|lt\d+\|cuda\d+", variant):
        raise SystemExit("--variant: the exact card line, for instance Tesla_T4|70W|lt130201|cuda13030")
    if not 1 <= tries <= 10:
        raise SystemExit("--tries: an integer from 1 to 10")
    if tries > 1 and not variant:
        raise SystemExit("--tries goes with --variant only")
    p = pathlib.Path(candidate).resolve()
    files = {"probe_t4.cu": (HERE / "probe_t4.cu").read_text()}
    for v in p.parent.iterdir():
        if v.suffix in (".cuh", ".h", ".hpp") or v == p:
            files[f"cand/{v.name}"] = v.read_text()
    fl = [[int(x) for x in f.split(",")] for f in shapes.split(";") if f.strip()]
    nc = [[int(x) for x in f.split(",")] for f in ncu.split(";") if f.strip()]
    for attempt in range(1, tries + 1):
        r = probe.remote(files, p.name, fl, arms, seconds, nc, metrics, rounds, limit, variant)
        if "#variant" not in r:
            break
        print(f"try {attempt}/{tries}: ANOTHER VARIANT: {r['#variant'].splitlines()[0]}")
    else:
        raise SystemExit(f"no container of the variant {variant} in {tries} try(ies): nothing is stored")
    o = pathlib.Path(out) if out else REPO / "turing" / "results" / f"probe_{p.stem}"
    o = fresh_out(o)
    rel = p.relative_to(REPO) if p.is_relative_to(REPO) else p.name
    digests = "; ".join(f"{k[5:]} md5 {hashlib.md5(v.encode()).hexdigest()}" for k, v in files.items() if k.startswith("cand/"))
    tools = (f"probe_t4.cu md5 {hashlib.md5(files['probe_t4.cu'].encode()).hexdigest()[:8]}; "
             f"{pathlib.Path(__file__).name} md5 {hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()[:8]}")
    (o / "probe.txt").write_text(f"candidate {rel}; {digests}\n{tools}{f'; variant {variant} (try {attempt}/{tries})' if variant else ''}\n"
                                 + r["log"])
    for b, raw in r.get("reports", {}).items():
        (o / f"ncu_{b}.ncu-rep").write_bytes(raw)
    print(r["log"][-6000:])
