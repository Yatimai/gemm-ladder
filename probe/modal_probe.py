"""modal_probe.py: the interleaved probe (probe.cu, next to it) on a Modal GPU of the card given by --card, outside the
judge. The candidate's folder is sent with it (its .cuh, .h, .hpp; of its .cu, the candidate alone); the card is checked
(--variant, --tries) as in the judge.
  - energy: ./probe M N K energy arms,... seconds rounds   for each shape of --shapes, under `timeout -k 5 <limit>`:
    after the limit, the probe ITSELF is killed, the shape is marked HANG, and the next shape goes on; the probe's
    stdout is line-buffered (stdbuf -oL), so that a crash keeps what it wrote. Each shape's log gives its arms (the
    hexadecimal of an x: arm included), its limit and its return code;
  - ncu (option --ncu "M,N,K;..."): first a try of ncu on the probe's first fill kernel, without a list of arms (NCU
    UNAVAILABLE and a stop if it fails); then one report per arm and per shape, the probe's two fill kernels skipped and
    the kernels of the arm's two calls profiled, 6 at most (--launch-skip 2 --launch-count 6): the arm's second call is
    ID 1 of the report, or IDs 2 and 3 when a call launches two kernels; --set full, or --metrics "a,b"; under the same
    kind of limit.
Arms (checked before any GPU is asked for): v<i> (the candidate's variants, g_var), twin (v0 timed again, as an arm of
its own: at most one per list, with v0 in the list, never under ncu) and x:<128 hexadecimal digits> (the exact
configuration of a REFERENCE line, at most one per shape; printed "ref0").
Usage (from the repository's root):
  modal run probe/modal_probe.py --card T4 --candidate turing/step6.cu \
      --shapes "2048,2560,2048;..." --arms "v0,twin,v1,v2" [--seconds 1.0 --rounds 7 --limit 0] [--ncu "..."] \
      --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4 --out turing/results/<date>_<name>_probe
  --card: T4, A100-80GB, H100! or B200, the cards of judge/modal_judge.py: the GPU asked for, nvcc's target (ARCH and
    GEN, as the judge builds) and the default output folder (<rung>/results/probe_<step>).
  --limit: the time limit per shape in s; 0 (the default) = automatic: 90 s + twice the nominal time (arms x rounds x
    4/3 x seconds).
  --variant "<card line>" [--tries N]: as in the judge, the GPU container reads its card line before any work and stops if
    it differs (the whole line, the driver included); at most N tries, each in a fresh container.
The header of probe.txt gives the candidate's path (from the repository's root) and the md5 of each of its files sent
(line 1), the md5 of probe.cu and of this launcher, the variant and the try (line 2), then the card line read in the
container (line 3).
The launcher ends with a non-zero code when the build, ncu or a shape's probe (a HANG, a crash, a refused arm) fails,
after writing probe.txt. The whole plan of a launch (build, every shape's limit, ncu) must fit in the GPU function's
time limit (6 h) less 10 minutes, or it is refused before any GPU is asked for.
"""
import hashlib
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent   # the repository: this file is probe/modal_probe.py
image = modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
app = modal.App("gemm-ladder-probe", image=image)
FUNCTION_LIMIT = 6 * 3600   # s, the GPU function's time limit (as the judge's)
NCU_LIMIT, NCU_TRY_LIMIT, BUILD_LIMIT = 1800, 600, 1200
# The cards, nvcc's target of each and its form, and the rung of each card: as in judge/modal_judge.py.
ARCH = {"T4": "sm_75", "A100-80GB": "sm_80", "H100!": "sm_90a", "B200": "sm_100a"}
GEN = {c: (f"-gencode arch=compute_{a[3:]},code={a}" if a.endswith("a") else f"-arch={a}") for c, a in ARCH.items()}
RUNG = {"T4": "turing", "A100-80GB": "ampere", "H100!": "hopper", "B200": "blackwell"}   # where the outputs go
# The arms the launcher accepts (v<i>: 1 to 3 digits 0-9; twin; x:<hex>); any other is refused
# before any GPU is asked for.
ARM = re.compile(r"v[0-9]{1,3}|twin|x:[0-9a-f]{128}")


def card_line() -> str:
    """The judge's card line (as judge/modal_judge.py reads it): name, NVML power cap, cuBLASLt, driver."""
    import ctypes
    try:
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
        return f"{re.sub(r'[ ,|]', '_', name.value.decode())}|{mw.value // 1000}W|lt{lt.cublasLtGetVersion()}|cuda{driver.value}"
    except (OSError, AttributeError) as e:
        return f"?({type(e).__name__}: {e})"


def bounded(cmd: str, t: float, cwd: str, env: dict):
    """Runs cmd under `timeout -k 5 t`: after t s, `timeout` kills the command itself (SIGTERM, then SIGKILL 5 s later),
    not only the shell that started it. Returns (stdout, stderr, return code, hung)."""
    import subprocess
    try:
        r = subprocess.run(f"timeout -k 5 {t:.0f} {cmd}", shell=True, cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=t + 30)
        return r.stdout, r.stderr, r.returncode, r.returncode in (124, 137)
    except subprocess.TimeoutExpired as e:
        dec = lambda x: (x.decode(errors="replace") if isinstance(x, bytes) else (x or ""))
        return dec(e.stdout), dec(e.stderr), -1, True


def shape_limit(arms: str, seconds: float, rounds: int, limit: float) -> float:
    """The time limit of one shape: --limit, or (0) 90 s + twice the nominal time (arms x rounds x 4/3 x seconds)."""
    return limit or 90 + 2 * len(arms.split(",")) * rounds * seconds * 4 / 3


def plan_seconds(fl: list, nc: list, arms: str, ncu_arms: str, seconds: float, rounds: int, limit: float) -> float:
    """The plan of a launch at the worst: the build, every shape's limit, the ncu try and every ncu report, each + 30 s."""
    return (BUILD_LIMIT + 30 + sum(shape_limit(arms + ("," + f[3] if f[3] else ""), seconds, rounds, limit) + 30 for f in fl)
            + (NCU_TRY_LIMIT + 30 if nc else 0) + sum((NCU_LIMIT + 30) * len((f[3] or ncu_arms or arms).split(",")) for f in nc))


def build_command(card: str, cand: str) -> str:
    """The build of the probe with the candidate, for the card."""
    return f"nvcc -O3 -std=c++17 {GEN[card]} probe.cu 'cand/{cand}' -o probe -lcublasLt -lcublas -ldl"


# single_use_containers: each call takes a fresh container (for --variant / --tries, as in the judge); the GPU is the
# card's, given in main (probe.with_options), as the judge does
@app.function(cpu=8.0, memory=32768, timeout=FUNCTION_LIMIT, single_use_containers=True)
def probe(files: dict, cand: str, card: str, shapes: list, arms: str, seconds: float, rounds: int, limit: float, ncu: list,
           ncu_arms: str, metrics: str = "", variant: str = "") -> dict:
    import os
    if variant:
        here = card_line()
        if here != variant:   # the whole card line, as in the judge
            return {"log": f"VARIANT: this container is {here}, expected {variant}; stopped before the build\n", "#variant": here}
    os.makedirs("/tmp/s", exist_ok=True)
    for name, content in files.items():
        p = pathlib.Path("/tmp/s") / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    so, _, _, _ = bounded("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv,noheader; date -u",
                          60, "/tmp/s", env)
    out = {"log": f"card {card_line()}\n" + so, "failed": []}
    so, se, rc, _ = bounded(build_command(card, cand), BUILD_LIMIT, "/tmp/s", env)
    out["log"] += f"build rc {rc}\n{so[-2000:]}{se[-2000:]}\n"
    if rc:
        out["failed"].append(f"build rc {rc}")
        return out
    for (m, n, k, extra) in shapes:
        b = arms + ("," + extra if extra else "")
        t = shape_limit(b, seconds, rounds, limit)
        so, se, rc, hung = bounded(f"stdbuf -oL ./probe {m} {n} {k} energy {b} {seconds} {rounds}", t, "/tmp/s", env)
        out["log"] += f"== energy {m} {n} {k} arms {b} (limit {t:.0f} s): rc {rc}\n{so}{se[-1000:]}"
        if hung:
            out["log"] += f"HANG: shape {m} {n} {k} cut at {t:.0f} s (rc {rc}): the probe was killed\n"
        if rc:
            out["failed"].append(f"shape {m} {n} {k}: rc {rc}{' (HANG)' if hung else ''}")
    out["reports"] = {}
    if ncu:                                                               # is ncu available on this card?
        m0, n0, k0, _ = ncu[0]
        # the probe without a list of arms: its fill kernels alone (the first one profiled), no arm runs in the try
        so, se, rc, hung = bounded(f"ncu --clock-control none --metrics gpu__time_duration.sum --launch-count 1 ./probe {m0} {n0} {k0} ncu",
                                   NCU_TRY_LIMIT, "/tmp/s", env)
        out["log"] += f"== ncu try: rc {rc}\n{so[-600:]}{se[-600:]}"
        if rc != 0 or "ERROR" in so + se:
            out["log"] += "NCU UNAVAILABLE on this card: stop\n"
            out["failed"].append("ncu unavailable")
            return out
    for (m, n, k, extra) in ncu:
        for b in (extra or ncu_arms or arms).split(","):
            report = f"/tmp/s/ncu_{b.replace(':', '_')}_{m}x{n}x{k}"
            what = f"--metrics {metrics}" if metrics else "--set full"
            so, se, rc, hung = bounded(f"ncu --clock-control none {what} --launch-skip 2 --launch-count 6 -f -o {report} ./probe {m} {n} {k} ncu {b}",
                                       NCU_LIMIT, "/tmp/s", env)
            out["log"] += f"== ncu {b} {m}x{n}x{k} : rc {rc}{' HANG' if hung else ''}\n{so[-800:]}{se[-800:]}"
            if rc:
                out["failed"].append(f"ncu {b} {m}x{n}x{k}: rc {rc}{' (HANG)' if hung else ''}")
            if os.path.exists(report + ".ncu-rep"):
                out["reports"][f"{b.replace(':', '_')}_{m}x{n}x{k}"] = open(report + ".ncu-rep", "rb").read()
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


def check_arms(arms: str, where: str, ncu: bool = False):
    """Refuses, before any GPU is asked for, an arm other than v<i> (1 to 3 digits 0-9), twin and x:<128 lower-case
    hexadecimal digits>, a list with more than one x: arm (they would all be printed "ref0"), and a list with twin
    twice, twin without v0, or twin among ncu arms (ncu=True: the arms of ncu reports)."""
    names = arms.split(",")
    bad = [a for a in names if not ARM.fullmatch(a)]
    if bad:
        raise SystemExit(f"{where}: unknown arm(s) {', '.join(repr(a) for a in bad)} (v0 to v999, twin, x:<128 hexadecimal digits>)")
    if sum(a.startswith("x:") for a in names) > 1:
        raise SystemExit(f"{where}: more than one x: arm (each would be printed ref0)")
    if "twin" in names and (ncu or names.count("twin") > 1 or "v0" not in names):
        raise SystemExit(f"{where}: twin (v0 timed again) goes at most once in a list with v0, and never under ncu")


@app.local_entrypoint()
def main(card: str, candidate: str, shapes: str = "", arms: str = "v0", seconds: float = 1.0, rounds: int = 7, limit: float = 0.0,
         ncu: str = "", ncu_arms: str = "", out: str = "", metrics: str = "", variant: str = "", tries: int = 1):
    if card not in ARCH:
        raise SystemExit(f"--card: {', '.join(ARCH)} (the cards of judge/modal_judge.py)")
    if variant and not re.fullmatch(r"[A-Za-z0-9_.-]+\|\d+W\|lt\d+\|cuda\d+", variant):
        raise SystemExit("--variant: the exact card line, for instance Tesla_T4|70W|lt130201|cuda13030")
    if not 1 <= tries <= 10:
        raise SystemExit("--tries: an integer from 1 to 10")
    if tries > 1 and not variant:
        raise SystemExit("--tries goes with --variant only")
    if limit < 0 or 0 < limit < 1:
        raise SystemExit("--limit: 0 (automatic) or at least 1 s")
    p = pathlib.Path(candidate).resolve()
    files = {"probe.cu": (HERE / "probe.cu").read_text()}
    for v in p.parent.iterdir():
        if v.suffix in (".cuh", ".h", ".hpp") or v == p:
            files[f"cand/{v.name}"] = v.read_text()
    fl = []   # "M,N,K" or "M,N,K=more_arms" (arms added for that shape only)
    for f in shapes.split(";"):
        if f.strip():
            mnk, _, extra = f.partition("=")
            fl.append([int(x) for x in mnk.split(",")] + [extra])
    nc = []   # "M,N,K" or "M,N,K=ncu_arms_of_this_shape"
    for f in ncu.split(";"):
        if f.strip():
            mnk, _, extra = f.partition("=")
            nc.append([int(x) for x in mnk.split(",")] + [extra])
    for f in fl:
        check_arms(arms + ("," + f[3] if f[3] else ""), f"shape {f[0]} {f[1]} {f[2]}")
    for f in nc:
        check_arms(f[3] or ncu_arms or arms, f"ncu shape {f[0]} {f[1]} {f[2]}", ncu=True)
    check_arms(arms, "--arms")
    if ncu_arms: check_arms(ncu_arms, "--ncu-arms", ncu=True)
    plan = plan_seconds(fl, nc, arms, ncu_arms, seconds, rounds, limit)
    if plan > FUNCTION_LIMIT - 600:
        raise SystemExit(f"the plan of this launch ({plan:.0f} s at the worst: build, every shape's limit, ncu) does not fit in "
                         f"the GPU function's limit ({FUNCTION_LIMIT} s, less 600): split it into several launches")
    probe_card = probe.with_options(gpu=card)
    for attempt in range(1, max(1, tries) + 1):
        r = probe_card.remote(files, p.name, card, fl, arms, seconds, rounds, limit, nc, ncu_arms, metrics, variant)
        if "#variant" not in r:
            break
        print(f"try {attempt}/{tries}: {r['log'].strip()}")
    else:
        raise SystemExit(f"no container of the variant {variant} in {tries} try(ies)")
    o = pathlib.Path(out) if out else REPO / RUNG[card] / "results" / f"probe_{p.stem}"
    o = fresh_out(o)
    rel = p.relative_to(REPO) if p.is_relative_to(REPO) else p.name
    digests = "; ".join(f"{k[5:]} md5 {hashlib.md5(v.encode()).hexdigest()}" for k, v in files.items() if k.startswith("cand/"))
    tools = (f"probe.cu md5 {hashlib.md5(files['probe.cu'].encode()).hexdigest()[:8]}; "
             f"{pathlib.Path(__file__).name} md5 {hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()[:8]}")
    (o / "probe.txt").write_text(f"candidate {rel}; {digests}\n{tools}{f'; variant {variant} (try {attempt}/{tries})' if variant else ''}\n"
                                 + r["log"])
    for b, raw in r.get("reports", {}).items():
        (o / f"ncu_{b}.ncu-rep").write_bytes(raw)
    print(r["log"][-8000:])
    print(f"=== -> {o}")
    if r.get("failed"):
        raise SystemExit(f"{len(r['failed'])} failure(s), written in {o / 'probe.txt'}: " + "; ".join(r["failed"]))
