"""Modal launcher of the judge (judge.cu, next to it): the time of each candidate against the judge's reference.

Builds each candidate with judge.cu for the card in a container WITHOUT a GPU (the same image, so the same nvcc and the
same libraries), then has the linked judge run in a GPU container, which gets from the candidate only that binary and
the text of its build, at the card's shapes (shapes/<card>.txt); if no candidate has a binary, no GPU container is
asked for. The output lands in <rung>/results/judge/<date>-<time>_<candidate>/ (turing for the T4) with a copy of the
judged sources (the .cu and its headers) and their md5.

Usage (from the repository's root):
  modal run judge/modal_judge.py --candidates turing/step6.cu[,turing/step5.cu] \
      [--card T4] [--options "--quick --only qkv_m16,ladder"] [--reference] \
      [--variant "Tesla_T4|70W|lt130201|cuda13030" [--tries N]]

Rules applied here:
- a candidate is a .cu that defines candidate_gemm (see judge.cu's header); the .cuh/.h/.hpp of its folder are sent
  with it; file names: letters, digits, "_", "-", "."; two candidates of the same name are refused; if the environment
  variable CUTLASS_INCLUDE gives the folder of the CUTLASS headers (4.4.1), they are mounted under
  /opt/cutlass/include (for candidates built on CUTLASS);
- the candidate's object is built apart; all its defined symbols but candidate_gemm are made LOCAL before the link (a
  candidate cannot replace a function of cuBLAS or of the judge); its unresolved symbols are filtered: cuBLAS, cuDNN,
  dynamic loading, capture and graphs, the GPU's state (the FORBIDDEN list); --control lifts the filter, for the
  control candidates of a judge/tests/ folder only (not provided here);
- --options accepts only: --quick, --only LIST, --pairs N, --duration S, --warmup S, --default;
- --shapes FILE: another list of shapes (a check on other shapes, extra shapes), in the format of shapes/<card>.txt;
  its name and md5 are written in the header of each output; without it, the card's 17 shapes;
- reference: without --reference or --quick, the judge sorts all of cuBLASLt's configurations and the launcher keeps
  the finalists of each shape, with the card's identity, in reference/<card>.txt (the new ones first, then up to 3 old
  ones still absent); with --reference, the judge sorts only those of ITS card, the heuristic's first 3 at 64 MB and
  its first 8 not yet present at 32, 4, 1 and 0 MB, then the same final;
- --variant V: the GPU container reads its card line (the card's name through NVML, the same as the judge's CUDA name
  on the cards seen; power cap; cuBLASLt; driver) BEFORE judging, and stops at once if it differs from V (about 10 s
  instead of a whole pass); --tries N (with --variant only) makes at most N tries, the first included, each in a fresh
  GPU container (which may land on the same machine), all with the binaries of the single build; only the pass of the
  right variant, by that reading, is stored, and its output.txt notes the variant and the try ("not checked" if there
  is no binary); a gap between that reading and the judge's card line is written there (WARNING);
- output.txt: line 1, the candidate's path and the md5 of each file sent; line 2, the md5 of judge.cu, of this launcher,
  of the shapes file and of the reference file sent (with --reference), the options, the variant and the try; then the
  launcher's log and the judge's output. Each output goes to a folder of its own (<date>-<time>-2_<candidate> when two
  outputs land in the same second);
- --recover FILE.json stores a pass copied to the volume: the files it records (the judged sources, the md5 of judge.cu,
  of the shapes, of the reference and of the launcher) are written, never the disk's; it refuses if a judged source,
  judge.cu or the shapes file on the disk differs from what was judged;
- a candidate's sources are read once, before they are sent; two candidates of one launch cannot send two different
  headers of the same name.
"""
import datetime
import fcntl
import hashlib
import os
import pathlib
import re
import shlex

import modal

HERE = pathlib.Path(__file__).resolve().parent   # judge/
REPO = HERE.parent
TESTS = HERE / "tests"
CUTLASS = os.environ.get("CUTLASS_INCLUDE", "")  # CUTLASS headers, for candidates built on CUTLASS only (optional)
ARCH = {"T4": "sm_75", "A100-80GB": "sm_80", "H100!": "sm_90a", "B200": "sm_100a"}
# The "a" targets (wgmma, tcgen05) need the -gencode form: with -arch=sm_90a, nvcc 13 refuses wgmma.
GEN = {c: (f"-gencode arch=compute_{a[3:]},code={a}" if a.endswith("a") else f"-arch={a}") for c, a in ARCH.items()}
CACHE_STEM = {"T4": "t4", "A100-80GB": "a100", "H100!": "h100", "B200": "b200"}
RUNG = {"T4": "turing", "A100-80GB": "ampere", "H100!": "hopper", "B200": "blackwell"}   # where the outputs go
SHAPES = {"T4": "T4.txt", "A100-80GB": "A100.txt", "H100!": "H100.txt", "B200": "B200.txt"}
FORBIDDEN = (r"cublas|cudnn|\bdl\w*(open|sym)|dl_iterate_phdr|cudaStreamIsCapturing|cudaStreamGetCaptureInfo|"
             r"cudaStreamBeginCapture|cudaStreamEndCapture|"
             r"cudaStreamUpdateCaptureDependencies|cudaStreamBeginCaptureToGraph|cudaThreadExchangeStreamCaptureMode|"
             r"cudaGraph|cudaStreamSetAttribute|cudaDeviceSetLimit|cudaCtxResetPersistingL2Cache|cudaDeviceSetCacheConfig|"
             r"cudaDeviceSetSharedMemConfig|cudaLaunchHostFunc|\b(setenv|putenv|unsetenv|clearenv)\b|cudaStreamAddCallback|\bcuStream(IsCapturing|GetCaptureInfo|BeginCapture|EndCapture|"
             r"UpdateCaptureDependencies|BeginCaptureToGraph|SetAttribute|AddCallback)|\bcuGraph|\bcuCtxSetLimit|"
             r"\bcuCtxResetPersistingL2Cache|\bcuCtxSetCacheConfig|\bcuLaunchHostFunc")
SYMBOL = "_Z14candidate_gemmPK6__halfS1_PS_iiiP11CUstream_st"   # void candidate_gemm(const half*, const half*, half*, int, int, int, cudaStream_t)
OPTIONS = {"--quick": 0, "--only": 1, "--pairs": 1, "--duration": 1, "--warmup": 1, "--default": 0}

image = modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
if CUTLASS:
    image = image.add_local_dir(CUTLASS, remote_path="/opt/cutlass/include")
# Each GPU pass also writes its output on this volume before it returns: if the local client is cut off (a launch
# with modal run --detach), the output is recovered with --recover (see main).
OUTPUTS = modal.Volume.from_name("judge-outputs", create_if_missing=True)
app = modal.App("gemm-ladder-judge", image=image)


def safe_options(options: str) -> list:
    words = shlex.split(options)
    i, kept = 0, []
    while i < len(words):
        m = words[i]
        if m not in OPTIONS:
            raise SystemExit(f"option refused in --options: {m} (allowed: {', '.join(OPTIONS)})")
        n = OPTIONS[m]
        val = words[i + 1:i + 1 + n]
        if len(val) != n:
            raise SystemExit(f"option {m} without a value")
        if m == "--pairs" and not (val[0].isdigit() and 1 <= int(val[0]) <= 32):
            raise SystemExit("--pairs: an integer from 1 to 32")
        if m in ("--duration", "--warmup") and not re.fullmatch(r"\d+(\.\d+)?", val[0]):
            raise SystemExit(f"{m}: a positive number")
        if m == "--only" and not re.fullmatch(r"[A-Za-z0-9_,]+", val[0]):
            raise SystemExit("--only: shape names separated by commas")
        kept += [m] + val
        i += 1 + n
    return kept


def card_line() -> str:
    """The card line the judge will write (judge.cu: the card's name, the NVML power cap enforced in W, the version of
    cuBLASLt, the driver's CUDA version), read without building anything, through NVML, cuBLASLt and the driver."""
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
    os.makedirs("/tmp/j/cand", exist_ok=True)
    for name, content in files.items():
        p = pathlib.Path("/tmp/j") / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _sh():
    import subprocess
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))

    def sh(args, t):
        """Runs without a shell; past the time limit, returns what was written so far."""
        try:
            r = subprocess.run(args, cwd="/tmp/j", env=env, capture_output=True, text=True, timeout=t)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            dec = lambda x: (x.decode(errors="replace") if isinstance(x, bytes) else (x or ""))
            return -9, dec(e.stdout), dec(e.stderr) + f"\nTIMEOUT ({t} s)\n"
    return sh


def _compiler(files: dict, candidates: list, card: str, control: bool, judge_md5: str) -> dict:
    """WITHOUT A GPU: builds each candidate, makes its symbols local but candidate_gemm, filters its forbidden calls,
    builds the judge and links. Returns, per candidate, the text of these steps and the judge's binary (None if a step
    fails)."""
    _write(files)
    sh = _sh()
    head = "container without a GPU (build):\n" + sh(["bash", "-c", "nvcc --version | tail -1; date -u"], 60)[1]
    gen = GEN[card].split()
    out = {}
    for c in candidates:
        stem = pathlib.Path(c).stem
        log, binary = head, None
        rc, so, se = sh(["nvcc", "-O3", "-std=c++17", *gen, "-I/opt/cutlass/include", "-c", f"cand/{c}", "-o", f"cand_{stem}.o"], 1200)
        log += f"build of the candidate: rc {rc}\n{so}{se}\n"
        if rc == 0:
            # All its defined symbols become local, but candidate_gemm (the exact name of the required signature; the
            # host compiler may split a ".cold" part off it, which stays internal to the object).
            _, nm_def, _ = sh(["nm", "--defined-only", f"cand_{stem}.o"], 60)
            if SYMBOL not in {l.split()[-1] for l in nm_def.splitlines() if l.split()}:
                log += f"FAILED: candidate_gemm missing from the candidate's object, or another signature (expected {SYMBOL})\n"
                out[c] = {"log": log, "binary": None}
                continue
            rc, so, se = sh(["objcopy", f"--keep-global-symbol={SYMBOL}", f"cand_{stem}.o", f"cand_{stem}_local.o"], 120)
            log += f"isolation of the symbols: rc {rc}\n{so}{se}"
        if rc == 0 and not control:
            _, nm_u, _ = sh(["nm", "-u", f"cand_{stem}_local.o"], 60)
            forbidden = sorted({l.split()[-1] for l in nm_u.splitlines() if re.search(FORBIDDEN, l, re.I)})
            if forbidden:
                log += "FAILED: the candidate calls a forbidden function: " + " ".join(forbidden) + "\n"
                out[c] = {"log": log, "binary": None}
                continue
        if rc == 0:
            rc, so, se = sh(["nvcc", "-O3", "-std=c++17", *gen, f'-DJUDGE_MD5="{judge_md5}"', "judge.cu", f"cand_{stem}_local.o",
                             "-o", f"judge_{stem}", "-lcublasLt", "-lcublas", "-ldl"], 1200)
            log += f"build of the judge: rc {rc}\n{so[-3000:]}{se[-3000:]}\n"
        if rc == 0:
            binary = pathlib.Path(f"/tmp/j/judge_{stem}").read_bytes()
        out[c] = {"log": log, "binary": binary}
    return out


def _run(files: dict, built: dict, candidates: list, options: list, variant: str = "") -> dict:
    """ON THE GPU: checks the variant (--variant), then runs the built judge of each candidate."""
    _write(files)
    sh = _sh()
    head = sh(["bash", "-c", "nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader; "
               "date -u"], 60)[1]
    if variant:
        here = card_line()
        if here != variant:
            return {"#variant": f"this container is {here}, expected {variant}; stopped before judging\n" + head}
    import time
    deadline = time.time() + 6 * 3600 - 600   # the function's budget (6 h), less 10 min to return the outputs
    out = {}
    for c in candidates:
        stem = pathlib.Path(c).stem
        log, judge_out = head + built[c]["log"], ""
        if built[c]["binary"] is not None:
            b = pathlib.Path(f"/tmp/j/judge_{stem}")
            b.write_bytes(built[c]["binary"])
            b.chmod(0o755)
            ref = ["--reference", "reference.txt"] if "reference.txt" in files else []
            rc, so, se = sh([f"./judge_{stem}", "shapes.txt", *options, *ref], max(60, min(12600, int(deadline - time.time()))))   # 3.5 h at most, within the budget left
            judge_out = f"{so}{se}"
            log += f"judge: rc {rc}\n{judge_out}"
        # The judge's output apart: the launcher reads the card line and the cache's lines there only, never in the
        # text of the build, which a candidate can fill (#pragma message).
        out[c] = {"log": log, "judge": judge_out}
    return out


# The build without a GPU: the same image, so the same nvcc and the same libraries as the container that runs.
@app.function(cpu=8.0, memory=16384, timeout=6 * 3600)
def compiler(files: dict, candidates: list, card: str, control: bool, judge_md5: str) -> dict:
    return _compiler(files, candidates, card, control, judge_md5)


# single_use_containers: each call takes a fresh container (a new try of --variant may still land on the same
# machine).
@app.function(cpu=4.0, memory=16384, timeout=6 * 3600, single_use_containers=True, volumes={"/outputs": OUTPUTS})
def judge(files: dict, built: dict, candidates: list, options: list, variant: str = "", attempt: int = 1, tries: int = 1,
          judged: dict = None) -> dict:
    res = _run(files, built, candidates, options, variant)
    try:   # a backup copy, before returning (the local client may have been cut off)
        import json, time, uuid
        name = f"/outputs/{time.strftime('%Y%m%d-%H%M%S')}_{'-'.join(c.rsplit('.', 1)[0] for c in candidates)[:80]}_{uuid.uuid4().hex[:6]}.json"
        with open(name, "w") as f:
            json.dump({"candidates": candidates, "options": options, "variant": variant, "attempt": attempt, "tries": tries,
                       "judged": judged, "res": res}, f)
        OUTPUTS.commit()
        print(f"output copied to the volume judge-outputs: {name}")
    except Exception as e:
        print(f"no copy to the volume: {e}")
    return res


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


def store_reference(cache: pathlib.Path, lines: list):
    """Stores the finalists of each judged shape: the new ones first, then up to 3 old ones still absent. A REFERENCE
    line is kept only if it carries the pass's token (the "token:" line of the judge's header) and if the CSV line of
    the same shape follows it in the output; it is stored without the token."""
    tokens = [l.split(":", 1)[1].strip() for l in lines if l.startswith("token: ")]
    if len(tokens) != 1:
        return
    fresh, pending = {}, {}
    for l in lines:
        m = l.split()
        if l.startswith("REFERENCE ") and len(m) == 10 and m[9] == f"token={tokens[0]}":
            l = " ".join(m[:9])
            pending.setdefault((m[1], m[2], m[3], m[4], m[5]), []).append(l)
        elif "," in l:
            f = l.split(",")
            for key in [k for k in pending if f[:4] == [k[0], k[1], k[2], k[3]]]:
                fresh[key] = pending.pop(key)
    if not fresh:
        return
    cache.parent.mkdir(exist_ok=True)
    with open(cache.with_suffix(".lock"), "w") as v:
        fcntl.flock(v, fcntl.LOCK_EX)
        previous = {}
        for l in (cache.read_text().splitlines() if cache.exists() else []):
            m = l.split()
            if l.startswith("REFERENCE ") and len(m) == 9:
                previous.setdefault((m[1], m[2], m[3], m[4], m[5]), []).append(l)
        for key, ls in fresh.items():
            seen = {l.split()[8] for l in ls}
            previous[key] = ls + [l for l in previous.get(key, []) if l.split()[8] not in seen][:3]
        tmp = cache.with_suffix(".tmp")
        tmp.write_text("".join(l + "\n" for ls in previous.values() for l in ls))
        os.replace(tmp, cache)


@app.local_entrypoint()
def main(candidates: str, card: str = "T4", options: str = "", control: bool = False, reference: bool = False,
         variant: str = "", tries: int = 1, shapes: str = "", recover: str = ""):
    # --recover FILE.json: the output of a pass copied to the volume judge-outputs (modal volume get judge-outputs
    # NAME .); nothing runs again, the output is stored as at the end of a pass, with the same arguments (candidates,
    # variant and --tries are checked against the file; the try noted is the file's). The md5 and the copies written are
    # those of the judged files, recorded in the file; a judged source, judge.cu or the shapes file that differs on the
    # disk now is refused.
    opts = safe_options(options)
    if variant and not re.fullmatch(r"[A-Za-z0-9_.-]+\|\d+W\|lt\d+\|cuda\d+", variant):
        raise SystemExit("--variant: the exact card line, for instance NVIDIA_B200|1000W|lt130201|cuda13010")
    if not 1 <= tries <= 10:
        raise SystemExit("--tries: an integer from 1 to 10")
    if tries > 1 and not variant:
        raise SystemExit("--tries goes with --variant only")
    judge_src = (HERE / "judge.cu").read_text()
    judge_md5 = hashlib.md5(judge_src.encode()).hexdigest()[:8]
    launcher_md5 = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()[:8]
    shapes_file = shapes or f"shapes/{SHAPES[card]}"
    shapes_text = (HERE / "shapes" / SHAPES[card]).read_text()
    if shapes:   # another list of shapes (a check on other shapes, extra shapes): the format of shapes/<card>.txt
        shapes_text = pathlib.Path(shapes).read_text()
        for l in shapes_text.splitlines():
            if l.strip() and not l.lstrip().startswith("#") and not re.fullmatch(r"\d+ \d+ \d+ [A-Za-z0-9_]+", l.strip()):
                raise SystemExit(f"--shapes: line refused (expected \"M N K label\"): {l}")
    files = {"judge.cu": judge_src, "shapes.txt": shapes_text}
    cache = HERE / "reference" / f"{CACHE_STEM[card]}.txt"
    if reference:
        if not cache.exists():
            raise SystemExit(f"--reference: no file {cache} (run a full pass first)")
        files["reference.txt"] = cache.read_text()
    names, sources = [], {}
    for c in candidates.split(","):
        p = pathlib.Path(c).resolve()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", p.name):
            raise SystemExit(f"candidate name refused (letters, digits, _ - . only): {p.name}")
        if p.name in names:
            raise SystemExit(f"two candidates have the same name: {p.name}")
        if control and TESTS not in p.parents:
            raise SystemExit("--control is for the control candidates of judge/tests/ only")
        sent = [v for v in p.parent.iterdir() if v.suffix in (".cuh", ".h", ".hpp")] + [p]
        # The sources are read ONCE, before they are sent; the md5 and the copies in the results folder come from what
        # was sent, not from the disk after the pass (a header changed in the meantime).
        snapshot = {v.name: v.read_bytes() for v in sent}
        for name, raw in snapshot.items():
            if files.get(f"cand/{name}", raw.decode()) != raw.decode():
                raise SystemExit(f"two candidates send a different {name}: launch them separately")
            files[f"cand/{name}"] = raw.decode()
        names.append(p.name)
        sources[p.name] = snapshot
    # What is judged, recorded with the pass's output (on the volume too): written by --recover instead of the disk.
    judged = {"judge_md5": judge_md5, "launcher_md5": launcher_md5, "shapes": shapes_file,
              "shapes_md5": hashlib.md5(shapes_text.encode()).hexdigest()[:8],
              "reference": f"reference/{CACHE_STEM[card]}.txt" if reference else "",
              "reference_md5": hashlib.md5(files["reference.txt"].encode()).hexdigest()[:8] if reference else "",
              "control": control,
              "sources": {n: {f: raw.decode() for f, raw in sources[n].items()} for n in names}}
    # The build is made once, without a GPU; the GPU container gets only the binaries.
    to_build = {k: v for k, v in files.items() if k == "judge.cu" or k.startswith("cand/")}
    to_run = {k: v for k, v in files.items() if k not in to_build}
    built = None if recover else compiler.remote(to_build, names, card, control, judge_md5)
    attempt = 0
    if recover:
        import json
        saved = json.loads(pathlib.Path(recover).read_text())
        if saved["candidates"] != names or saved["variant"] != variant:
            raise SystemExit("--recover: candidates or variant other than the file's")
        if saved.get("tries") != tries:
            raise SystemExit(f"--recover: --tries {tries}, the file's pass was launched with --tries {saved.get('tries')}")
        if "#variant" in saved["res"]:
            raise SystemExit("--recover: this file is a try stopped for another variant")
        if not saved.get("judged"):
            raise SystemExit("--recover: this file does not record the judged files (an earlier launcher): not stored")
        rec = saved["judged"]
        if saved.get("options") != opts or bool(rec["reference"]) != reference or rec.get("control", False) != control:
            raise SystemExit(f"--recover: options {opts}, --reference {reference}, --control {control}; the file's "
                             f"pass: options {saved.get('options')}, --reference {bool(rec['reference'])}, "
                             f"--control {rec.get('control', False)}")
        changed = [f"{n}: {f}" for n in names for f in sorted(set(rec["sources"][n]) | set(sources[n]))
                   if f not in rec["sources"][n] or f not in sources[n] or rec["sources"][n][f].encode() != sources[n][f]]
        changed += [f"judge.cu (md5 {judge_md5}, judged {rec['judge_md5']})"] if rec["judge_md5"] != judge_md5 else []
        changed += ([f"{shapes_file} (md5 {judged['shapes_md5']}, judged {rec['shapes_md5']})"]
                    if rec["shapes"] != shapes_file or rec["shapes_md5"] != judged["shapes_md5"] else [])
        for n in names:   # the judge's own header names the judge it ran
            m = re.search(r"^judge v4 \(md5 (\w+)\)", saved["res"][n]["judge"], re.M)
            changed += [f"{n}: the judge's output says md5 {m.group(1)}"] if m and m.group(1) != rec["judge_md5"] else []
        if changed:
            raise SystemExit("--recover: the files on the disk differ from the judged ones: " + "; ".join(changed))
        judged = rec   # the pass's record: its launcher and its reference file as sent
        sources = {n: {f: t.encode() for f, t in rec["sources"][n].items()} for n in names}
        res, attempt = saved["res"], saved["attempt"]
        print(f"output recovered from {recover}")
    elif all(built[n]["binary"] is None for n in names):   # nothing to time: no GPU container is asked for
        res = {n: {"log": built[n]["log"], "judge": ""} for n in names}
    else:
        judge_card = judge.with_options(gpu=card)
        for attempt in range(1, tries + 1):
            res = judge_card.remote(to_run, built, names, opts, variant, attempt, tries, judged)
            if "#variant" not in res:
                break
            print(f"try {attempt}/{tries}: ANOTHER VARIANT: {res['#variant'].splitlines()[0]}")
        else:
            raise SystemExit(f"no container of the variant {variant} in {tries} try(ies): nothing is stored")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full = not reference and "--quick" not in opts
    for c in candidates.split(","):
        p = pathlib.Path(c).resolve()
        dest = fresh_dir(REPO / RUNG[card] / "results" / "judge", stamp, p.stem)
        judge_lines = res[p.name]["judge"].splitlines()
        seen_card = next((l.split(":", 1)[1].strip() for l in judge_lines if l.startswith("card: ")), None)
        warning = (f"WARNING: the judge writes the card {seen_card}, the variant asked for was {variant} (the launcher's reading needs fixing)\n"
                  if variant and seen_card and seen_card != variant else "")
        digests = "; ".join(f"{name} md5 {hashlib.md5(raw).hexdigest()}" for name, raw in sources[p.name].items())
        (dest / "output.txt").write_text(f"candidate {p.relative_to(REPO) if p.is_relative_to(REPO) else p.name}; {digests}\n"
                                          f"judge md5 {judged['judge_md5']}; {pathlib.Path(__file__).name} md5 {judged['launcher_md5']}; "
                                          f"shapes {judged['shapes']} md5 {judged['shapes_md5']}"
                                          f"{f'; reference ' + judged['reference'] + ' md5 ' + judged['reference_md5'] if judged['reference'] else ''}"
                                          f"; options: {' '.join(opts)}"
                                          f"{' --reference' if reference else ''}{' (control)' if control else ''}"
                                          f"{f'; variant {variant} (try {attempt}/{tries})' if variant and attempt else ''}"
                                          f"{f'; variant {variant} not checked (no binary, no GPU container)' if variant and not attempt else ''}\n"
                                          + warning + res[p.name]["log"])
        for name, raw in sources[p.name].items():
            (dest / name).write_bytes(raw)
        lines = res[p.name]["log"].splitlines()
        if warning:
            print(warning, end="")
        if full and judge_lines:   # the judge's output only (without a binary, there is none)
            store_reference(cache, judge_lines)
        print(f"=== {p.name} -> {dest}")
        print("\n".join(l for l in lines if l.startswith(("SCORE", "CYCLES", "CARD THROTTLED", "WARNING", "FAILED", "build of", "isolation of", "judge:", "cuBLAS", "CUDA", "TIMEOUT", "info: the reference of")))
              or "\n".join(lines[-15:]))
