"""Archive one V5 build, numerical validation, and sequential baseline timing."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=int)
    parser.add_argument("--register-usage-level", type=int, choices=range(11))
    parser.add_argument("--max-registers", type=int)
    parser.add_argument("--yield-interval", type=int)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--reorder", action="store_true")
    parser.add_argument("--dual-issue", action="store_true")
    parser.add_argument("--bank-assignment", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    target = ROOT / f"docs/cuda-fa2/measurements/v5/run{args.run}"
    data = target / "data"
    data.mkdir(parents=True, exist_ok=True)
    (target / "snapshot").mkdir(exist_ok=True)
    (target / "binaries").mkdir(exist_ok=True)
    for source in (ROOT / "cpp_cuda/fa2/v5").glob("*"):
        if source.suffix in (".cu", ".cuh"):
            shutil.copy2(source, target / "snapshot" / source.name)
    status = {}
    steps = [
        ("build", [sys.executable, "cpp_cuda/fa2/build_extension.py", "--version", "v5",
                   "build_ext", "--inplace", "--force", "--build-temp", "/tmp/fa2-v5-build/temp",
                   "--build-lib", "/tmp/fa2-v5-build/lib"]),
        ("cuda-validation", [sys.executable, "-m", "tests.test_cuda_fa2_forward", "--impl",
                             "cuda_fa2_v5", "--device", "cuda", "--long-sequence",
                             "--output-json", str(data / "row-validation.json")]),
    ]
    for implementation, name in [("cuda_fa2_v5", "custom"), ("efficient", "efficient")]:
        steps.append((f"benchmark-{name}", [sys.executable, "-m", "scripts.benchmark_attention",
                     "--impl", implementation, "--device", "cuda", "--seq-len", "16384",
                     "--head-dim", "64", "--no-causal", "--no-verify",
                     "--output-json", str(data / f"benchmark-{name}.json")]))
    if args.register_usage_level is not None:
        steps[0][1].extend(["--register-usage-level", str(args.register_usage_level)])
    if args.max_registers is not None:
        steps[0][1].extend(["--max-registers", str(args.max_registers)])
    schedule_enabled = (args.yield_interval is not None or args.reuse or args.reorder
                        or args.dual_issue or args.bank_assignment)
    if schedule_enabled:
        extensions = list((ROOT / "src").glob("cuda_fa2_v5_extension*.so"))
        if len(extensions) != 1:
            raise RuntimeError("Expected one V5 extension")
        schedule = [sys.executable, "scripts/schedule_fa2_v5.py", str(extensions[0]),
                    "--output-json", str(data / "schedule.json")]
        schedule.extend(["--bank-assignment"] if args.bank_assignment else
                        ["--dual-issue"] if args.dual_issue else
                        ["--reorder"] if args.reorder else ["--reuse"] if args.reuse
                        else ["--yield-interval", str(args.yield_interval)])
        steps.insert(1, ("schedule", schedule))
    (target / "commands.json").write_text(json.dumps(steps, indent=2) + "\n")
    for name, command in steps:
        with (data / f"{name}.log").open("w") as output:
            status[name] = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        if name == ("schedule" if schedule_enabled else "build") and status[name] == 0:
            for extension in (ROOT / "src").glob("cuda_fa2_v5_extension*.so"):
                shutil.copy2(extension, target / "binaries")
        (target / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if status[name]:
            break
    owner = (int(os.environ.get("SUDO_UID", os.getuid())),
             int(os.environ.get("SUDO_GID", os.getgid())))
    if os.geteuid() == 0:
        for extension in (ROOT / "src").glob("cuda_fa2_v5_extension*.so"):
            os.chown(extension, *owner)
        for directory, _, filenames in os.walk(target):
            os.chown(directory, *owner)
            for filename in filenames:
                os.chown(Path(directory) / filename, *owner)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
