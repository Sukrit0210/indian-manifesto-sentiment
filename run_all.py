"""Run the sentiment study starting from the raw PDFs."""
import argparse
from pathlib import Path
import subprocess
import sys
import hashlib
import venv

def project_python():
    if sys.version_info[:2] != (3, 14):
        raise SystemExit("Select Python 3.14 in VS Code and run this file again.")

    env = ROOT / ".venv314"
    python = env / "Scripts" / "python.exe"
    requirements = ROOT / "requirements.txt"
    marker = env / ".setup-complete"
    key = hashlib.sha256(requirements.read_bytes()).hexdigest()

    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(env)

    if not marker.is_file() or marker.read_text().strip() != key:
        pip = [str(python), "-m", "pip"]
        subprocess.run(
            [*pip, "install", "torch==2.10.0+cu126", "--index-url",
             "https://download.pytorch.org/whl/cu126"],
            check=True,
        )
        subprocess.run([*pip, "install", "-r", str(requirements)], check=True)
        subprocess.run(
            [str(python), "-c",
             "import pymupdf, torch; print(torch.cuda.get_device_name(0)); "
             "print('GPU test:', torch.ones(3, device='cuda').sum().item())"],
            check=True,
        )
        marker.write_text(key, encoding="ascii")

    return str(python)

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_build_dataset.py",
    "02_score_sentiment.py",
    "03_assign_topics.py",
    "04_analyze.py",
    "05_plot.py",
    "06_audit_sensitivity.py",
    "07_expanded_audit.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, help="Directory containing the registered PDFs")
    parser.add_argument(
        "--third-model", action=argparse.BooleanOptionalAction, default=True,
        help="Run the expanded BART prompt comparison in stage 7 (enabled by default)",
    )
    parser.add_argument("--start", type=int, default=1, choices=range(1, len(STEPS) + 1))
    parser.add_argument("--stop", type=int, default=len(STEPS), choices=range(1, len(STEPS) + 1))
    args = parser.parse_args()
    if args.start > args.stop:
        parser.error("--start must be less than or equal to --stop")
    python = project_python()
    for number in range(args.start, args.stop + 1):
        script = STEPS[number - 1]
        command = [python, str(ROOT / "src" / script)]
        if number == 1 and args.pdf_dir:
            command += ["--pdf-dir", str(args.pdf_dir.resolve())]
        if script == "07_expanded_audit.py" and not args.third_model:
            command.append("--no-third-model")
        print(f"Running {script}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
