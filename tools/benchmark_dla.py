import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT DLA engine with random inputs.")
    parser.add_argument("engine", type=Path)
    parser.add_argument("--dla-core", type=int, default=0)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--avg-runs", type=int, default=100)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--export-times", type=Path)
    args = parser.parse_args()

    command = [
        str(args.trtexec),
        f"--loadEngine={args.engine}",
        f"--useDLACore={args.dla_core}",
        f"--warmUp={args.warmup}",
        f"--duration={args.duration}",
        f"--avgRuns={args.avg_runs}",
    ]
    if args.export_times:
        command.append(f"--exportTimes={args.export_times}")
    print("$", " ".join(map(str, command)))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
