"""Train from hash-bound T11 caches; never opens a camera or final test cache."""
import argparse
import json
from pathlib import Path
import sys

from algorithm.training.config import load_train_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory; existing only for --resume")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--resume", type=Path, help="trusted local output/last.pt; exact epoch boundary only")
    parser.add_argument("--stop-after-epoch", type=int, help="pause cleanly after this absolute epoch for resume verification")
    args = parser.parse_args(argv)
    try:
        from algorithm.training.trainer import train
        report = train(args.train_cache, args.val_cache, args.output, config=load_train_config(args.training_config),
                       resume=args.resume, stop_after_epoch=args.stop_after_epoch)
        print(json.dumps(report, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
