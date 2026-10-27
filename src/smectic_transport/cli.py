"""Command line: `smectic-transport run CONFIG` produces the output data; `info` reports the engine."""
import argparse
import sys

import yaml

from .engines import EngineSelector
from .experiment import ExperimentConfig, Simulation
from .storage import ResultStore


def run_config(path, overrides=None, engine=None, overwrite=False, dry_run=False):
    """Run every point of the configuration's sweep; existing output files are skipped."""
    base = ExperimentConfig.from_yaml(path)
    if overrides:
        base = base.with_overrides(overrides)
    points = base.expand_sweep()
    for i, cfg in enumerate(points, 1):
        cfg = cfg.with_seed()
        out = cfg.output_path()
        tag = f"[{i}/{len(points)}] {out}"
        if dry_run:
            print(f"{tag}  ({cfg.n_total:,} steps x {cfg.particles.n} particles)")
        elif out.exists() and not overwrite:
            print(f"{tag} exists, skipped", file=sys.stderr)
        else:
            ResultStore.save(Simulation(cfg).run(engine=engine), out)
            print(f"{tag} saved", file=sys.stderr)


def main(argv=None):
    """Parse the command line and dispatch."""
    ap = argparse.ArgumentParser(prog="smectic-transport", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="report which engine (GPU/CPU) will be used")
    r = sub.add_parser("run", help="run a configuration file (all points of its sweep)")
    r.add_argument("config")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a value, e.g. dynamics.peclet=4")
    r.add_argument("--engine", choices=["auto", "cpu", "gpu"])
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="list the runs without executing them")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        name, reason = EngineSelector.resolve("auto")
        print(f"engine: {name} ({reason})")
        return
    overrides = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    run_config(args.config, overrides, args.engine, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
