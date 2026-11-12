"""Rebuild the three result tables from the original, locally retained logs."""
import argparse
import csv
import json
import math
from pathlib import Path
import re


def read_json_after(text, marker):
    start = text.index("{", text.index(marker) + len(marker))
    return json.JSONDecoder().raw_decode(text[start:])[0]


def extract(source_root):
    work = source_root / "Work"
    text = (work / "train-8835637.log").read_text(encoding="utf-8")
    config = read_json_after(text, "New Training Run with Config")
    steps = config["experiment"]["epoch_every_n_steps"]
    losses = []
    for match in re.finditer(r"Train Epoch (\d+)\s*\n", text):
        stats = read_json_after(text[match.end():], "")
        epoch = int(match[1])
        losses.append(dict(epoch=epoch, gradient_updates=epoch * steps, loss=stats["Loss"]))
    successes = []
    for match in re.finditer(r"Epoch (\d+) Rollouts took [^\n]+\nEnv: [^\n]+\n", text):
        stats = read_json_after(text[match.end():], "")
        epoch = int(match[1])
        successes.append(dict(epoch=epoch, gradient_updates=epoch * steps,
                              success_rate=stats["Success_Rate"]))
    candidate_text = (work / "eval-candidates-8917289.log").read_text(encoding="utf-8")
    candidate_stats = {}
    for match in re.finditer(r"=== Evaluating Epoch (\d+) ===", candidate_text):
        candidate_stats[int(match[1])] = read_json_after(candidate_text[match.end():], "Average Rollout Stats")
    candidate_stats[65] = read_json_after(
        (work / "eval65-8914812.log").read_text(encoding="utf-8"), "Average Rollout Stats")
    for script in ["eval_candidates.sh", "eval_65.sh"]:
        commands = (work / script).read_text(encoding="utf-8")
        for flag, value in [("n_rollouts", "50"), ("horizon", "400"), ("seed", "123")]:
            if not re.search(r"--" + flag + r"\s+" + value + r"\b", commands):
                raise ValueError(f"Unverified {flag} in {script}")
    candidates = []
    for epoch, stats in sorted(candidate_stats.items()):
        count = int(stats["Num_Success"])
        if not math.isclose(count / 50, stats["Success_Rate"], abs_tol=1e-12):
            raise ValueError("Inconsistent success count and success rate")
        candidates.append(dict(epoch=epoch, gradient_updates=epoch * steps,
                               success_rate_50=stats["Success_Rate"],
                               num_success=count, num_rollouts=50))
    if [r["epoch"] for r in losses] != list(range(1, 121)):
        raise ValueError("Expected one loss per epoch, 1 through 120")
    if [r["epoch"] for r in successes] != list(range(5, 121, 5)):
        raise ValueError("Expected 24 preliminary evaluations")
    if [r["epoch"] for r in candidates] != [55, 60, 65, 75]:
        raise ValueError("Unexpected candidate epochs")
    return {"training_loss.csv": losses, "success_rate.csv": successes,
            "verified_checkpoints.csv": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Local directory containing Work/")
    parser.add_argument("--output-dir", type=Path, help="Write rebuilt tables here")
    parser.add_argument("--check-against", type=Path, help="Compare every numeric cell with existing tables")
    args = parser.parse_args()
    if args.output_dir is None and args.check_against is None:
        parser.error("Supply --output-dir or --check-against")
    tables = extract(args.source_root)
    for name, rows in tables.items():
        if args.check_against:
            with (args.check_against / name).open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
            if len(existing) != len(rows):
                raise ValueError(f"Row count differs in {name}")
            for expected, actual in zip(rows, existing):
                if set(expected) != set(actual):
                    raise ValueError(f"Column names differ in {name}")
                for key, value in expected.items():
                    if not math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"Mismatch: {name}, epoch {expected['epoch']}, {key}")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        print(f"{name}: {len(rows)} rows verified")


if __name__ == "__main__":
    main()
