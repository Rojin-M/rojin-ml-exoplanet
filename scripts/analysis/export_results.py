#!/usr/bin/env python3
"""Verify a completed workflow and export compact tables and figures for publication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.data.check_processed import main as check_processed

SEEDS = [42, 43, 44, 45, 46]
METRICS = {"accuracy": "Accuracy", "f1": "F1", "pr_auc": "Average precision", "roc_auc": "ROC-AUC"}
LIMITS = (
    "AP is scikit-learn average precision. Five-seed error bars are sample standard deviations "
    "of training runs on the same previously consulted Kepler split; they are not confidence intervals. "
    "Metadata comparisons change both inputs and model capacity. Model families use different training "
    "recipes and precision modes, so their comparison is descriptive. These experiments do not measure TESS transfer."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Publication:
    def __init__(self, work, destination):
        self.work = work
        self.destination = destination
        self.sources = {}

    def portable(self, value):
        if isinstance(value, dict):
            return {self.portable(k): self.portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.portable(v) for v in value]
        if isinstance(value, str):
            if value == str(ROOT):
                return "."
            value = value.replace(str(ROOT) + "/", "")
            if not self.work.is_relative_to(ROOT):
                value = value.replace(str(self.work) + "/", "workflow/")
            return value
        return value

    def remember(self, path):
        self.sources[self.portable(str(path.resolve()))] = sha256(path)
        return path

    def read_json(self, path):
        return json.loads(self.remember(path).read_text())

    def read_csv(self, path):
        return pd.read_csv(self.remember(path))

    def write(self, name, text):
        target = self.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip() + "\n")

    def write_json(self, name, data):
        self.write(name, json.dumps(self.portable(data), indent=2, allow_nan=False))

    def write_csv(self, name, frame):
        target = self.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = frame.copy()
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].map(self.portable)
        frame.to_csv(target, index=False)

    def copy(self, source, name):
        target = self.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.remember(source), target)

    def copy_figure(self, source, name):
        for suffix in (".png", ".pdf"):
            self.copy(source.with_suffix(suffix), name + suffix)

    def verify_hashes(self, values):
        for value, expected in values.items():
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            actual = sha256(self.remember(path))
            if actual != expected:
                raise ValueError(f"Saved evidence changed: {path}")

    def small_plot(self, frame, title, name, metrics=("pr_auc", "f1")):
        labels = [METRICS.get(m, m.capitalize()) for m in metrics]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(metrics))
        width = .75 / len(frame)
        colors = ("#0072B2", "#D55E00", "#009E73")
        for i, (_, row) in enumerate(frame.iterrows()):
            values = [row["test_" + m] for m in metrics]
            bars = ax.bar(x + (i - (len(frame) - 1) / 2) * width, values, width,
                          label=row["model"], color=colors[i % len(colors)])
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.13), ylabel="Test score", title=title)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=len(frame), frameon=False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.text(.5, .015, "Individual runs; thresholds selected on validation. No seed uncertainty estimate.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, 1))
        target = self.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            fig.savefig(target.with_suffix(suffix), dpi=220, bbox_inches="tight")
        plt.close(fig)

    def paired_table(self, family, runs):
        source = self.work / ("cnn_comparison/results" if family == "cnn" else "transformer/results")
        comparison = self.read_csv(source / "comparison_metrics.csv")
        per_name = "per_seed_metrics.csv" if family == "cnn" else "per_run_metrics.csv"
        per_run = self.read_csv(source / per_name)
        paired = self.read_csv(source / "paired_seed_metrics.csv")
        prefix = "CNN" if family == "cnn" else "Transformer"
        baseline = runs[runs.model == prefix + " v0"].set_index("seed").sort_index()
        candidate = runs[runs.model == prefix + " v1 + metadata"].set_index("seed").sort_index()
        if list(baseline.index) != SEEDS or list(candidate.index) != SEEDS:
            raise ValueError(f"Missing seeds: {family}")
        # Verify the published paired summaries independently against the final predictions.
        for _, row in comparison.iterrows():
            metric = "pr_auc" if row.metric == "average_precision" else row.metric
            a = baseline[row["split"] + "_" + metric]
            b = candidate[row["split"] + "_" + metric]
            for key, expected in (("v0_mean", a.mean()), ("v0_std", a.std(ddof=1)),
                                  ("v1_mean", b.mean()), ("v1_std", b.std(ddof=1)),
                                  ("delta_mean", (b-a).mean()), ("delta_std", (b-a).std(ddof=1))):
                np.testing.assert_allclose(row[key], expected, rtol=1e-10, atol=1e-12)
            if int(row.improved_seeds) != int((b > a).sum()):
                raise ValueError(f"Incorrect paired improvement count: {family}")
        self.write_csv(f"{family}/comparison_metrics.csv", comparison)
        self.write_csv(f"{family}/{per_name}", per_run)
        self.write_csv(f"{family}/paired_seed_metrics.csv", paired)
        return comparison

    def build(self):
        workflow = self.read_json(self.work / "workflow.json")
        if workflow["status"] != "complete" or any(r["status"] != "complete" for r in workflow["stages"].values()):
            raise ValueError("Finish the complete reproduction workflow before publishing")
        self.verify_hashes(workflow["source_sha256"])
        check_processed()
        self.verify_hashes(self.read_json(ROOT / "configs/data_checksums.json"))
        cnn_audit = self.read_json(self.work / "cnn_comparison/results/comparison.json")
        transformer_audit = self.read_json(self.work / "transformer/results/verification.json")
        self.verify_hashes(cnn_audit["source_files_sha256"])
        self.verify_hashes(transformer_audit["source_files_sha256"])
        self.verify_hashes(transformer_audit["export_sha256"])

        # This analyzer only reads predictions/configs; it never runs model inference or training.
        subprocess.run([sys.executable, str(ROOT / "scripts/analysis/compare_all_models.py"),
                        "--workflow_dir", str(self.work), "--output_dir", str(self.destination / "comparison")],
                       cwd=ROOT, check=True)
        runs = pd.read_csv(self.destination / "comparison/runs.csv")
        for name in ("runs.csv", "summary.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(self.destination / "comparison" / name),
                                          self.read_csv(self.work / "comparison" / name),
                                          check_exact=False, rtol=1e-10, atol=1e-12)
        if len(runs) != 26:
            raise ValueError("Expected all 26 final evaluated runs")

        def subset(names):
            return runs.set_index("model").loc[names].reset_index()

        def report(frame):
            lines = ["| Model | Accuracy | F1 | AP | ROC-AUC |", "| --- | ---: | ---: | ---: | ---: |"]
            for _, row in frame.iterrows():
                lines.append("| " + row["model"] + " | " + " | ".join(f"{row['test_' + m]:.4f}" for m in METRICS) + " |")
            return "\n".join(lines)

        def metadata_report(frame):
            test = frame[frame["split"] == "test"]
            lines = ["| Test metric | v0 mean ± SD | v1 mean ± SD | Seeds improved |",
                     "| --- | ---: | ---: | ---: |"]
            for _, row in test.iterrows():
                lines.append(f"| {row.label} | {row.v0_mean:.4f} ± {row.v0_std:.4f} | "
                             f"{row.v1_mean:.4f} ± {row.v1_std:.4f} | {int(row.improved_seeds)}/{int(row.n)} |")
            ap = test[test.metric.isin(["pr_auc", "average_precision"])].iloc[0]
            lines += ["", f"Adding metadata and its branch changed mean test AP from {ap.v0_mean:.4f} "
                      f"to {ap.v1_mean:.4f} ({100 * ap.delta_mean:+.2f} percentage points). "
                      "The paired counts above show how consistently each metric improved across seeds."]
            return "\n".join(lines)

        tabular = subset(["HGB scalar", "HGB full"])
        self.write_csv("tabular/comparison.csv", tabular)
        self.small_plot(tabular, "HGB: scalar versus full transit inputs", "tabular/figures/scalar_vs_full")
        self.write("tabular/README.md", "# Tabular baselines\n\n" + report(tabular) +
                   "\n\nBoth models use the same saved candidate pool and star-grouped split. "
                   "The full input includes flattened global/local views and scalars. The full HGB configuration "
                   "was selected on validation from the recorded regularization search. Each entry is one seed.\n\n"
                   "[Metrics](comparison.csv) · [Figure](figures/scalar_vs_full.png) · [PDF](figures/scalar_vs_full.pdf)")

        ablations = subset(["CNN baseline", "CNN wide branch", "CNN regularized"])
        self.write_csv("cnn/ablations.csv", ablations)
        self.small_plot(ablations, "CNN architecture and regularization variants", "cnn/figures/ablations")
        cnn_comparison = self.paired_table("cnn", runs)
        self.copy_figure(self.work / "cnn_comparison/figures/test_comparison", "cnn/figures/v0_vs_v1")
        self.copy_figure(self.work / "cnn_comparison/figures/test_paired_differences", "cnn/figures/paired_differences")
        curve_sources = {}
        for variant in ("v0", "v1"):
            study = self.read_json(self.work / ("cnn_" + variant) / "experiment.json")
            record = next(r for r in study["runs"] if r["seed"] == 42)
            path = Path(record["run_dir"])
            self.remember(path / "history.csv")
            curve_sources["cnn_" + variant] = path.name
            self.copy_figure(Path(study["figure_dir"]) / path.name / "learning_curves", f"cnn/figures/learning_curves_{variant}_seed42")
        self.write("cnn/README.md", "# CNN experiments\n\n" + report(ablations) +
                   "\n\nThe three individual runs compare the baseline CNN, wide-branch variant, and regularized variant. "
                   "The metadata comparison separately includes all five seeds for both v0 and v1.\n\n" +
                   metadata_report(cnn_comparison) + "\n\n"
                   "- [Ablation metrics](ablations.csv) and [plot](figures/ablations.png)\n"
                   "- [v0/v1 summary](comparison_metrics.csv), [all seed metrics](per_seed_metrics.csv), and [paired metrics](paired_seed_metrics.csv)\n"
                   "- [v0/v1 plot](figures/v0_vs_v1.png) and [paired differences](figures/paired_differences.png)\n"
                   "- Seed-42 learning curves: [v0](figures/learning_curves_v0_seed42.png), [v1](figures/learning_curves_v1_seed42.png)\n\n"
                   "Seed 42 illustrates training behavior; it was chosen consistently, without selecting the best test score. "
                   "Training curves use evolving weights/dropout, while validation uses evaluation mode. "
                   "Learning-curve F1 thresholds are tuned separately per split/epoch; final test thresholds come from validation. "
                   "Every PNG has a PDF with the same filename.\n\n" + LIMITS)

        blend = subset(["CNN baseline", "HGB full", "CNN + HGB blend"])
        blend_record = self.read_json(self.work / "blend/run.json")
        blend_summary = self.read_json(Path(blend_record["run_dir"]) / "summary.json")
        self.write_csv("blend/comparison.csv", blend)
        self.write_json("blend/selection.json", {k: blend_summary[k] for k in (
            "method", "selection_metric", "cnn_weight", "hgb_weight", "best_threshold", "val_metrics", "test_metrics")})
        self.small_plot(blend, "CNN/HGB blend: ranking and classification", "blend/figures/metrics",
                        metrics=("pr_auc", "f1", "precision", "recall"))
        error_lines = ["| Model | True positives | False positives | False negatives | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        manifest = self.read_csv(ROOT / "data/processed/kepler/v0/manifest_kepler_v0.csv")
        with np.load(ROOT / "data/processed/kepler/v0/splits_kepler_v0_seed42.npz") as saved:
            positives = int(manifest.iloc[saved["test_idx"]].y.sum())
        for _, row in blend.iterrows():
            error_lines.append(f"| {row['model']} | {positives-int(row.test_false_negatives)} | {int(row.test_false_positives)} | {int(row.test_false_negatives)} | {row.test_precision:.4f} | {row.test_recall:.4f} |")
        cnn_component = blend[blend.model == "CNN baseline"].iloc[0]
        blended = blend[blend.model == "CNN + HGB blend"].iloc[0]
        ap_direction = "improved" if blended.test_pr_auc > cnn_component.test_pr_auc else "decreased" if blended.test_pr_auc < cnn_component.test_pr_auc else "was unchanged"
        f1_direction = "improved" if blended.test_f1 > cnn_component.test_f1 else "decreased" if blended.test_f1 < cnn_component.test_f1 else "was unchanged"
        self.write("blend/README.md", "# CNN and HGB probability blend\n\n" + report(blend) + "\n\n" + "\n".join(error_lines) +
                   f"\n\nAgainst the individual CNN component, test AP {ap_direction} "
                   f"({cnn_component.test_pr_auc:.4f} → {blended.test_pr_auc:.4f}) and test F1 {f1_direction} "
                   f"({cnn_component.test_f1:.4f} → {blended.test_f1:.4f}). Validation-selected blending does not guarantee improvement on every test metric." +
                   f"\n\nValidation F1 selected weights {blend_summary['cnn_weight']:.2f} CNN / {blend_summary['hgb_weight']:.2f} HGB "
                   f"and threshold {blend_summary['best_threshold']:.3f}. Higher recall can come with more false positives; "
                   "inspect AP and F1 separately. The CNN component is the individual baseline, not the five-seed mean or CNN v1.\n\n"
                   "[Full metrics](comparison.csv) · [Selection](selection.json) · [Figure](figures/metrics.png) · [PDF](figures/metrics.pdf)")

        transformer_comparison = self.paired_table("transformer", runs)
        screen = self.read_csv(self.work / "transformer/results/normalization_screen.csv")
        screen = screen[["variant", "seed", "epochs_run", "best_epoch", "use_input_norm", "val_pr_auc"]]
        if len(screen) != 4 or any(sorted(g.seed.tolist()) != [42, 43] for _, g in screen.groupby("variant")):
            raise ValueError("Expected the four planned validation screen runs")
        study = self.read_json(self.work / "transformer/experiment.json")
        for variant, group in screen.groupby("variant"):
            np.testing.assert_allclose(group.val_pr_auc.mean(), study["selection"]["validation_mean_ap"][variant], rtol=1e-10)
        self.write_csv("transformer/normalization_screen.csv", screen)
        self.write_csv("transformer/cnn_transformer_context.csv", self.read_csv(self.work / "transformer/results/cnn_transformer_context.csv"))
        for source, target in (("normalization_screen", "normalization_screen"),
                               ("auxiliary_comparison", "v0_vs_v1"), ("cnn_transformer_context", "cnn_transformer_context")):
            self.copy_figure(self.work / "transformer/figures" / source, "transformer/figures/" + target)
        for variant, source in (("v0", study["selection"]["selected_variant"]), ("v1", "selected_aux")):
            record = next(r for r in study["runs"] if r["seed"] == 42 and r["variant"] == source)
            self.remember(Path(record["run_dir"]) / "history.csv")
            curve_sources["transformer_" + variant] = Path(record["run_dir"]).name
            self.copy_figure(self.work / "transformer/figures/learning_curves" / source / "seed_42/learning_curves",
                             f"transformer/figures/learning_curves_{variant}_seed42")
        means = study["selection"]["validation_mean_ap"]
        self.write("transformer/README.md", "# Transformer normalization and metadata\n\n"
                   f"The two-seed screen gave mean validation AP {means['legacy_input_norm']:.4f} with the original input normalization "
                   f"and {means['bypass_input_norm']:.4f} with its bypass. The final metadata study produced:\n\n" +
                   metadata_report(transformer_comparison) + "\n\n"
                   "- [Validation-only normalization screen](normalization_screen.csv) and [plot](figures/normalization_screen.png)\n"
                   "- [v0/v1 summary](comparison_metrics.csv), [all 12 training runs](per_run_metrics.csv), and [paired final metrics](paired_seed_metrics.csv)\n"
                   "- [Metadata plot](figures/v0_vs_v1.png) and [CNN/transformer context](figures/cnn_transformer_context.png)\n"
                   "- Seed-42 learning curves: [v0](figures/learning_curves_v0_seed42.png), [v1](figures/learning_curves_v1_seed42.png)\n\n"
                   f"Selected normalization: `{study['selection']['selected_variant']}`, using two-seed mean validation AP. "
                   "The screen CSV deliberately contains no test columns. Its two selected runs also belong to the final five-seed v0 group; "
                   "their later test results appear in the full run table. Final test evaluation occurred after all 12 training runs finished. "
                   "The final metadata comparison includes all five seeds per variant.\n\n"
                   "Learning curves always use seed 42. Training metrics use evolving weights/dropout; validation uses evaluation mode. "
                   "Curve F1 thresholds are tuned per split/epoch. Every PNG has a matching PDF.\n\n" + LIMITS)

        summary = pd.read_csv(self.destination / "comparison/summary.csv")
        def score(model, metric):
            return float(summary[(summary.model == model) & (summary.metric == metric)]["mean"].iloc[0])
        cnn_ap = score("CNN v1 + metadata", "pr_auc")
        trans_ap = score("Transformer v1 + metadata", "pr_auc")
        cnn_higher = [label for metric, label in METRICS.items()
                      if metric != "pr_auc" and score("CNN v1 + metadata", metric) > score("Transformer v1 + metadata", metric)]
        model_choice = "CNN v1 is the main practical model used in this project. "
        if cnn_higher:
            model_choice += "Its mean " + ", ".join(cnn_higher) + " exceed transformer v1 in this reproduction. "
        self.write("README.md", "# Reproduced experiment results\n\n"
                   "This directory contains verified results from the completed reproduction: 27 training runs "
                   "(two HGB, three individual CNN, ten CNN seed runs, twelve transformer runs), one blend, "
                   "and ten deferred transformer test evaluations. The overall comparison has 26 final evaluated entries; "
                   "the two unselected transformer screen runs have validation evidence only.\n\n"
                   "| Evidence | Files |\n| --- | --- |\n"
                   "| Overall comparison | [Table](comparison/COMPARISON.md), [plot](comparison/comparison.png), [individual runs](comparison/runs.csv), [means/SDs](comparison/summary.csv) |\n"
                   "| Scalar versus full HGB | [Tabular results](tabular/README.md) |\n"
                   "| CNN variants and metadata | [CNN results](cnn/README.md) |\n"
                   "| Probability blending | [Blend results](blend/README.md) |\n"
                   "| Normalization and transformer metadata | [Transformer results](transformer/README.md) |\n"
                   "| Verification and source hashes | [Provenance](provenance.json) |\n\n"
                   + model_choice + f"Mean test AP is {cnn_ap:.6f} for CNN v1 and {trans_ap:.6f} for transformer v1. "
                   "Compare all four metrics and seed variability before interpreting a small ranking difference. "
                   "The [storyline](../STORYLINE.md) explains the model choice and experimental findings.\n\n"
                   "All planned seeds and unsuccessful alternatives remain in the tables. Learning-curve illustrations use seed 42 consistently. "
                   "Figures are PNG for browsing and PDF for reports. Checkpoints, predictions, and detailed logs remain in local outputs/. "
                   "Paths in CSV/JSON refer to the source workflow relative to the repository; timestamped run IDs preserve traceability.\n\n"
                   "To reproduce this export after training, run from the repository root:\n\n"
                   "```bash\npython scripts/analysis/export_results.py --workflow_dir outputs/reproduce --output_dir outputs/published_results\n```\n\n"
                   "The destination must be new or empty. For an initial publication, use `--output_dir results`. "
                   "The exporter checks processed hashes, saved study evidence, and validation/test predictions, then writes derived tables and plots. "
                   "It does not train models or change the source runs. New training runs can produce slightly different scores.\n\n" + LIMITS)

        self.remember(Path(__file__).resolve())
        outputs = {str(p.relative_to(self.destination)): sha256(p) for p in sorted(self.destination.rglob("*")) if p.is_file()}
        self.write_json("provenance.json", {
            "workflow": self.portable(str(self.work / "workflow.json")),
            "workflow_status": workflow["status"], "execution": workflow["execution"],
            "completed_stages": list(workflow["stages"]), "training_runs": 27,
            "final_evaluated_entries": len(runs), "deferred_transformer_test_evaluations": 10,
            "checks": {"processed_checksums_verified": True, "v0_v1_base_inputs_equal": True,
                       "star_groups_disjoint": True, "prediction_metrics_recomputed": True,
                       "saved_validation_thresholds_applied": True, "all_planned_seeds_present": True,
                       "paired_statistics_recomputed": True, "saved_comparison_tables_match": True,
                       "study_evidence_hashes_verified": True,
                       "transformer_validation_selection_reproduced": transformer_audit["validation_selection_reproduced"],
                       "transformer_test_deferred": transformer_audit["test_evaluation_deferred_until_training_complete"]},
            "learning_curve_selection": {"seed": 42, "rule": "Same fixed seed for both variants and model families; no best-test selection.", "runs": curve_sources},
            "training_source_sha256": workflow["source_sha256"],
            "source_files_sha256": self.sources, "published_files_sha256": outputs,
            "commands": {
                "train": shlex.join([
                    "python", "scripts/experiments/reproduce.py", "--stage", "all",
                    "--device", workflow["execution"]["device"], "--wandb_mode", workflow["execution"]["wandb_mode"],
                    "--output_dir", self.portable(str(self.work)),
                    *(["--gpu_uuids", *workflow["execution"]["gpu_uuids"]] if workflow["execution"]["gpu_uuids"] else []),
                ]),
                "export": shlex.join(["python", "scripts/analysis/export_results.py", "--workflow_dir",
                                      self.portable(str(self.work)), "--output_dir", "results"]),
            },
            "limitations": LIMITS,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow_dir", type=Path, default=ROOT / "outputs/reproduce")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    work, destination = args.workflow_dir.resolve(), args.output_dir.resolve()
    if destination == ROOT or destination in ROOT.parents or destination == work or work in destination.parents or destination in work.parents:
        raise ValueError("The publication destination must be separate from the source workflow")
    for directory in ("data", "scripts", "training", "configs"):
        protected = ROOT / directory
        if destination == protected or protected in destination.parents:
            raise ValueError("Cannot publish into data or source directories")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Publication destination must be new or empty; existing results are preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        Publication(work, staging).build()
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    print("Verified publication exported:", destination)


if __name__ == "__main__":
    main()
