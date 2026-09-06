#!/usr/bin/env python3
"""Verify a completed transformer study and export its controlled comparisons and learning curves."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paths import processed_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from training.transformer import train_kepler_transformer as trainer
from plot_cnn_history import draw_metric, METRICS as CURVE_METRICS, CURVE_NOTE

METRICS = (("accuracy", "Accuracy"), ("f1", "F1"), ("pr_auc", "Average precision"), ("roc_auc", "ROC-AUC"))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#A07800")


def sha(path):
    return hashlib.sha256(processed_path(path).read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def wandb_records(directory):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    files = list((directory/"wandb").glob("offline-run-*/*.wandb"))
    if len(files) != 1:
        raise ValueError(f"Expected one offline W&B file: {directory}")
    reader = DataStore(); reader.open_for_scan(str(files[0]))
    rows, config = [], {}
    try:
        while True:
            raw = reader.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record(); record.ParseFromString(raw)
            configs = ([record.run.config] if record.HasField("run") else []) + ([record.config] if record.HasField("config") else [])
            for update in configs:
                config.update({item.key or "/".join(item.nested_key):json.loads(item.value_json) for item in update.update})
            if record.HasField("history"):
                rows.append({item.key or "/".join(item.nested_key):json.loads(item.value_json) for item in record.history.item})
    finally:
        reader.close()
    return rows, config


def audit_run(record, manifest, splits):
    path = Path(record["run_dir"])
    cfg = json.loads((path/"config.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    history = pd.read_csv(path/"history.csv")
    if record["status"] != "complete" or cfg != record["config"] or cfg["evaluate_test"]:
        raise ValueError("Run is incomplete, differs from its plan, or evaluated test during training")
    if summary["test_evaluated"] or "test_metrics" in summary:
        raise ValueError("Original training summary must record deferred test evaluation")
    np.testing.assert_array_equal(history.epoch, np.arange(1, len(history)+1))
    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("History contains nonfinite values")
    best = history.loc[history.val_pr_auc.idxmax()]
    assert summary["best_epoch"] == int(best.epoch)
    close(summary["best_threshold"], best.val_threshold)
    close(summary["val_metrics"]["pr_auc"], best.val_pr_auc)
    assert len(history) == cfg["epochs"] or len(history)-summary["best_epoch"] == cfg["patience"]
    close(history.learning_rate, cfg["learning_rate"]*(1+np.cos(np.pi*(history.epoch-1)/cfg["epochs"]))/2)
    checkpoint = torch.load(path/"best_model.pt", map_location="cpu")
    assert checkpoint["config"] == cfg and checkpoint["epoch"] == summary["best_epoch"]
    close(checkpoint["val_threshold"], summary["best_threshold"])
    model_config = json.loads((path/"model_config.json").read_text())
    assert model_config == checkpoint["model_config"]
    assert model_config["use_input_norm"] == cfg["use_input_norm"]
    assert model_config["use_aux_branch"] == cfg["use_aux_branch"]
    model = trainer.ExoFormer(**model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    assert all(torch.isfinite(v).all() for v in checkpoint["model_state"].values())
    row = {"variant":record["variant"], "seed":record["seed"], "run_dir":str(path),
           "epochs_run":len(history), "best_epoch":summary["best_epoch"], "threshold":summary["best_threshold"],
           "use_input_norm":cfg["use_input_norm"], "use_aux_branch":cfg["use_aux_branch"],
           "parameter_count":sum(p.numel() for p in model.parameters()),
           "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if cfg["use_aux_branch"]:
        assert checkpoint["aux_preprocessing"] == json.loads((path/"aux_preprocessing.json").read_text())
        assert checkpoint["aux_preprocessing"]["feature_names"] == summary["aux_feature_names"]
        assert model_config["aux_features"] == summary["aux_feature_count"] == 11
    for split in ("train", "val"):
        for metric, _ in METRICS:
            row[split+"_"+metric] = summary[split+"_metrics"][metric]
    if cfg["wandb_mode"] == "offline":
        training_logs, logged_config = wandb_records(path)
        assert all(logged_config[k] == value for k,value in cfg.items())
        close(logged_config["effective_attention_dropout"], cfg["dropout"])
        assert len(training_logs) == len(history)+1
        assert all(not any(k.startswith("test/") for k in r) for r in training_logs)
        for logged, (_, epoch) in zip(training_logs[:-1], history.iterrows()):
            for key,value in epoch.items():
                close(logged[key.replace("train_", "train/", 1).replace("val_", "val/", 1)], value)
        for prefix,split in (("final_train","train"),("final_val","val")):
            for key,value in summary[split+"_metrics"].items():
                close(training_logs[-1][prefix+"/"+key], value)
    evaluation = None
    if record.get("evaluation_status") == "complete":
        evaluation = json.loads((path/"test_evaluation.json").read_text())
        assert evaluation["checkpoint_sha256"] == sha(path/"best_model.pt")
        assert evaluation["training_config_sha256"] == sha(path/"config.json")
        assert evaluation["validation_checkpoint_replay_matches"]
        close(evaluation["best_threshold"], summary["best_threshold"])
        if cfg["wandb_mode"] == "offline":
            evaluation_logs, evaluation_cfg = wandb_records(path/"test_evaluation")
            assert len(evaluation_logs) == 1 and evaluation_cfg["evaluate_test"] is True
            assert evaluation_cfg["parent_run_dir"] == str(path)
            for metric,value in evaluation["test_metrics"].items():
                close(evaluation_logs[0]["test/"+metric], value)
        for metric,_ in METRICS:
            row["test_"+metric] = evaluation["test_metrics"][metric]
    else:
        assert not (path/"test_predictions.csv").exists()
    for split in (["val", "test"] if evaluation else ["val"]):
        prediction = pd.read_csv(path/(split+"_predictions.csv"))
        pd.testing.assert_frame_equal(prediction[manifest.columns], manifest.iloc[splits[split]].reset_index(drop=True))
        assert prediction.candidate_id.is_unique and prediction.probability.between(0,1).all()
        y,p = prediction.y.to_numpy(), prediction.probability.to_numpy()
        labels = p >= summary["best_threshold"]
        np.testing.assert_array_equal(labels.astype(int), prediction.prediction)
        metrics = {"accuracy":accuracy_score(y,labels), "precision":precision_score(y,labels,zero_division=0),
                   "recall":recall_score(y,labels,zero_division=0), "f1":f1_score(y,labels,zero_division=0),
                   "pr_auc":average_precision_score(y,p), "roc_auc":roc_auc_score(y,p)}
        expected = evaluation["test_metrics"] if split == "test" else summary["val_metrics"]
        for key,value in metrics.items():
            close(value,expected[key])
        row[split+"_false_positives"] = int(((y==0)&labels).sum())
        row[split+"_false_negatives"] = int(((y==1)&~labels).sum())
        if split == "val":
            grid=np.linspace(.05,.95,cfg["threshold_grid_size"])
            threshold=grid[np.argmax([f1_score(y,p>=t,zero_division=0) for t in grid])]
            close(threshold,summary["best_threshold"])
    sources = {str(p):sha(p) for p in path.iterdir() if p.is_file()}
    return row, history, sources


def save_figure(fig, directory, name):
    for suffix in ("png","pdf"):
        fig.savefig(directory/(name+"."+suffix),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def learning_curves(record, row, history, figures):
    destination=figures/"learning_curves"/record["variant"]/f"seed_{record['seed']}"
    destination.mkdir(parents=True)
    fig,axes=plt.subplots(1,3,figsize=(12,5))
    for ax,(metric,label,_) in zip(axes,CURVE_METRICS):
        draw_metric(ax,history,metric,label,row["best_epoch"])
    fig.suptitle("Kepler transformer learning curves",fontsize=15,fontweight="bold")
    fig.text(.5,.91,f"{record['variant']} · seed {record['seed']} · float32 training",ha="center",fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.87),ncol=3,frameon=False)
    fig.text(.5,.025,CURVE_NOTE,ha="center",fontsize=9)
    fig.subplots_adjust(top=.73,bottom=.24,wspace=.28)
    save_figure(fig,destination,"learning_curves")


def paired_plot(first, second, figures, name, title, labels, split="test"):
    pairs=first.merge(second,on="seed",suffixes=("_first","_second"),validate="one_to_one")
    fig,axes=plt.subplots(1,4,figsize=(12,5.5),sharey=True)
    limits=[]
    for ax,(metric,label) in zip(axes,METRICS):
        left=pairs[split+"_"+metric+"_first"].to_numpy()
        right=pairs[split+"_"+metric+"_second"].to_numpy()
        for i,(seed,x,y) in enumerate(zip(pairs.seed,left,right)):
            ax.plot([0,1],[x,y],"o-",color=COLORS[i%len(COLORS)],alpha=.85,markersize=4,label=f"Seed {seed}")
        means=[left.mean(),right.mean()];stds=[left.std(ddof=1),right.std(ddof=1)]
        ax.errorbar([.17,1.17],means,yerr=stds,fmt="D",color="#222222",capsize=5,label="Mean ± sample SD",zorder=5)
        limits.extend(left);limits.extend(right);limits.extend(np.array(means)-stds);limits.extend(np.array(means)+stds)
        ax.set(xticks=[0,1],xticklabels=labels,xlim=(-.3,1.4),title=label)
        ax.grid(axis="y",alpha=.25);ax.spines[["top","right"]].set_visible(False)
    axes[0].set(ylim=(max(0,min(limits)-.025),min(1.01,max(limits)+.02)),ylabel=split.capitalize()+" score")
    fig.suptitle(title,fontsize=15,fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
    fig.text(.5,.04,"Lines join matching seeds. Black diamonds: mean ± sample SD (not confidence intervals).\n"
             "Same candidates and split; checkpoints and thresholds selected on validation. Score axis is truncated.",ha="center",fontsize=9)
    fig.subplots_adjust(top=.73,bottom=.22,wspace=.2)
    save_figure(fig,figures,name)


def statistics_table(first,second):
    paired=first.merge(second,on="seed",suffixes=("_v0","_v1"),validate="one_to_one")
    rows=[]
    for split in ("train","val","test"):
        for metric,label in METRICS:
            key=split+"_"+metric
            a,b=paired[key+"_v0"],paired[key+"_v1"]
            delta=b-a;paired[key+"_delta"]=delta
            rows.append({"split":split,"metric":metric,"label":label,"n":len(paired),
                "v0_mean":a.mean(),"v0_std":a.std(ddof=1),"v1_mean":b.mean(),"v1_std":b.std(ddof=1),
                "delta_mean":delta.mean(),"delta_std":delta.std(ddof=1),"improved_seeds":int((delta>0).sum())})
    return paired,pd.DataFrame(rows)


def cnn_context(first,second,figures,sources,cnn_paths):
    groups={"Transformer v0":first,"Transformer v1":second}
    for label,path in zip(("CNN v0","CNN v1"),cnn_paths):
        sources[str(path)]=sha(path)
        experiment=json.loads(path.read_text())
        assert experiment["status"]=="complete" and experiment["seeds"]==[42,43,44,45,46]
        rows=[]
        for run in experiment["runs"]:
            path=Path(run["run_dir"])/"summary.json";sources[str(path)]=sha(path)
            summary=json.loads(path.read_text())
            rows.append({"seed":run["seed"],**{"test_"+k:v for k,v in summary["test_metrics"].items()}})
        groups[label]=pd.DataFrame(rows)
    rows=[]
    for label,values in groups.items():
        for metric,display in METRICS:
            scores=values["test_"+metric]
            rows.append({"model":label,"metric":metric,"label":display,"mean":scores.mean(),"std":scores.std(ddof=1),"n":len(scores)})
    fig,ax=plt.subplots(figsize=(10,5.5))
    labels=[label for label in ["CNN v0","Transformer v0","CNN v1","Transformer v1"] if label in groups]
    for i,label in enumerate(labels):
        scores=groups[label].sort_values("seed").test_pr_auc.to_numpy()
        ax.scatter(i+np.linspace(-.12,.12,len(scores)),scores,c=COLORS,s=35,zorder=3)
        ax.errorbar(i+.25,scores.mean(),yerr=scores.std(ddof=1),fmt="D",color="#222222",capsize=5)
    ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylabel="Test average precision",title="CNN and transformer: existing five-seed results")
    ax.grid(axis="y",alpha=.25);ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.04,"Same Kepler split. Each architecture uses its own fixed training recipe and precision mode.\n"
        "Points: individual seeds; diamonds: mean ± sample SD. Descriptive comparison, not an isolated architecture test.",ha="center",fontsize=9)
    fig.subplots_adjust(bottom=.23)
    save_figure(fig,figures,"cnn_transformer_context")
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_dir",type=Path,required=True)
    parser.add_argument("--figure_dir",type=Path,required=True)
    parser.add_argument("--output_dir",type=Path,help="New results directory; default: <study_dir>/results")
    parser.add_argument("--cnn_v0",type=Path)
    parser.add_argument("--cnn_v1",type=Path)
    args=parser.parse_args()
    if bool(args.cnn_v0) != bool(args.cnn_v1):
        parser.error("Provide both CNN experiment paths or neither")
    group=args.study_dir.resolve();figures=args.figure_dir.resolve();output=(args.output_dir or group/"results").resolve()
    if output.exists() or figures.exists():
        raise ValueError("Use new result and figure directories to preserve earlier exports")
    torch.set_num_threads(1)
    experiment=json.loads((group/"experiment.json").read_text())
    protocol=json.loads((group/"protocol.json").read_text())
    assert experiment["status"]=="complete" and len(experiment["runs"])==12
    protected={}
    for path,expected in {**protocol["protected_files_sha256"],**experiment["source_sha256"]}.items():
        candidates=[processed_path(path),group/"source"/Path(path).name]
        verified=next((p for p in candidates if p.is_file() and sha(p)==expected),None)
        if verified is None:
            raise ValueError(f"Source changed and no matching snapshot exists: {path}")
        protected[str(verified)]=expected
    selection=json.loads((group/"selection.json").read_text())
    assert selection==experiment["selection"] and selection["test_results_available"] is False
    selected=selection["selected_variant"]
    reference=experiment["runs"][0]["config"]
    manifest=pd.read_csv(processed_path(reference["manifest_path"]))
    with np.load(processed_path(reference["splits_path"])) as saved:
        splits={split:saved[split+"_idx"].copy() for split in ("train","val","test")}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))),np.arange(len(manifest)))
    stars=[set(manifest.iloc[idx].kepid) for idx in splits.values()]
    assert all(not stars[i]&stars[j] for i in range(3) for j in range(i))
    allowed={"seed","output_root","use_input_norm","use_aux_branch","dataset_path","splits_path","manifest_path","prepare_version"}
    fixed={k:v for k,v in reference.items() if k not in allowed}
    rows=[];histories=[];sources=dict(protected)
    for record in experiment["runs"]:
        cfg=record["config"]
        assert cfg["use_amp"] is False and {k:v for k,v in cfg.items() if k not in allowed}==fixed
        assert record["gpu_uuid"]==experiment["gpu_by_seed"][str(record["seed"])]
        for key in ("manifest_path","splits_path"):
            assert processed_path(cfg[key]).read_bytes()==processed_path(reference[key]).read_bytes()
        if record.get("evaluation_status"):
            assert record["evaluation_status"]=="complete"
            assert record["evaluation_started_at"]>=experiment["all_training_completed_at"]>selection["frozen_at"]
        row,history,run_sources=audit_run(record,manifest,splits)
        rows.append(row);histories.append(history);sources.update(run_sources)
        print("Verified",record["variant"],record["seed"],flush=True)
    data=pd.DataFrame(rows)
    screen=data[data.variant.isin(["legacy_input_norm","bypass_input_norm"])&data.seed.isin(protocol["screen_seeds"])].copy()
    assert len(screen)==4
    for variant in ("legacy_input_norm","bypass_input_norm"):
        close(screen[screen.variant==variant].val_pr_auc.mean(),selection["validation_mean_ap"][variant])
    expected="bypass_input_norm" if selection["validation_mean_ap"]["bypass_input_norm"]>selection["validation_mean_ap"]["legacy_input_norm"] else "legacy_input_norm"
    assert expected==selected
    first=data[data.variant==selected].sort_values("seed")
    second=data[data.variant=="selected_aux"].sort_values("seed")
    assert first.seed.tolist()==second.seed.tolist()==protocol["final_seeds"]==[42,43,44,45,46]
    assert len(first)==len(second)==5
    paired,stats=statistics_table(first,second)
    output.mkdir(parents=True);figures.mkdir(parents=True)
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42}):
        for record,row,history in zip(experiment["runs"],rows,histories):
            learning_curves(record,row,history,figures)
        paired_plot(screen[screen.variant=="legacy_input_norm"],screen[screen.variant=="bypass_input_norm"],
            figures,"normalization_screen","Transformer input normalization: validation screen",["Original","Bypassed"],split="val")
        paired_plot(first,second,figures,"auxiliary_comparison","Transformer: matched auxiliary-metadata comparison",["v0","v1 + aux"])
        context=cnn_context(first,second,figures,sources,[args.cnn_v0,args.cnn_v1] if args.cnn_v0 else [])
    data.to_csv(output/"per_run_metrics.csv",index=False)
    screen.to_csv(output/"normalization_screen.csv",index=False)
    paired.to_csv(output/"paired_seed_metrics.csv",index=False)
    stats.to_csv(output/"comparison_metrics.csv",index=False)
    context.to_csv(output/"cnn_transformer_context.csv",index=False)
    captions="""# Transformer study figures

Learning curves: each of the 12 completed runs, with training/validation loss, AP, and F1. The dotted line marks the validation-AP-selected checkpoint. Training uses evolving weights and dropout; validation uses evaluation mode. F1 thresholds are tuned separately each epoch. Curves are unsmoothed.

normalization_screen: paired validation scores for seeds 42 and 43, comparing original input LayerNorm with its bypass. Both use float32 and the same remaining settings. These are development results used to choose the normalization mode; no legacy-screen test comparison was performed.

auxiliary_comparison: paired test metrics for the selected v0 transformer and its auxiliary extension, across all five seeds. Test evaluation was deferred until training and configuration selection finished.

cnn_transformer_context: descriptive test-AP comparison with the existing CNN experiments. Candidates and split match, but architectures use different training recipes, precision modes, and physical GPUs. This is not a controlled architecture-only ablation.

All summary diamonds show mean ± sample SD (ddof=1), not confidence intervals. Lines join matching seed numbers; architectures can consume random numbers differently. Score axes are truncated. Every figure is available as PNG and vector PDF.
"""
    (figures/"CAPTIONS.md").write_text(captions)
    report=["# Transformer normalization and auxiliary-metadata study","",
        "## Validation-only normalization screen","",
        "| Seed | Original input LayerNorm AP | Bypassed input LayerNorm AP | Change (pp) |",
        "| --- | --- | --- | --- |"]
    for seed in protocol["screen_seeds"]:
        part=screen[screen.seed==seed].set_index("variant")
        a=part.loc["legacy_input_norm","val_pr_auc"];b=part.loc["bypass_input_norm","val_pr_auc"]
        report.append(f"| {seed} | {a:.6f} | {b:.6f} | {(b-a)*100:+.2f} |")
    report += ["",f"Selected using mean validation AP: `{selected}`. No test metrics were available for this choice.","",
        "## Fixed five-seed metadata comparison","",
        "| Test metric | Transformer v0 mean ± SD | Transformer v1 mean ± SD | Paired change ± SD (pp) | Improved seeds |",
        "| --- | --- | --- | --- | --- |"]
    for row in stats[stats.split=="test"].itertuples():
        report.append(f"| {row.label} | {row.v0_mean:.4f} ± {row.v0_std:.4f} | {row.v1_mean:.4f} ± {row.v1_std:.4f} | "
                      f"{row.delta_mean*100:+.2f} ± {row.delta_std*100:.2f} | {row.improved_seeds}/{row.n} |")
    report += ["","## Scope and interpretation","",
        "The normalization comparison changes only whether initial LayerNorm(3) is applied. It preserves initial shared weights/RNG and holds effective dropout, encoder initialization, optimizer, schedule, and precision fixed. Its evidence is limited to two validation seeds; it does not establish a five-seed test effect of normalization.","",
        "The final metadata comparison uses five matched seeds, identical candidates/split/shared transformer settings, and the selected normalization mode. The extension changes both inputs and model capacity; individual feature groups and their physical mechanisms are not isolated.","",
        "The same Kepler split was consulted during earlier development. These SDs describe training-seed variability, not confidence intervals or new-dataset uncertainty. Catalog-assisted classification is not blind light-curve discovery or TESS transfer.","",
        "All 12 study runs use float32. CNN runs use their own fixed training recipe and precision mode, so comparisons between architectures are descriptive.","",
        f"Learning curves and comparison figures: {figures}",""]
    (output/"REPORT_SUMMARY.md").write_text("\n".join(report))
    from PIL import Image
    exports=[]
    for path in figures.rglob("*"):
        if path.suffix==".png":
            with Image.open(path) as picture:picture.verify()
            exports.append(path)
        elif path.suffix==".pdf":
            assert path.read_bytes().startswith(b"%PDF-");exports.append(path)
    assert len(exports)==30
    for path,expected in sources.items():
        assert sha(path)==expected,path
    metadata={"verified_at":datetime.now(timezone.utc).isoformat(),"completed_training_runs":12,"final_test_runs":10,
        "test_evaluations_during_screening":0,"source_files_sha256":sources,
        "protected_artifacts_unchanged":len(protocol["protected_files_sha256"]),"checkpoints_predictions_verified":True,"wandb_verified":reference["wandb_mode"]=="offline",
        "validation_selection_reproduced":True,"test_evaluation_deferred_until_training_complete":True,
        "figure_exports_verified":len(exports),"selection":selection,
        "export_sha256":{str(p):sha(p) for p in [*exports,*output.iterdir()] if p.is_file()}}
    (output/"verification.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(stats[stats.split=="test"].to_string(index=False))
    print("Verified study. Results:",output,"Figures:",figures)


if __name__=="__main__":
    main()
