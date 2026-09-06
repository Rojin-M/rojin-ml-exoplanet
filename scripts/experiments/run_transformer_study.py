#!/usr/bin/env python3
"""Screen input normalization on validation, then compare a fixed transformer with/without metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paths import configuration_paths
from training.transformer.train_kepler_transformer import TrainConfig


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_config", type=Path, required=True)
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument("--gpu_uuids", nargs="+", help="Optional distinct GPU UUIDs; otherwise run sequentially on --device")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/transformer_protocol.json")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline"), default="offline")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", default=None)
    args = parser.parse_args()
    group = args.study_dir.resolve()
    if group.exists():
        raise ValueError("Use a new study directory to preserve previous runs")
    protocol = json.loads(args.protocol.read_text())
    if args.gpu_uuids and len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        raise ValueError("GPU UUIDs must be distinct")
    if args.gpu_uuids and args.device == "cpu":
        raise ValueError("GPU UUIDs cannot be combined with --device cpu")
    source = configuration_paths(asdict(TrainConfig(**json.loads(args.source_config.read_text()))))
    precision = source["use_amp"] if args.use_amp is None else args.use_amp
    source.update(device="cuda:0" if args.gpu_uuids else args.device, evaluate_test=False,
                  wandb_mode=args.wandb_mode, wandb_group=group.name, use_amp=precision)
    if protocol.get("use_amp") != precision:
        raise ValueError("Precision mode differs from the frozen study protocol")
    if protocol["screen_seeds"] != [42,43] or protocol["final_seeds"] != [42,43,44,45,46]:
        raise ValueError("This study uses the fixed two-seed screen and five-seed comparison")
    devices = args.gpu_uuids or [source["device"]]
    workers = len(devices)
    group.mkdir(parents=True)
    protocol["protected_files_sha256"] = {}
    write(group/"protocol.json", protocol)
    sources = [Path(__file__).resolve(), args.source_config.resolve(), ROOT/"training/transformer/train_kepler_transformer.py",
               ROOT/"training/cnn/train_kepler_cnn.py", ROOT/"scripts/experiments/evaluate_transformer_checkpoint.py", group/"protocol.json"]
    sources += [ROOT/"scripts/paths.py"]
    sources += [Path(source[key]) for key in ("dataset_path","splits_path","manifest_path")]
    v1 = ROOT/"data/processed/kepler/v1"
    aux_paths = {"dataset_path":str(next(v1.glob("dataset_kepler_v1_*.npz"))),
                 "splits_path":str(v1/"splits_kepler_v1_seed42.npz"), "manifest_path":str(v1/"manifest_kepler_v1.csv")}
    sources += [Path(path) for path in aux_paths.values()]
    hashes = {str(path):sha(path) for path in sources}
    source_dir = group/"source"; source_dir.mkdir()
    for path in sources[:6]:
        shutil.copy2(path,source_dir/path.name)
    (group/"requirements-frozen.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    if shutil.which("nvidia-smi"):
        (group/"gpu_environment.csv").write_text(subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv"],text=True))
    gpu_map = {str(seed):devices[i%workers] for i,seed in enumerate(protocol["final_seeds"])}
    experiment = {"status":"screening","created_at":now(),"protocol":str(group/"protocol.json"),
                  "source_sha256":hashes,"gpu_by_seed":gpu_map,"runs":[]}
    manifest = group/"experiment.json"
    write(manifest,experiment)

    def unchanged():
        for path,expected in {**protocol["protected_files_sha256"],**hashes}.items():
            if sha(path) != expected:
                raise RuntimeError(f"Protected source/artifact changed: {path}")

    def prepare(variant,seed,use_norm,auxiliary,phase):
        directory = group/variant/f"seed_{seed}"
        directory.mkdir(parents=True,exist_ok=False)
        cfg = dict(source,seed=seed,output_root=str(directory),use_input_norm=use_norm,
                   use_aux_branch=auxiliary,aux_dropout=0.1)
        if auxiliary:
            cfg.update(**aux_paths,prepare_version="v1")
        path=directory/"requested_config.json";write(path,cfg)
        record={"variant":variant,"seed":seed,"phase":phase,"status":"pending","config":cfg,
                "gpu_uuid":gpu_map[str(seed)],"requested_config":str(path),"log_path":str(directory/"training.log")}
        experiment["runs"].append(record)
        return record

    def execute(record,evaluation=False):
        environment=os.environ.copy()
        if args.gpu_uuids:
            environment["CUDA_VISIBLE_DEVICES"]=record["gpu_uuid"]
        if evaluation:
            command=[sys.executable,"-u",str(ROOT/"scripts/experiments/evaluate_transformer_checkpoint.py"),
                     "--run_dir",record["run_dir"],"--device",source["device"]]
            log_path=Path(record["config"]["output_root"])/"test_evaluation.log"
        else:
            code="import json,sys; from training.transformer.train_kepler_transformer import TrainConfig,train_model; train_model(TrainConfig(**json.load(open(sys.argv[1]))))"
            command=[sys.executable,"-u","-c",code,record["requested_config"]]
            log_path=Path(record["log_path"])
        with log_path.open("w") as log:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        if evaluation:
            return record["run_dir"]
        summaries=list(Path(record["config"]["output_root"]).glob("*/summary.json"))
        if len(summaries)!=1:
            raise RuntimeError("Expected exactly one completed run")
        path=summaries[0].parent
        summary=json.loads(summaries[0].read_text())
        if summary.get("test_evaluated") is not False or (path/"test_predictions.csv").exists():
            raise RuntimeError("A training run evaluated test before the study was frozen")
        if json.loads((path/"config.json").read_text()) != record["config"]:
            raise RuntimeError("Saved configuration differs from plan")
        return str(path)

    def wave(records,evaluation=False):
        unchanged()
        if len({r["gpu_uuid"] for r in records}) != len(records):
            raise RuntimeError("A wave must assign one job per GPU")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={}
            for record in records:
                record["evaluation_status" if evaluation else "status"]="running"
                record["evaluation_started_at" if evaluation else "started_at"]=now()
                futures[pool.submit(execute,record,evaluation)]=record
                print("Evaluating" if evaluation else "Training",record["variant"],record["seed"],flush=True)
            write(manifest,experiment)
            for future in as_completed(futures):
                record=futures[future]
                try:
                    path=future.result()
                except BaseException as exc:
                    record["evaluation_status" if evaluation else "status"]="failed"
                    record["error"]=repr(exc);write(manifest,experiment)
                    raise
                record["run_dir"]=path
                record["evaluation_status" if evaluation else "status"]="complete"
                record["evaluation_completed_at" if evaluation else "completed_at"]=now()
                write(manifest,experiment)
                print("Completed",record["variant"],record["seed"],"evaluation" if evaluation else "training",flush=True)
        unchanged()

    try:
        for variant,use_norm in [("legacy_input_norm",True),("bypass_input_norm",False)]:
            for start in range(0,len(protocol["screen_seeds"]),workers):
                wave([prepare(variant,seed,use_norm,False,"screen") for seed in protocol["screen_seeds"][start:start+workers]])
        scores={}
        for variant in ("legacy_input_norm","bypass_input_norm"):
            values=[json.loads((Path(r["run_dir"])/"summary.json").read_text())["val_metrics"]["pr_auc"]
                    for r in experiment["runs"] if r["variant"]==variant]
            scores[variant]=sum(values)/len(values)
        selected="bypass_input_norm" if scores["bypass_input_norm"]>scores["legacy_input_norm"] else "legacy_input_norm"
        selection={"frozen_at":now(),"criterion":"mean selected-checkpoint validation AP; ties retain legacy",
                   "screen_seeds":protocol["screen_seeds"],"validation_mean_ap":scores,"selected_variant":selected,
                   "test_results_available":False}
        write(group/"selection.json",selection)
        experiment.update(status="final_training",selection=selection);write(manifest,experiment)
        print("Validation selection frozen:",json.dumps(selection),flush=True)
        use_norm=selected=="legacy_input_norm"
        missing=[s for s in protocol["final_seeds"] if s not in protocol["screen_seeds"]]
        for start in range(0,len(missing),workers):
            wave([prepare(selected,s,use_norm,False,"final") for s in missing[start:start+workers]])
        for start in range(0,len(protocol["final_seeds"]),workers):
            wave([prepare("selected_aux",s,use_norm,True,"final") for s in protocol["final_seeds"][start:start+workers]])
        experiment["status"]="test_evaluation";experiment["all_training_completed_at"]=now();write(manifest,experiment)
        for variant in (selected,"selected_aux"):
            records=sorted([r for r in experiment["runs"] if r["variant"]==variant],key=lambda r:r["seed"])
            for start in range(0,len(records),workers):wave(records[start:start+workers],evaluation=True)
        unchanged()
        experiment.update(status="complete",completed_at=now(),sources_unchanged=True)
        write(manifest,experiment)
        print("Transformer study training/evaluation complete:",manifest,flush=True)
    except BaseException as exc:
        experiment.update(status="failed",error=repr(exc),stopped_at=now());write(manifest,experiment)
        raise


if __name__ == "__main__":
    main()
