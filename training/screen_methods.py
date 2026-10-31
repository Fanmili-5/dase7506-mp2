"""Screen fixed candidates using only the official training split.

Fit/tune/holdout are question-level partitions. No validation/test bank or
reference is opened. This is exploratory screening, not a frozen final method.
"""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# No concurrent fits; BLAS stays single-threaded. XGBoost explicitly uses <=4.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "methods"))
from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from selector import answers_for, candidate_margins, select_choices
from probe_environment import collect


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_training():
    check_data(DATA, ("train",))
    bank, arrays = load_inputs("train")
    refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    if len(bank) != len(refs):
        raise ValueError("Training reference count mismatch")
    answers = answers_for(bank)
    labels = []
    for row, ref, parsed in zip(bank, refs, answers):
        if row["id"] != ref["id"] or row["question"] != ref["question"]:
            raise ValueError("Training references are not aligned")
        gold = parse_answer(ref["answer"])
        if gold is None:
            raise ValueError("Unparseable training reference")
        label = [a is not None and a == gold for a in parsed]
        if not any(label):
            raise ValueError("No positive training candidate")
        labels.append(label)
    hard = np.array([len({a for a in parsed if a is not None}) >= 2 for parsed in answers])
    return bank, arrays["embeddings"], np.asarray(labels, dtype=np.float32), hard


def partition_questions(count, config):
    fit_end = int(count * config["fit_fraction"])
    tune_end = fit_end + int(count * config["tune_fraction"])
    if not 0 < fit_end < tune_end < count:
        raise ValueError("Need nonempty fit, tune and holdout partitions")
    order = np.random.default_rng(config["split_seed"]).permutation(count)
    return {"fit": order[:fit_end], "tune": order[fit_end:tune_end], "holdout": order[tune_end:]}


def torch_loss(scores, labels, objective):
    import torch
    from torch.nn import functional as F

    if objective == "bce":
        return F.binary_cross_entropy_with_logits(scores, labels)
    if objective != "pairwise":
        raise ValueError("Unknown neural objective")
    # Axis 1 is the positive candidate; axis 2 is the negative candidate.
    mask = labels[:, :, None] * (1 - labels[:, None, :])
    counts = mask.sum(dim=(1, 2))
    values = F.softplus(scores[:, None, :] - scores[:, :, None])
    per_question = (values * mask).sum(dim=(1, 2)) / counts.clamp(min=1)
    valid = counts > 0
    return per_question[valid].mean() if valid.any() else scores.sum() * 0


def fit_neural(x_fit, y_fit, x_tune, y_tune, config, seed, device, bundle, mean, scale):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; run probe_environment.py --probe-torch")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if config["hidden"]:
        model = nn.Sequential(nn.Linear(1024, config["hidden"]), nn.ReLU(),
                              nn.Dropout(config["dropout"]), nn.Linear(config["hidden"], 1))
    else:
        model = nn.Sequential(nn.Linear(1024, 1))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    fit_x = torch.as_tensor(x_fit, device=device)
    fit_y = torch.as_tensor(y_fit, device=device)
    tune_x = torch.as_tensor(x_tune, device=device)
    tune_y = torch.as_tensor(y_tune, device=device)
    rng = np.random.default_rng(seed)
    history, best_state = [], None
    best_loss, best_epoch = float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(fit_x))
        losses = []
        for offset in range(0, len(order), config["batch_questions"]):
            indices = torch.as_tensor(order[offset:offset + config["batch_questions"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch_loss(model(fit_x[indices]).squeeze(-1), fit_y[indices], config["objective"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(indices)))
        model.eval()
        with torch.no_grad():
            tune_loss = float(torch_loss(model(tune_x).squeeze(-1), tune_y, config["objective"]).cpu())
        if not np.isfinite(tune_loss):
            raise RuntimeError("Nonfinite tuning loss")
        history.append({"epoch": epoch,
                        "fit_loss": sum(v * n for v, n in losses) / len(fit_x),
                        "tune_loss": tune_loss})
        if tune_loss < best_loss - 1e-7:
            best_loss, best_epoch = tune_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'{config["name"]}: epoch={epoch}, tune_loss={tune_loss:.5f}', flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        reference_margins = model(tune_x).squeeze(-1).cpu().numpy()
    bundle.mkdir(parents=True)
    weights = {"mean": mean, "scale": scale}
    for i, layer in enumerate(m for m in model.modules() if isinstance(m, nn.Linear)):
        weights[f"w{i}"] = layer.weight.detach().cpu().numpy().T.copy()
        weights[f"b{i}"] = layer.bias.detach().cpu().numpy().copy()
    np.savez(bundle / "weights.npz", **weights)
    write_json(bundle / "config.json", {**config, "format_version": 1, "aggregation": "max"})
    return {"fit_seconds": time.perf_counter() - start, "best_epoch": best_epoch,
            "best_tune_loss": best_loss, "history": history,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "device": device}, reference_margins


def fit_xgboost(x_fit, y_fit, x_tune, y_tune, config, seed, bundle, mean, scale):
    import xgboost as xgb

    start = time.perf_counter()
    fit = xgb.DMatrix(x_fit.reshape(-1, 1024), label=y_fit.ravel(), nthread=1)
    tune = xgb.DMatrix(x_tune.reshape(-1, 1024), label=y_tune.ravel(), nthread=1)
    history = {}
    model = xgb.train({**config["params"], "seed": seed}, fit,
                      num_boost_round=config["rounds"], evals=[(tune, "tune")],
                      early_stopping_rounds=config["patience"], evals_result=history,
                      verbose_eval=False)
    best_rounds = model.best_iteration + 1
    best_loss = float(model.best_score)
    model = model[:best_rounds]
    reference_margins = model.predict(tune, output_margin=True).reshape(-1, 4)
    bundle.mkdir(parents=True)
    model.save_model(bundle / "model.ubj")
    np.savez(bundle / "weights.npz", mean=mean, scale=scale)
    write_json(bundle / "config.json", {**config, "format_version": 1, "aggregation": "max"})
    return {"fit_seconds": time.perf_counter() - start, "best_rounds": best_rounds,
            "best_tune_loss": best_loss, "history": history, "device": "cpu"}, reference_margins


def summarize(bank, labels, hard, choices, majority):
    selected = np.asarray(choices, dtype=int)
    correct = labels[np.arange(len(bank)), selected].astype(bool)
    base_correct = labels[np.arange(len(bank)), majority].astype(bool)
    result = {}
    for name, mask in (("all", np.ones(len(bank), dtype=bool)), ("disagreement", hard)):
        differences = correct[mask].astype(float) - base_correct[mask].astype(float)
        row = {"questions": int(mask.sum()), "correct": int(correct[mask].sum()),
               "accuracy_percent": float(100 * correct[mask].mean()) if mask.any() else None,
               "rescued_majority_errors": int((correct & ~base_correct & mask).sum()),
               "lost_majority_correct": int((~correct & base_correct & mask).sum())}
        if len(differences):
            rng = np.random.default_rng(9017)
            means = differences[rng.integers(len(differences), size=(2000, len(differences)))].mean(axis=1)
            row["paired_delta_vs_majority_pp"] = float(100 * differences.mean())
            row["exploratory_bootstrap_95_percent_interval_pp"] = (100 * np.quantile(means, [0.025, 0.975])).tolist()
        result[name] = row
    result["choices"] = choices
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/screen_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or np.__version__ != "2.5.3":
        raise RuntimeError("Screening requires Python 3.12 and the official NumPy 2.5.3")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["training_groups"] != "at_least_two_distinct_valid_answers":
        raise ValueError("This screening version supports disagreement-group fitting only")
    models = config["models"]
    if any(m["family"] == "xgboost" and not 1 <= m["params"].get("nthread", 1) <= 4 for m in models):
        raise ValueError("XGBoost must use one to four threads")
    if args.models:
        if set(args.models) - {m["name"] for m in models}:
            parser.error("Unknown requested model")
        models = [m for m in models if m["name"] in args.models]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bank, embeddings, labels, hard = load_training()
    partitions = partition_questions(len(bank), config)
    # The same input-defined disagreement filter applies to every model.
    fit_indices = partitions["fit"][hard[partitions["fit"]]]
    tune_indices = partitions["tune"][hard[partitions["tune"]]]
    holdout = partitions["holdout"]
    if not len(fit_indices) or not len(tune_indices):
        raise ValueError("Disagreement training/tuning partition is empty")
    fit_flat = embeddings[fit_indices].reshape(-1, 1024)
    mean = fit_flat.mean(axis=0).astype(np.float32)
    scale = np.maximum(fit_flat.std(axis=0), 1e-6).astype(np.float32)
    x_fit = ((embeddings[fit_indices] - mean) / scale).astype(np.float32)
    x_tune = ((embeddings[tune_indices] - mean) / scale).astype(np.float32)
    holdout_bank = [bank[int(i)] for i in holdout]
    majority = select_choices(holdout_bank)
    code_paths = [Path(__file__), ROOT / "methods/selector.py", ROOT / "code/common.py", args.config]
    report = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "purpose": "exploratory_train_only_method_screening",
        "protocol": PROTOCOL, "source_split": "train", "test_scored": False,
        "validation_opened": False, "seed": config["model_seed"], "config": config,
        "environment": collect(probe_torch=False),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "training_files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (DATA / "gsm8k_train.jsonl", DATA / "candidates/gsm8k_train.jsonl",
                                            DATA / "candidates/gsm8k_train.npz", DATA / "manifest.json")},
        "training_question_count": len(fit_indices), "tuning_question_count": len(tune_indices),
        "partitions": {k: {"questions": len(v), "disagreement_questions": int(hard[v].sum()),
                            "ids": [bank[int(i)]["id"] for i in v]} for k, v in partitions.items()},
        "holdout_ids": [row["id"] for row in holdout_bank],
        "baseline": summarize(holdout_bank, labels[holdout], hard[holdout], majority, majority),
        "models": {},
        "limitations": ["One split and one model seed; intervals are exploratory, without multiple-comparison correction",
                        "These are training-split holdout results, not official validation/test scores",
                        "All models train on disagreement groups; all-groups training remains an ablation",
                        "Asset bytes are measured; full official CPU evaluation time remains unmeasured"]}
    report_path = args.output / "screen_results.json"
    write_json(report_path, report)
    for model_config in models:
        name = model_config["name"]
        print(f"Starting {name}; fit questions={len(fit_indices)}", flush=True)
        bundle = args.output / name / "bundle"
        try:
            if model_config["family"] == "neural":
                details, reference = fit_neural(x_fit, labels[fit_indices], x_tune, labels[tune_indices],
                                              model_config, config["model_seed"], args.device, bundle, mean, scale)
            else:
                details, reference = fit_xgboost(x_fit, labels[fit_indices], x_tune, labels[tune_indices],
                                               model_config, config["model_seed"], bundle, mean, scale)
            exported, _ = candidate_margins({"embeddings": embeddings[tune_indices]}, bundle)
            np.testing.assert_allclose(exported, reference, rtol=2e-4, atol=2e-4,
                                       err_msg="Exported CPU inference differs from training backend")
            margins, _ = candidate_margins({"embeddings": embeddings[holdout]}, bundle)
            aggregations = ["max", "weighted_vote"] if model_config["objective"] == "bce" else ["max"]
            details["evaluation"] = {a: summarize(holdout_bank, labels[holdout], hard[holdout],
                                                 select_choices(holdout_bank, margins, a), majority)
                                      for a in aggregations}
            details["export_max_abs_error"] = float(np.abs(exported - reference).max())
            details["inference_assets_bytes"] = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
            details["assets_within_16_MiB"] = details["inference_assets_bytes"] <= 16 * 1024**2
            details["asset_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in bundle.iterdir() if p.is_file()}
            report["models"][name] = details
            print(f"Finished {name}: {json.dumps({a: v['disagreement']['accuracy_percent'] for a, v in details['evaluation'].items()})}", flush=True)
        except Exception as error:
            report["models"][name] = {"error": repr(error)}
            write_json(report_path, report)
            raise
        write_json(report_path, report)
    report["total_wall_seconds"] = time.perf_counter() - started
    write_json(report_path, report)
    print(f"Screen results: {report_path.resolve()}")


if __name__ == "__main__":
    main()
