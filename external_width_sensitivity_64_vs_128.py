#!/usr/bin/env python3
"""Exploratory, leakage-free capacity sensitivity: 64 vs 128 Transformer width.

The two variants use the same external CNC-milling data, 13 leave-one-tool-out
folds, five prespecified seeds, masks, losses, optimizer, and source-only
inner selection. Only d_model changes (and its necessary 4*d_model FFN).
This tests width sensitivity, NOT every possible 120-feature adaptation.
This is an additional exploratory study, not a reproduction of earlier tables.

DGX usage:
  python external_width_sensitivity_64_vs_128.py --audit-only
  CUDA_VISIBLE_DEVICES=0 python external_width_sensitivity_64_vs_128.py --smoke
  CUDA_VISIBLE_DEVICES=0 python external_width_sensitivity_64_vs_128.py
Optional: --csv PATH --output-root PATH
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

WINDOW = 15
MASK = .30
LAMBDA = .35
TOL = .015
BATCH = 64
SEEDS = (13, 42, 77, 101, 202)
WIDTHS = (64, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def write_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def canonical(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def numeric(series):
    a = pd.to_numeric(series, errors="coerce")
    if a.notna().mean() < .99:
        a = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return a.astype(float)


def load_and_audit(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = ["FileName", "NumberOfCycle", "SampleIndex", "TollIndex",
                "MillingToolType", "ADOC", "RDOC", "HardnessMean",
                "ToolHolderLength", "CycleToFailure", "CycleToFailureNormalized"]
    raw = None
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            d = pd.read_csv(path, sep=";", decimal=",", header=0,
                            dtype=str, encoding=encoding, keep_default_na=False,
                            na_filter=False, engine="python", on_bad_lines="error")
        except (UnicodeError, pd.errors.ParserError):
            continue
        if len(d.columns) != 131:
            continue
        if list(d.columns) == [f"Column{i}" for i in range(1, 132)] and len(d) == 969:
            header = [str(x).strip() for x in d.iloc[0].tolist()]
            if [canonical(x) for x in header[-11:]] != [canonical(x) for x in metadata]:
                continue
            d = d.iloc[1:].reset_index(drop=True).copy()
            d.columns = header
        elif len(d) == 968 and [canonical(x) for x in d.columns[-11:]] == [canonical(x) for x in metadata]:
            pass
        else:
            continue
        raw = d
        break
    if raw is None or raw.shape != (968, 131) or raw.columns.duplicated().any():
        raise ValueError("Expected verified 968 x 131 external CNC CSV with descriptive header")
    columns = {canonical(x): x for x in raw.columns}
    required = ["tollindex", "numberofcycle", "cycletofailure", "cycletofailurenormalized"]
    if any(x not in columns for x in required):
        raise ValueError("Missing tool, cycle, raw RUL, or normalized RUL column")
    features = [f"F{i:03d}" for i in range(1, 121)]
    matrix = np.column_stack([numeric(raw[x]).to_numpy() for x in raw.columns[:120]])
    data = pd.DataFrame(matrix, columns=features)
    data["Tool"] = numeric(raw[columns["tollindex"]]).to_numpy()
    data["Cycle"] = numeric(raw[columns["numberofcycle"]]).to_numpy()
    data["RawRUL"] = numeric(raw[columns["cycletofailure"]]).to_numpy()
    data["Target"] = numeric(raw[columns["cycletofailurenormalized"]]).to_numpy()
    data["OriginalRow"] = np.arange(len(data), dtype=int)
    if not np.isfinite(data.to_numpy(float)).all():
        raise ValueError("Missing or non-finite CSV data")
    for field in ("Tool", "Cycle", "RawRUL"):
        if not np.allclose(data[field], np.round(data[field])):
            raise ValueError(f"Non-integer {field}")
        data[field] = data[field].astype(int)
    if data.Tool.nunique() != 14 or (data.Cycle < 1).any() or (data.RawRUL < 0).any():
        raise ValueError("Unexpected tool count, cycle, or raw RUL")
    if data.Target.min() < -1e-8 or data.Target.max() > 1 + 1e-8:
        raise ValueError("Normalized RUL outside 0..1")
    data = data.sort_values(["Tool", "Cycle", "OriginalRow"], kind="mergesort").reset_index(drop=True)
    data["Position"] = data.groupby("Tool", sort=False).cumcount() + 1
    eligible = []
    tool_rows = []
    for tool, group in data.groupby("Tool", sort=True):
        y = group.Target.to_numpy(float)
        r = group.RawRUL.to_numpy(float)
        cycle = group.Cycle.to_numpy(int)
        dup = group[group.duplicated("Cycle", keep=False)]
        for _, ds in dup.groupby("Cycle"):
            if np.ptp(ds.RawRUL.to_numpy(float)) > 1e-8 or np.ptp(ds.Target.to_numpy(float)) > 1e-8:
                raise ValueError(f"Conflicting duplicate cycle in tool {tool}")
        if np.any(np.diff(y) > 1e-8) or np.any(np.diff(r) > 1e-8):
            raise ValueError(f"RUL reset in tool {tool}")
        expected = r / r.max() if r.max() else np.zeros_like(r)
        if np.max(np.abs(y - expected)) > 5e-6:
            raise ValueError(f"RUL normalization mismatch tool {tool}")
        ok = len(group) >= 2 and np.ptp(y) > 0
        tool_rows.append({"Tool": int(tool), "Rows": len(group),
                          "DuplicateCycles": len(cycle)-len(np.unique(cycle)), "Eligible": ok})
        if ok:
            eligible.append(int(tool))
    if len(eligible) != 13 or len(data[data.Tool.isin(eligible)]) != 967:
        raise ValueError("Expected 13 evaluable tools and 967 evaluable rows")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return data, features, eligible, digest, pd.DataFrame(tool_rows)


def split_source_tools(source, held):
    ordered = sorted(source)
    rotated = ordered[held % len(ordered):] + ordered[:held % len(ordered)]
    idx = np.linspace(0, len(rotated)-1, 2, dtype=int)
    valid = sorted(set(rotated[int(i)] for i in idx))
    assert len(valid) == 2
    return [x for x in ordered if x not in valid], valid


def prepare(data, features, fit_tools):
    fit = data[data.Tool.isin(fit_tools)]
    xs = StandardScaler().fit(fit[features].to_numpy(float))
    ys = StandardScaler().fit(fit[["Target"]].to_numpy(float))
    z = data.copy()
    z[features] = xs.transform(data[features].to_numpy(float))
    z["ZTarget"] = ys.transform(data[["Target"]].to_numpy(float)).ravel()
    return z, ys


def make_windows(data, features, tools):
    x, y, pos, tool_ids = [], [], [], []
    subset = data[data.Tool.isin(tools)]
    for tool, group in subset.groupby("Tool", sort=True):
        group = group.sort_values("Position")
        v = group[features].to_numpy(np.float32)
        pad = np.pad(v, ((WINDOW-1, 0), (0, 0)), mode="edge")
        for i, (_, row) in enumerate(group.iterrows()):
            x.append(pad[i:i+WINDOW])
            y.append(row.ZTarget)
            pos.append(row.Position)
            tool_ids.append(tool)
    return (np.asarray(x, np.float32), np.asarray(y, np.float32),
            np.asarray(pos, int), np.asarray(tool_ids, int))


class Backbone(nn.Module):
    def __init__(self, n_features, width):
        super().__init__()
        self.project = nn.Linear(n_features, width)
        self.position = nn.Parameter(torch.zeros(1, WINDOW, width))
        nn.init.normal_(self.position, std=.02)
        block = nn.TransformerEncoderLayer(d_model=width, nhead=4,
                                           dim_feedforward=4*width, dropout=.18,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        self.decoder = nn.Linear(width, n_features)

    def encode(self, x):
        return self.encoder(self.project(x)+self.position[:, :x.shape[1]])

    def forward(self, x):
        return self.decoder(self.encode(x))


class Model(nn.Module):
    def __init__(self, pretrained, width):
        super().__init__()
        self.backbone = copy.deepcopy(pretrained)
        self.head = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Dropout(.20),
                                  nn.Linear(48, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.head(self.backbone.encode(x).mean(1)).squeeze(1)


def batches(x, y=None, seed=0):
    tensors = [torch.from_numpy(x)]
    if y is not None:
        tensors.append(torch.from_numpy(y))
    return DataLoader(TensorDataset(*tensors), batch_size=min(BATCH, len(x)),
                      shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def pretrain(x, n_features, width, seed, epochs):
    seed_everything(seed)
    model = Backbone(n_features, width).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for (xb,) in batches(x, seed=seed):
            xb = xb.to(DEVICE)
            mask = (torch.rand(xb.shape[:2], device=DEVICE) < MASK).unsqueeze(2)
            opt.zero_grad(set_to_none=True)
            recon = model(xb.masked_fill(mask, 0))
            loss = ((recon-xb).square()*mask).sum()/(mask.sum()*xb.shape[2]+1e-12)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
    return model


def physics_step(model, opt, x, pos, tool_ids):
    was_training = model.training
    model.eval()
    opt.zero_grad(set_to_none=True)
    valid = [t for t in np.unique(tool_ids) if np.sum(tool_ids == t) >= 2]
    for t in valid:
        ids = np.flatnonzero(tool_ids == t)
        ids = ids[np.argsort(pos[ids])]
        p = model(torch.from_numpy(x[ids]).to(DEVICE))
        (LAMBDA*torch.relu(p[1:]-p[:-1]-TOL).mean()/len(valid)).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    opt.step()
    model.train(was_training)


def predict(model, x):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            output.append(model(torch.from_numpy(x[start:start+BATCH]).to(DEVICE)).cpu().numpy())
    return np.concatenate(output)


def fit(model, x, y, pos, tool_ids, seed, epochs, x_val=None, y_val=None, patience=15):
    seed_everything(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    best, best_epoch, wait = math.inf, 1, 0
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in batches(x, y, seed+epoch):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        physics_step(model, opt, x, pos, tool_ids)
        if x_val is not None:
            mse = float(np.mean((predict(model, x_val)-y_val)**2))
            if mse < best - 1e-8:
                best, best_epoch, wait = mse, epoch, 0
            else:
                wait += 1
            if wait >= patience:
                break
    return best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path.home()/"toolwear_data/cnc/FeatureAndMetadata_Milling.csv")
    parser.add_argument("--output-root", type=Path, default=Path.home()/"toolwear_results/external_width_sensitivity")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if args.smoke:
        run_id += "_SMOKE_NOT_FOR_INFERENCE"
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root/run_id
    i = 1
    while out.exists():
        i += 1
        out = args.output_root/f"{run_id}_{i:02d}"
    out.mkdir()
    data, features, test_tools, digest, tool_audit = load_and_audit(args.csv)
    write_csv(tool_audit, out/"tool_audit.csv")
    manifest = {"status": "audit passed", "input": str(args.csv), "input_sha256": digest,
                "tools": test_tools, "seed_list": list(SEEDS), "widths": list(WIDTHS),
                "device": str(DEVICE), "smoke": args.smoke, "audit_only": args.audit_only,
                "selection": "2 complete source tools for epoch selection",
                "held_out_tool_never_in_scaling_ssl_training_or_selection": True,
                "epochs_ssl_selection": 15, "epochs_ssl_refit": 20,
                "max_supervised_epochs": 100, "early_stop_patience": 15}
    write_json(manifest, out/"manifest.json")
    print(f"AUDIT PASSED: {len(data)} rows, {len(test_tools)} evaluable tools, SHA256={digest}", flush=True)
    if args.audit_only:
        return
    max_epochs, select_ssl, refit_ssl, patience = (2, 1, 1, 1) if args.smoke else (100, 15, 20, 15)
    cases, predictions, splits = [], [], []
    outer_tools = test_tools[:1] if args.smoke else test_tools
    seeds = SEEDS[:1] if args.smoke else SEEDS
    for fold, held in enumerate(outer_tools, 1):
        source = [t for t in test_tools if t != held]
        train_tools, valid_tools = split_source_tools(source, held)
        splits.append({"Fold":fold,"HeldTool":held,
                       "TrainTools":json.dumps(train_tools),"ValidationTools":json.dumps(valid_tools),
                       "RefitTools":json.dumps(source)})
        write_csv(pd.DataFrame(splits), out/"split_manifest.csv")
        for seed in seeds:
            runseed = seed + fold*10000
            selection, _ = prepare(data, features, train_tools)
            xt, yt, pt, gt = make_windows(selection, features, train_tools)
            xv, yv, _, _ = make_windows(selection, features, valid_tools)
            refit_data, y_scaler = prepare(data, features, source)
            xs, ys, ps, gs = make_windows(refit_data, features, source)
            xh, yh, ph, _ = make_windows(refit_data, features, [held])
            observed = y_scaler.inverse_transform(yh[:, None]).ravel()
            for width in WIDTHS:
                print(f"fold {fold}/{len(outer_tools)} held {held} seed {seed} width {width}", flush=True)
                pre_sel = pretrain(xt, len(features), width, runseed+1, select_ssl)
                modelseed = runseed+20100
                seed_everything(modelseed)
                model = Model(pre_sel, width)
                selected = fit(model, xt, yt, pt, gt, modelseed, max_epochs,
                               xv, yv, patience)
                del model, pre_sel
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pre_refit = pretrain(xs, len(features), width, runseed+1001, refit_ssl)
                seed_everything(runseed+40100)
                model = Model(pre_refit, width)
                fit(model, xs, ys, ps, gs, runseed+40100, selected)
                predicted = y_scaler.inverse_transform(predict(model, xh)[:, None]).ravel()
                del model, pre_refit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rmse = float(np.sqrt(np.mean((predicted-observed)**2)))
                mae = float(np.mean(np.abs(predicted-observed)))
                span = float(np.ptp(observed))
                cases.append({"Fold":fold,"HeldTool":held,"Seed":seed,"Width":width,
                              "N":len(observed),"SelectedEpochs":selected,"RMSE":rmse,
                              "MAE":mae,"NRMSE":rmse/span,"MeanBias":float(np.mean(predicted-observed))})
                for j in range(len(observed)):
                    predictions.append({"HeldTool":held,"Seed":seed,"Width":width,
                                        "Position":int(ph[j]),"Truth":float(observed[j]),
                                        "Prediction":float(predicted[j])})
                write_csv(pd.DataFrame(cases), out/"metrics_progress.csv")
                write_csv(pd.DataFrame(predictions), out/"predictions_progress.csv")
    case_df = pd.DataFrame(cases)
    pred_df = pd.DataFrame(predictions)
    write_csv(case_df, out/"tool_seed_metrics.csv")
    write_csv(pred_df, out/"predictions.csv")
    pooled = []
    for (width, seed), rows in pred_df.groupby(["Width", "Seed"]):
        y, p = rows.Truth.to_numpy(float), rows.Prediction.to_numpy(float)
        rmse = float(np.sqrt(np.mean((y-p)**2)))
        pooled.append({"Width":int(width),"Seed":int(seed),"RMSE":rmse,
                       "NRMSE":rmse/float(np.ptp(y)),"MAE":float(np.mean(np.abs(y-p)))})
    write_csv(pd.DataFrame(pooled), out/"pooled_seed_metrics.csv")
    toolmeans = case_df.groupby(["HeldTool", "Width"], as_index=False)["NRMSE"].mean()
    wide = toolmeans.pivot(index="HeldTool", columns="Width", values="NRMSE")
    delta = wide[128]-wide[64]
    if not args.smoke and (len(delta)!=13 or not np.isfinite(delta).all()):
        raise ValueError("13 physical tool comparisons required")
    raw_p = float(wilcoxon(delta).pvalue) if len(delta)>1 and not np.allclose(delta,0) else float("nan")
    result = {"unit":"physical tool (five seeds averaged within each)",
              "comparison":"width128_minus_width64", "tools":len(delta),
              "mean_delta_NRMSE":float(delta.mean()), "width128_lower_count":int((delta<0).sum()),
              "wilcoxon_two_sided_raw_p":raw_p,
              "holm_p":"not applicable (one prespecified comparison)",
              "exploratory":True,"smoke":args.smoke}
    write_csv(toolmeans, out/"tool_mean_nrmse.csv")
    write_json(result, out/"width_comparison.json")
    script = Path(__file__)
    if script.exists():
        shutil.copy2(script, out/script.name)
    if not args.smoke:
        shutil.make_archive(str(out), "zip", out)
    print("DONE", result, "results:", out, flush=True)


if __name__ == "__main__":
    main()
