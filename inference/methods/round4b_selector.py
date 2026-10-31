"""Complete-space Nyström and OOF-trained answer-group CPU selectors."""

import json
from pathlib import Path
import numpy as np

from round2_features import groups, raw_dense
from round2_selector import relative_probabilities
from round4_selector import candidate_scores as previous_scores, linear_inputs, question_standardize
from selector import select_choices

GROUP_FEATURE_NAMES = (
    "vote", "max_score", "mean_score", "min_score", "std_score",
    "max_minus_global_max", "max_minus_competitor", "mean_minus_competitor", "within_group_top_gap",
    "probability_mass", "mean_sigmoid", "question_entropy", "competitor_votes", "vote_advantage",
    "answer_count", "higher_group_fraction", "question_std", "max_zscore", "same_cos_mean", "same_cos_max",
    "group_unit_spread", "largest_votes", "is_majority", "max_minus_global_min", "different_cos_mean",
    "different_cos_max", "group_norm_ratio")


def group_features(bank, embeddings, scores):
    membership, valid, vote = groups(bank)
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-6)
    cosine = np.clip(np.einsum("nid,njd->nij", unit, unit), -1, 1)
    probabilities = relative_probabilities(scores, valid)
    rows, mapping, counts = [], [], []
    for q in range(len(bank)):
        active = np.flatnonzero(membership[q].sum(axis=1))
        counts.append(len(active))
        ix_all = np.flatnonzero(valid[q])
        if not len(ix_all): continue
        maxima = {int(g): float(scores[q, membership[q, g].astype(bool)].max()) for g in active}
        sizes = {int(g): int(membership[q, g].sum()) for g in active}
        full = scores[q, ix_all]; std = max(float(full.std()), 1e-6)
        entropy = -float(np.sum(probabilities[q] * np.log(np.maximum(probabilities[q], 1e-12))))
        for g in active:
            inside = np.flatnonzero(membership[q, g]); outside = np.asarray([i for i in ix_all if i not in inside], dtype=int)
            values = scores[q, inside]; maximum = float(values.max())
            others = [int(h) for h in active if h != g]
            comp_max = max(maxima[h] for h in others) if others else maximum
            comp_count = max(sizes[h] for h in others if maxima[h] == comp_max) if others else 0
            same_cos = [cosine[q, i, j] for i in inside for j in inside if i != j]
            diff_cos = [cosine[q, i, j] for i in inside for j in outside]
            group_unit = unit[q, inside]; mean_unit = group_unit.mean(axis=0)
            top_gap = maximum - float(np.sort(values)[-2]) if len(values) > 1 else 0.
            row = [len(inside)/4, maximum, float(values.mean()), float(values.min()), float(values.std()),
                maximum-float(full.max()), maximum-comp_max,
                float(values.mean())-(float(scores[q, outside].mean()) if len(outside) else float(values.mean())),
                top_gap, float(probabilities[q, inside].sum()), float(np.mean(1/(1+np.exp(-np.clip(values,-30,30))))),
                entropy, comp_count/4, (len(inside)-comp_count)/4, len(active)/4,
                sum(v > maximum for v in maxima.values())/4, std, (maximum-float(full.mean()))/std,
                float(np.mean(same_cos)) if same_cos else 0., float(max(same_cos)) if same_cos else 0.,
                float(np.mean(np.sum((group_unit-mean_unit)**2,axis=1))), max(sizes.values())/4,
                float(len(inside)==max(sizes.values())), maximum-float(full.min()),
                float(np.mean(diff_cos)) if diff_cos else 0., float(max(diff_cos)) if diff_cos else 0.,
                float(np.linalg.norm(embeddings[q, inside].mean(axis=0))/max(np.linalg.norm(embeddings[q, ix_all].mean(axis=0)),1e-6))]
            rows.append(row); mapping.append((q, int(g)))
    matrix = np.asarray(rows, np.float32).reshape(-1, len(GROUP_FEATURE_NAMES))
    if not np.isfinite(matrix).all(): raise ValueError("Nonfinite answer-group feature")
    return matrix, mapping, np.asarray(counts, dtype=np.uint32), membership


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve(); config = json.loads((root/"config.json").read_text())
    if config.get("format") == "mp2-round4-v1": return previous_scores(bank, arrays, root)
    if config.get("format") != "mp2-round4b-v1": raise ValueError("Unexpected continuation checkpoint")
    if not bank: return np.empty((0,4),np.float32)
    if config["family"] in ("average","blend"):
        values=[]
        for name in config["members"]:
            child=(root/name).resolve()
            if not child.is_relative_to(root): raise ValueError("Member outside checkpoint")
            if config["family"]=="blend" and name=="baseline":
                from context_selector import candidate_scores as base_scores
                s,_=base_scores(bank,arrays,child)
            else:s=candidate_scores(bank,arrays,child)
            values.append(s)
        if config["family"]=="blend":
            _,valid,_=groups(bank);values=[question_standardize(s,valid) for s in values]
        return np.mean(np.stack(values),axis=0).astype(np.float32)
    raw=raw_dense(bank,arrays["embeddings"],{"dense":"v1_count","embedding":True})
    with np.load(root/"weights.npz",allow_pickle=False) as w:
        x=((raw-w["mean"])/w["scale"]).astype(np.float32)
        if config["family"]=="nystrom":
            xx=linear_inputs(x).reshape(-1,x.shape[-1]).astype(np.float64)
            z=(x[...,:1024]/np.float32(32)).reshape(-1,1024).astype(np.float64)
            support=w["support"];norm=np.sum(support*support,axis=1)
            result=xx@w["linear_w"]
            for start in range(0,len(z),256):
                chunk=z[start:start+256]
                d=np.maximum(np.sum(chunk*chunk,axis=1,keepdims=True)+norm[None]-2*chunk@support.T,0)
                result[start:start+len(chunk)]+=np.exp(-float(w["gamma"])*d)@w["alpha"]
            result=result.reshape(-1,4)
        elif config["family"]=="group_meta":
            import xgboost as xgb
            base=(x@w["base_w"]+w["base_b"]).squeeze(-1)
            features,mapping,_,membership=group_features(bank,arrays["embeddings"],base)
            result=np.full((len(bank),4),-1e6,np.float32)
            if len(features):
                model=xgb.Booster();model.load_model(root/"model.ubj");model.set_param({"device":"cpu","nthread":1})
                scores=model.predict(xgb.DMatrix(features,nthread=1),output_margin=True)
                for (q,g),score in zip(mapping,scores):result[q,membership[q,g].astype(bool)]=score
        else:raise ValueError("Unexpected continuation family")
    if result.shape!=(len(bank),4) or not np.isfinite(result).all():raise ValueError("Malformed continuation scores")
    return result.astype(np.float32)


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),"max") if bank else []
