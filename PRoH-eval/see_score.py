#!/usr/bin/env python3
# see_score.py
import os
import argparse
import json
from datetime import datetime
from statistics import mean
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='hypertension')
parser.add_argument('--part', default='')
parser.add_argument('--ts', default='', help='Run timestamp decided by input')
args = parser.parse_args()

data_source = args.data_source
part = args.part.strip()
part_tag = part if part else "orig"

RUN_ROOT = f"results/{data_source}" if part == "" else f"results/{data_source}_{part}"
RUN_TS = args.ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join(RUN_ROOT, RUN_TS)

RESULT_FILE = os.path.join(RUN_DIR, "test_result.json")
SCORE_JSON = os.path.join(RUN_DIR, "score_summary.json")

with open(RESULT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise ValueError("test_result.json must be a list of records")

def get_val(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def to_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default

def to_float(v):
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except Exception:
        return None

def agg_mean(vals):
    return (mean(vals) if vals else None, len(vals))

METRICS = ("em", "f1", "rsim", "gen")
TREE_FIELDS = ("peak_tree_width", "peak_tree_depth",
               "total_visited_states", "total_seen_states", "max_seen_dag_level")

# Tree stats (overall)
tree_max = {k: 0 for k in TREE_FIELDS}
tree_vals = {k: [] for k in TREE_FIELDS}
# Tree stats by nhop
tree_by_nhop_vals = defaultdict(lambda: {k: [] for k in TREE_FIELDS})

# Score buckets
overall = {m: [] for m in METRICS}
by_nhop = defaultdict(lambda: {m: [] for m in METRICS})  # nhop -> metric -> list
by_nary = defaultdict(lambda: {m: [] for m in METRICS})  # exact integer nary -> metric -> list
by_nary_unknown = {m: [] for m in METRICS}               # for missing nary

# token_usage accumulators (overall)
token_sum = defaultdict(int)
token_present_records = 0
token_keys = set()
token_total_per_record = []

# token_usage accumulators by nhop
token_by_nhop_sum = defaultdict(lambda: defaultdict(int))   # nh -> key -> sum
token_by_nhop_present = defaultdict(int)                    # nh -> count of records having any token_usage
token_by_nhop_keys = defaultdict(set)                       # nh -> set(keys)
token_by_nhop_total_per_rec = defaultdict(list)             # nh -> list of totals per record

# graph_search_depth_stats (overall only)
depth_sum = defaultdict(int)  # depth(str) -> count sum
depth_present_records = 0
depth_keys_set = set()

# Main pass over records
for rec in data:
    # nhop and nary (may be absent)
    nary = get_val(rec, "n_ary", "nary")
    nhop = get_val(rec, "n_hop", "nhop")
    nary_i = to_int(nary)  # None if missing
    nhop_i = to_int(nhop)  # None if missing

    # Tree stats (overall + by_nhop if nhop known)
    tr = rec.get("tree_record", {}) or {}
    for k in TREE_FIELDS:
        v = to_int(tr.get(k, 0), default=0)
        tree_max[k] = max(tree_max[k], v)
        tree_vals[k].append(v)
        if nhop_i is not None:
            tree_by_nhop_vals[nhop_i][k].append(v)

    # token_usage (overall + by_nhop if nhop known)
    tu = rec.get("token_usage") or {}
    if isinstance(tu, dict) and tu:
        token_present_records += 1
        per_rec_total = 0
        for k, v in tu.items():
            try:
                iv = int(v)
            except Exception:
                continue
            token_sum[k] += iv
            token_keys.add(k)
            per_rec_total += iv
            if nhop_i is not None:
                token_by_nhop_sum[nhop_i][k] += iv
                token_by_nhop_keys[nhop_i].add(k)
        token_total_per_record.append(per_rec_total)
        if nhop_i is not None:
            token_by_nhop_present[nhop_i] += 1
            token_by_nhop_total_per_rec[nhop_i].append(per_rec_total)

    # graph_search_depth_stats (overall only)
    gsd = rec.get("graph_search_depth_stats") or {}
    if isinstance(gsd, dict) and gsd:
        depth_present_records += 1
        for k, v in gsd.items():
            try:
                iv = int(v)
            except Exception:
                continue
            depth_sum[str(k)] += iv
            depth_keys_set.add(str(k))

    # Scores aggregation:
    # - Always try to contribute to overall if metric exists.
    # - Contribute to by_nhop if nhop is known.
    # - Contribute to by_nary exact bin if nary is known; otherwise to unknown bin.
    for m in METRICS:
        v = to_float(rec.get(m))
        if v is None:
            continue
        overall[m].append(v)
        if nhop_i is not None:
            by_nhop[nhop_i][m].append(v)
        if nary_i is not None:
            by_nary[nary_i][m].append(v)
        else:
            by_nary_unknown[m].append(v)

# Build JSON report skeleton
report = {
    "meta": {
        "data_source": data_source,
        "run_ts": RUN_TS,
        "run_dir": RUN_DIR,
        "num_records": len(data)
    },
    "tree_stats": {
        "overall": {},
        "by_nhop": {}
    },
    "scores": {
        "overall": {},
        "by_nhop": {},
        "by_nary": {},
        "by_nary_unknown": {}
    },
    "token_usage": {
        "overall": {},
        "by_nhop": {}
    },
    "graph_search_depth_stats": {}
}

# Tree stats: overall
tree_avg = {k: (mean(tree_vals[k]) if tree_vals[k] else 0.0) for k in TREE_FIELDS}
report["tree_stats"]["overall"] = {k: {"max": tree_max[k], "avg": tree_avg[k]} for k in TREE_FIELDS}

# Tree stats: by_nhop
for nh, dct in tree_by_nhop_vals.items():
    report["tree_stats"]["by_nhop"][nh] = {k: {"max": (max(dct[k]) if dct[k] else 0),
                                               "avg": (mean(dct[k]) if dct[k] else 0.0)}
                                           for k in TREE_FIELDS}

# Scores: overall
for m in METRICS:
    mu, cnt = agg_mean(overall[m])
    report["scores"]["overall"][m] = {"mean": (mu if mu is not None else None), "count": cnt}

# Console glance (overall)
print(f"Overall counts: {report['scores']['overall'][METRICS[0]]['count']}")
for m in METRICS:
    mu = report["scores"]["overall"][m]["mean"]
    score = mu * 100 if mu is not None else 0.0
    print(f"  {m}: {score:.2f}", end="")
print()

# Scores: by_nhop
for nh in sorted(by_nhop.keys()):
    report["scores"]["by_nhop"][nh] = {}
    for m in METRICS:
        mu, cnt = agg_mean(by_nhop[nh][m])
        report["scores"]["by_nhop"][nh][m] = {"mean": (mu if mu is not None else None), "count": cnt}
    print(f"NHOP={nh} counts: {report['scores']['by_nhop'][nh][METRICS[0]]['count']}")
    for m in METRICS:
        mu = report["scores"]["by_nhop"][nh][m]["mean"]
        score = mu * 100 if mu is not None else 0.0
        print(f"  {m}: {score:.2f}", end="")
    print()

# Scores: by exact integer nary
for ny in sorted(by_nary.keys()):
    report["scores"]["by_nary"][ny] = {}
    for m in METRICS:
        mu, cnt = agg_mean(by_nary[ny][m])
        report["scores"]["by_nary"][ny][m] = {"mean": (mu if mu is not None else None), "count": cnt}

# Scores: nary unknown
for m in METRICS:
    mu, cnt = agg_mean(by_nary_unknown[m])
    report["scores"]["by_nary_unknown"][m] = {"mean": (mu if mu is not None else None), "count": cnt}

# token_usage: overall (sum/mean/max/overall totals)
if token_keys:
    sorted_keys = sorted(token_keys)
    sum_dict = {k: int(token_sum.get(k, 0)) for k in sorted_keys}
    mean_present = {k: (token_sum[k] / token_present_records) for k in sorted_keys} if token_present_records > 0 else {k: None for k in sorted_keys}
    # max by key (across records)
    token_max = {k: 0 for k in sorted_keys}
    for rec in data:
        tu = rec.get("token_usage") or {}
        for k in sorted_keys:
            try:
                iv = int(tu.get(k, 0))
                if iv > token_max[k]:
                    token_max[k] = iv
            except Exception:
                continue
    overall_total_sum = sum(sum_dict.values())
    overall_total_mean_present = (sum(token_total_per_record) / token_present_records) if token_present_records > 0 else None
    overall_total_max = max(token_total_per_record) if token_total_per_record else 0
    report["token_usage"]["overall"] = {
        "present_records": token_present_records,
        "keys": sorted_keys,
        "sum": sum_dict,
        "mean_per_present_record": mean_present,
        "max_per_record": token_max,
        "overall": {
            "total_sum": int(overall_total_sum),
            "total_mean_per_present_record": overall_total_mean_present,
            "total_max_per_record": overall_total_max
        }
    }
else:
    report["token_usage"]["overall"] = {
        "present_records": 0, "keys": [],
        "sum": {}, "mean_per_present_record": {}, "max_per_record": {},
        "overall": {"total_sum": 0, "total_mean_per_present_record": None, "total_max_per_record": 0}
    }

# token_usage: by_nhop (optional, you already accumulate; now summarise)
for nh, sums in token_by_nhop_sum.items():
    keys = sorted(token_by_nhop_keys[nh]) if token_by_nhop_keys[nh] else []
    if keys:
        mean_present = {k: (sums[k] / token_by_nhop_present[nh]) for k in keys} if token_by_nhop_present[nh] > 0 else {k: None for k in keys}
        token_max = {k: 0 for k in keys}
        # compute per-record max by key for this nhop
        for rec in data:
            rec_nh = to_int(get_val(rec, "n_hop", "nhop"))
            if rec_nh != nh:
                continue
            tu = rec.get("token_usage") or {}
            for k in keys:
                try:
                    iv = int(tu.get(k, 0))
                    if iv > token_max[k]:
                        token_max[k] = iv
                except Exception:
                    continue
        total_sum = int(sum(sums.values()))
        total_mean_present = (sum(token_by_nhop_total_per_rec[nh]) / token_by_nhop_present[nh]) if token_by_nhop_present[nh] > 0 else None
        total_max = max(token_by_nhop_total_per_rec[nh]) if token_by_nhop_total_per_rec[nh] else 0
        report["token_usage"]["by_nhop"][nh] = {
            "present_records": int(token_by_nhop_present[nh]),
            "keys": keys,
            "sum": {k: int(sums[k]) for k in keys},
            "mean_per_present_record": mean_present,
            "max_per_record": token_max,
            "overall": {
                "total_sum": total_sum,
                "total_mean_per_present_record": total_mean_present,
                "total_max_per_record": total_max
            }
        }

# graph_search_depth_stats
if depth_keys_set:
    sorted_depth_keys = sorted(depth_keys_set, key=lambda s: int(s))
    depth_sum_dict = {k: int(depth_sum.get(k, 0)) for k in sorted_depth_keys}
    try:
        max_depth_observed = max(int(k) for k in depth_keys_set)
    except ValueError:
        max_depth_observed = None
    total_depth_events = sum(depth_sum_dict.values())
    report["graph_search_depth_stats"] = {
        "present_records": depth_present_records,
        "depth_keys": sorted_depth_keys,
        "sum_histogram": depth_sum_dict,
        "overall": {
            "total_events": int(total_depth_events),
            "max_depth_observed": max_depth_observed
        }
    }
else:
    report["graph_search_depth_stats"] = {
        "present_records": 0,
        "depth_keys": [],
        "sum_histogram": {},
        "overall": {"total_events": 0, "max_depth_observed": None}
    }

# Write JSON only
os.makedirs(RUN_DIR, exist_ok=True)
with open(SCORE_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
