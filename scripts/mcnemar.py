#!/usr/bin/env python3
"""
McNemar test for paired accuracy comparison between configurations.

Parses the same CSV log format as analyze_logs.py, aligns predictions by image
index, and reports:
  - contingency table (n01, n10, n00, n11)
  - exact binomial two-sided p-value
  - continuity-corrected chi-square statistic and p-value

Usage:
    python mcnemar.py lite_n1.log fatcnn_n4.log [--json out.json]
"""

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path


def parse_log(path):
    """Return dict {img_idx: match (0/1)} and detected config name."""
    preds = {}
    config = None
    pattern = re.compile(r'CSV,(\w+),(\d+),(\d+),(\d+),([01]),(\d+)')
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            if config is None:
                config = m.group(1)
            img = int(m.group(2))
            match = int(m.group(5))
            preds[img] = match
    if config is None:
        raise ValueError(f"No CSV rows found in {path}")
    return config, preds


def exact_binomial_two_sided(n01, n10):
    """Two-sided exact binomial p-value for McNemar (b, c) under H0: p=0.5."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 0.0
    for i in range(k + 1):
        p += comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def chi2_cc(n01, n10):
    """Continuity-corrected McNemar chi-square statistic and p-value (df=1)."""
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    # chi-square survival for df=1: P(X>=stat) = erfc(sqrt(stat/2))
    from math import erfc, sqrt
    pval = erfc(sqrt(stat / 2.0))
    return stat, pval


def mcnemar(a_preds, b_preds):
    common = sorted(set(a_preds) & set(b_preds))
    n00 = n01 = n10 = n11 = 0
    for i in common:
        a, b = a_preds[i], b_preds[i]
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    p_exact = exact_binomial_two_sided(n01, n10)
    chi2, p_chi2 = chi2_cc(n01, n10)
    return {
        "n_paired": len(common),
        "n00_both_wrong": n00,
        "n01_a_wrong_b_right": n01,
        "n10_a_right_b_wrong": n10,
        "n11_both_right": n11,
        "acc_a": (n10 + n11) / len(common) if common else 0.0,
        "acc_b": (n01 + n11) / len(common) if common else 0.0,
        "chi2_cc": chi2,
        "p_chi2_cc": p_chi2,
        "p_exact_binomial": p_exact,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg_a, preds_a = parse_log(args.log_a)
    cfg_b, preds_b = parse_log(args.log_b)
    result = {"config_a": cfg_a, "config_b": cfg_b, **mcnemar(preds_a, preds_b)}

    print(f"A = {cfg_a} ({Path(args.log_a).name})")
    print(f"B = {cfg_b} ({Path(args.log_b).name})")
    print(f"n_paired:  {result['n_paired']}")
    print(f"acc_A:     {result['acc_a']:.4f}")
    print(f"acc_B:     {result['acc_b']:.4f}")
    print(f"contingency (rows=A correct, cols=B correct):")
    print(f"              B=0       B=1")
    print(f"  A=0    {result['n00_both_wrong']:6d}    {result['n01_a_wrong_b_right']:6d}")
    print(f"  A=1    {result['n10_a_right_b_wrong']:6d}    {result['n11_both_right']:6d}")
    print(f"chi2_cc:   {result['chi2_cc']:.4f}  p = {result['p_chi2_cc']:.4e}")
    print(f"exact bin: p = {result['p_exact_binomial']:.4e}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
