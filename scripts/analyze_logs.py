#!/usr/bin/env python3
"""
SwarmInfer: Parse experiment logs and generate LaTeX tables for IEEE Access paper.

Usage:
    python analyze_logs.py logs/lite_n1.log logs/fatcnn_n2.log logs/fatcnn_n4.log

Each log file is a captured ESP32 monitor output containing CSV lines like:
    CSV,lite_n1,0,3,3,1,1890123,456789,678901,234567,1234,567
    CSV,fatcnn_n4,0,3,3,1,2267000,2345,100234,3456,200456,4567,300678

Outputs:
    - results/summary_table.tex   (main comparison table)
    - results/per_layer_table.tex (per-layer breakdown)
    - results/accuracy_table.tex  (per-class accuracy)
    - results/stats.json          (raw statistics for further use)
"""

import sys
import os
import json
import re
import numpy as np
from collections import defaultdict


def parse_log(filepath):
    """Parse a monitor log file and extract CSV data lines."""
    rows = []
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            # Match CSV lines: may be preceded by ESP log prefix like "I (12345) COORD: "
            m = re.search(r'CSV,(\w+),(\d+),(\d+),(\d+),([01]),(\d+)', line)
            if not m:
                continue
            config = m.group(1)
            fields = line[m.start():].strip().split(',')
            row = {'config': config, 'img': int(fields[2]), 'label': int(fields[3]),
                   'pred': int(fields[4]), 'match': int(fields[5]),
                   'total_us': int(fields[6])}
            # Per-layer timing (different columns for single vs distributed)
            if config.startswith('lite'):
                # lite_n1: total_us, conv1_us, conv2_us, conv3_us, dense1_us, dense2_us
                if len(fields) >= 12:
                    row['conv1_us'] = int(fields[7])
                    row['conv2_us'] = int(fields[8])
                    row['conv3_us'] = int(fields[9])
                    row['dense1_us'] = int(fields[10])
                    row['dense2_us'] = int(fields[11])
            else:
                # fatcnn_nX: total_us, l1_bcast, l1_gather, l2_bcast, l2_gather, l3_bcast, l3_gather
                if len(fields) >= 13:
                    row['l1_bcast_us'] = int(fields[7])
                    row['l1_gather_us'] = int(fields[8])
                    row['l2_bcast_us'] = int(fields[9])
                    row['l2_gather_us'] = int(fields[10])
                    row['l3_bcast_us'] = int(fields[11])
                    row['l3_gather_us'] = int(fields[12])
            rows.append(row)
    return rows


def compute_stats(values):
    """Compute mean, std, min, max, p50, p95, p99 for a list of values."""
    a = np.array(values, dtype=np.float64)
    return {
        'mean': float(np.mean(a)),
        'std': float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'p50': float(np.percentile(a, 50)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
        'n': len(a),
    }


CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

CONFIG_LABELS = {
    'lite_n1': 'FatCNN-Lite (N=1)',
    'fatcnn_n2': 'FatCNN (N=2)',
    'fatcnn_n4': 'FatCNN (N=4)',
}


def analyze(all_rows):
    """Analyze parsed rows grouped by config."""
    configs = defaultdict(list)
    for r in all_rows:
        configs[r['config']].append(r)

    results = {}
    for cfg, rows in configs.items():
        latencies = [r['total_us'] for r in rows]
        matches = [r['match'] for r in rows]
        n_images = len(rows)
        n_correct = sum(matches)

        res = {
            'label': CONFIG_LABELS.get(cfg, cfg),
            'n_images': n_images,
            'accuracy': n_correct / n_images if n_images > 0 else 0,
            'n_correct': n_correct,
            'latency': compute_stats(latencies),
        }

        # Per-layer stats (single node)
        if cfg.startswith('lite') and 'conv1_us' in rows[0]:
            for layer in ['conv1_us', 'conv2_us', 'conv3_us', 'dense1_us', 'dense2_us']:
                vals = [r[layer] for r in rows if layer in r]
                if vals:
                    res[layer] = compute_stats(vals)

        # Per-layer stats (distributed)
        if cfg.startswith('fatcnn') and 'l1_bcast_us' in rows[0]:
            for phase in ['l1_bcast_us', 'l1_gather_us', 'l2_bcast_us', 'l2_gather_us',
                          'l3_bcast_us', 'l3_gather_us']:
                vals = [r[phase] for r in rows if phase in r]
                if vals:
                    res[phase] = compute_stats(vals)

        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for r in rows:
            class_total[r['label']] += 1
            if r['match']:
                class_correct[r['label']] += 1
        res['per_class'] = {}
        for c in range(10):
            total = class_total.get(c, 0)
            correct = class_correct.get(c, 0)
            res['per_class'][CIFAR_CLASSES[c]] = {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0,
            }

        results[cfg] = res
    return results


def gen_summary_table(results):
    """Generate main comparison LaTeX table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Inference performance comparison across configurations.}')
    lines.append(r'\label{tab:summary}')
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(r'Configuration & Images & Accuracy (\%) & Mean (ms) & Std (ms) & P95 (ms) & P99 (ms) \\')
    lines.append(r'\midrule')

    for cfg in ['lite_n1', 'fatcnn_n2', 'fatcnn_n4']:
        if cfg not in results:
            continue
        r = results[cfg]
        lat = r['latency']
        lines.append(
            f"{r['label']} & {r['n_images']} & {r['accuracy']*100:.1f} "
            f"& {lat['mean']/1000:.1f} & {lat['std']/1000:.1f} "
            f"& {lat['p95']/1000:.1f} & {lat['p99']/1000:.1f} \\\\"
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_per_layer_table(results):
    """Generate per-layer timing breakdown table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Per-layer latency breakdown (mean $\pm$ std, in ms).}')
    lines.append(r'\label{tab:perlayer}')

    # Single-node section
    if 'lite_n1' in results:
        r = results['lite_n1']
        lines.append(r'\begin{tabular}{lcc}')
        lines.append(r'\toprule')
        lines.append(r'\multicolumn{3}{c}{\textbf{FatCNN-Lite (single node)}} \\')
        lines.append(r'\midrule')
        lines.append(r'Layer & Mean (ms) & Std (ms) \\')
        lines.append(r'\midrule')
        layer_names = {'conv1_us': 'Conv1 (5x5, 3$\\to$32)',
                       'conv2_us': 'Conv2 (3x3, 32$\\to$64)',
                       'conv3_us': 'Conv3 (3x3, 64$\\to$128)',
                       'dense1_us': 'Dense1 (128$\\to$64)',
                       'dense2_us': 'Dense2 (64$\\to$10)'}
        for key, name in layer_names.items():
            if key in r:
                s = r[key]
                lines.append(f"{name} & {s['mean']/1000:.1f} & {s['std']/1000:.1f} \\\\")
        lines.append(r'\midrule')
        lat = r['latency']
        lines.append(f"Total & {lat['mean']/1000:.1f} & {lat['std']/1000:.1f} \\\\")
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append('')

    # Distributed section
    dist_cfgs = [c for c in ['fatcnn_n2', 'fatcnn_n4'] if c in results]
    if dist_cfgs:
        lines.append(r'\vspace{1em}')
        ncols = 1 + 2 * len(dist_cfgs)
        col_spec = 'l' + 'cc' * len(dist_cfgs)
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append(r'\toprule')
        header = 'Phase'
        for cfg in dist_cfgs:
            label = results[cfg]['label']
            header += f' & \\multicolumn{{2}}{{c}}{{{label}}}'
        header += r' \\'
        lines.append(header)
        subheader = ''
        for _ in dist_cfgs:
            subheader += ' & Mean (ms) & Std (ms)'
        lines.append(f'{subheader} \\\\')
        lines.append(r'\midrule')

        phase_names = {
            'l1_bcast_us': 'L1 Broadcast',
            'l1_gather_us': 'L1 Gather',
            'l2_bcast_us': 'L2 Broadcast',
            'l2_gather_us': 'L2 Gather',
            'l3_bcast_us': 'L3 Broadcast',
            'l3_gather_us': 'L3 Gather',
        }
        for key, name in phase_names.items():
            row = name
            for cfg in dist_cfgs:
                r = results[cfg]
                if key in r:
                    s = r[key]
                    row += f" & {s['mean']/1000:.1f} & {s['std']/1000:.1f}"
                else:
                    row += ' & -- & --'
            row += r' \\'
            lines.append(row)

        lines.append(r'\midrule')
        row = 'Total'
        for cfg in dist_cfgs:
            lat = results[cfg]['latency']
            row += f" & {lat['mean']/1000:.1f} & {lat['std']/1000:.1f}"
        row += r' \\'
        lines.append(row)
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_scalability_table(results):
    """Generate scalability / speedup table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Scalability analysis: speedup relative to N=2.}')
    lines.append(r'\label{tab:scalability}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Configuration & Mean latency (ms) & Speedup vs N=2 & Accuracy (\%) \\')
    lines.append(r'\midrule')

    base_lat = None
    for cfg in ['fatcnn_n2', 'fatcnn_n4']:
        if cfg not in results:
            continue
        r = results[cfg]
        lat = r['latency']['mean']
        if base_lat is None:
            base_lat = lat
        speedup = base_lat / lat if lat > 0 else 0
        lines.append(
            f"{r['label']} & {lat/1000:.1f} & {speedup:.2f}$\\times$ & {r['accuracy']*100:.1f} \\\\"
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_accuracy_table(results):
    """Generate per-class accuracy table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Per-class accuracy breakdown (\%).}')
    lines.append(r'\label{tab:perclass}')

    cfgs = [c for c in ['lite_n1', 'fatcnn_n2', 'fatcnn_n4'] if c in results]
    col_spec = 'l' + 'c' * len(cfgs)
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')

    header = 'Class'
    for cfg in cfgs:
        header += f" & {results[cfg]['label']}"
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    for cls in CIFAR_CLASSES:
        row = cls.capitalize()
        for cfg in cfgs:
            pc = results[cfg]['per_class'].get(cls, {})
            acc = pc.get('accuracy', 0) * 100
            row += f' & {acc:.0f}'
        row += r' \\'
        lines.append(row)

    lines.append(r'\midrule')
    row = r'\textbf{Overall}'
    for cfg in cfgs:
        row += f" & \\textbf{{{results[cfg]['accuracy']*100:.1f}}}"
    row += r' \\'
    lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log1> [log2] [log3] ...")
        print("  Each log is a captured ESP32 monitor output with CSV lines.")
        sys.exit(1)

    # Parse all logs
    all_rows = []
    for logpath in sys.argv[1:]:
        print(f"Parsing {logpath}...")
        rows = parse_log(logpath)
        print(f"  Found {len(rows)} CSV data rows")
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No CSV data found in any log file.")
        sys.exit(1)

    # Analyze
    results = analyze(all_rows)

    # Print summary to console
    print("\n" + "=" * 60)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    for cfg in ['lite_n1', 'fatcnn_n2', 'fatcnn_n4']:
        if cfg not in results:
            continue
        r = results[cfg]
        lat = r['latency']
        print(f"\n{r['label']}:")
        print(f"  Images:   {r['n_images']}")
        print(f"  Accuracy: {r['n_correct']}/{r['n_images']} = {r['accuracy']*100:.1f}%")
        print(f"  Latency:  {lat['mean']/1000:.1f} +/- {lat['std']/1000:.1f} ms "
              f"(min={lat['min']/1000:.1f}, max={lat['max']/1000:.1f}, "
              f"p95={lat['p95']/1000:.1f}, p99={lat['p99']/1000:.1f})")

    # Output directory
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)

    # Generate LaTeX tables
    tables = {
        'summary_table.tex': gen_summary_table(results),
        'per_layer_table.tex': gen_per_layer_table(results),
        'scalability_table.tex': gen_scalability_table(results),
        'accuracy_table.tex': gen_accuracy_table(results),
    }

    for fname, content in tables.items():
        path = os.path.join(outdir, fname)
        with open(path, 'w') as f:
            f.write(content + '\n')
        print(f"\nWrote {path}")

    # Save raw stats as JSON
    json_path = os.path.join(outdir, 'stats.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    print(f"\nDone! LaTeX tables in {outdir}/")


if __name__ == '__main__':
    main()
