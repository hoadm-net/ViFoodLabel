#!/usr/bin/env python3
"""Descriptive statistics for the ViFoodLabel Data Description section.

Reads the per-image metadata (``data/processed/dataset_meta.json``) and, if
present, the frozen split (``data/processed/splits.json``), then reports:

  * dataset totals (images / tokens / entities / relations) and per-split sizes
  * token (sequence-length) distribution
  * entity distribution over the 11 types, with class-imbalance / long-tail
  * HAS_VALUE relation counts and per-image distribution
  * product_category / background_color / material distributions

Prints a human-readable report, writes ``data/processed/statistics.json``, and
saves histograms/bar charts to ``data/processed/figures/`` (skipped if
matplotlib is unavailable). Run once now (current annotated set) and re-run when
annotation reaches the 560-image target for the final numbers.
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ENTITY_TYPES = (
    "PRODUCT_NAME", "INGREDIENT", "ADDITIVE", "NUTRITION_NAME", "NUTRITION_VALUE",
    "MANUFACTURER", "ORIGIN", "NET_WEIGHT", "MFG_DATE", "EXPIRY_DATE", "WARNING",
)
# The dataset targets the information panel; these are the primary entities the
# images are framed around. The rest (product name, manufacturer, origin, net
# weight, dates) are auxiliary — captured when in frame, not the collection focus.
PRIMARY_ENTITIES = {"INGREDIENT", "ADDITIVE", "NUTRITION_NAME", "NUTRITION_VALUE", "WARNING"}
PRIMARY_FIELDS = ("ingredients", "additives", "nutrition_value", "warnings")
SECONDARY_FIELDS = ("product_name", "manufacturer", "origin", "net_weight", "mfg_date", "expiry_date")
LONG_TAIL_PCT = 2.0


def group_of(name, primary):
    return "primary" if name in primary else "secondary"


def summarize(values):
    values = list(values)
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": min(s),
        "max": max(s),
        "mean": round(st.mean(s), 2),
        "median": st.median(s),
        "std": round(st.pstdev(s), 2) if len(s) > 1 else 0.0,
        "p25": st.quantiles(s, n=4)[0] if len(s) > 1 else s[0],
        "p75": st.quantiles(s, n=4)[2] if len(s) > 1 else s[0],
        "p90": st.quantiles(s, n=10)[8] if len(s) > 1 else s[0],
    }


def categorical(meta, field):
    counts = Counter(m.get(field, "unknown") for m in meta.values())
    n = sum(counts.values())
    return {k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in counts.most_common()}


def field_coverage(records):
    n = len(records)
    def present(v):
        return bool(v.strip()) if isinstance(v, str) else bool(v)
    cov = {}
    for f in PRIMARY_FIELDS + SECONDARY_FIELDS:
        c = sum(1 for r in records if present(r.get("kie", {}).get(f)))
        cov[f] = {
            "present": c,
            "pct": round(100 * c / n, 1) if n else 0.0,
            "group": group_of(f, set(PRIMARY_FIELDS)),
        }
    return cov


def compute(meta, splits, records=None):
    token_counts = [m.get("token_count", 0) for m in meta.values()]
    relation_counts = [m.get("num_relations", 0) for m in meta.values()]

    entity_totals = Counter()
    for m in meta.values():
        for t, c in m.get("entity_counts", {}).items():
            entity_totals[t] += c
    total_entities = sum(entity_totals.values())
    entity_dist = {
        t: {
            "count": entity_totals.get(t, 0),
            "pct": round(100 * entity_totals.get(t, 0) / total_entities, 1) if total_entities else 0.0,
            "group": group_of(t, PRIMARY_ENTITIES),
        }
        for t in sorted(ENTITY_TYPES, key=lambda x: -entity_totals.get(x, 0))
    }
    primary_spans = sum(entity_totals.get(t, 0) for t in PRIMARY_ENTITIES)
    entity_groups = {
        "primary": {"spans": primary_spans,
                    "pct": round(100 * primary_spans / total_entities, 1) if total_entities else 0.0},
        "secondary": {"spans": total_entities - primary_spans,
                      "pct": round(100 * (total_entities - primary_spans) / total_entities, 1) if total_entities else 0.0},
    }

    stats = {
        "n_images": len(meta),
        "totals": {
            "tokens": sum(token_counts),
            "entities": total_entities,
            "relations": sum(relation_counts),
        },
        "token_length": summarize(token_counts),
        "entity_distribution": entity_dist,
        "entity_groups": entity_groups,
        "long_tail_entities": [t for t, d in entity_dist.items() if d["pct"] < LONG_TAIL_PCT],
        "relation_per_image": {
            **summarize(relation_counts),
            "total": sum(relation_counts),
            "images_with_zero": sum(1 for c in relation_counts if c == 0),
        },
        "product_category": categorical(meta, "product_category"),
        "background_color": categorical(meta, "background_color"),
        "material": categorical(meta, "material"),
    }

    if splits:
        per_split = {}
        for name in ("train", "dev", "test"):
            ids = splits.get(name, [])
            ms = [meta[i] for i in ids if i in meta]
            per_split[name] = {
                "images": len(ms),
                "tokens": sum(m.get("token_count", 0) for m in ms),
                "relations": sum(m.get("num_relations", 0) for m in ms),
            }
        stats["splits"] = per_split
    if records:
        stats["field_coverage"] = field_coverage(records)
    return stats


def collect_centroids(token_files):
    """Per entity type, the (x, y) centroids of its token boxes in [0, 1000]."""
    cent = defaultdict(lambda: ([], []))
    for f in token_files:
        for img in json.loads(Path(f).read_text()):
            for tok in img.get("tokens", []):
                lab = tok.get("label", "O")
                if lab == "O":
                    continue
                et = lab.split("-", 1)[1] if "-" in lab else lab
                b = tok.get("bbox")
                if not b or len(b) != 4:
                    continue
                cent[et][0].append((b[0] + b[2]) / 2)
                cent[et][1].append((b[1] + b[3]) / 2)
    return cent


PRIMARY_C, AUX_C = "#1f77b4", "#ff7f0e"


def save_figures(meta, stats, centroids, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    legend = [Patch(color=PRIMARY_C, label="primary"), Patch(color=AUX_C, label="auxiliary")]

    # 1. Entity distribution, coloured by primary/auxiliary group
    ed = stats["entity_distribution"]
    colors = [PRIMARY_C if d["group"] == "primary" else AUX_C for d in ed.values()]
    plt.figure(figsize=(9, 4))
    plt.bar(list(ed.keys()), [d["count"] for d in ed.values()], color=colors)
    plt.xticks(rotation=45, ha="right"); plt.ylabel("spans")
    plt.title("Entity distribution"); plt.legend(handles=legend)
    plt.tight_layout(); plt.savefig(outdir / "entity_distribution.png", dpi=150); plt.close()

    # 2. Token (sequence-length) distribution
    plt.figure()
    plt.hist([m.get("token_count", 0) for m in meta.values()], bins=30)
    plt.xlabel("tokens per image"); plt.ylabel("images")
    plt.title("Token (sequence-length) distribution")
    plt.tight_layout(); plt.savefig(outdir / "token_length_hist.png", dpi=150); plt.close()

    # 3. HAS_VALUE relations per image
    plt.figure()
    plt.hist([m.get("num_relations", 0) for m in meta.values()], bins=30)
    plt.xlabel("HAS_VALUE relations per image"); plt.ylabel("images")
    plt.title("Relation-count distribution")
    plt.tight_layout(); plt.savefig(outdir / "relation_per_image_hist.png", dpi=150); plt.close()

    # 4. Field coverage, grouped by primary/auxiliary
    fc = stats.get("field_coverage")
    if fc:
        items = sorted(fc.items(), key=lambda kv: (kv[1]["group"] != "primary", -kv[1]["pct"]))
        colors = [PRIMARY_C if d["group"] == "primary" else AUX_C for _, d in items]
        plt.figure(figsize=(9, 4))
        plt.bar([k for k, _ in items], [d["pct"] for _, d in items], color=colors)
        plt.xticks(rotation=45, ha="right"); plt.ylim(0, 100)
        plt.ylabel("% images present"); plt.title("Field coverage"); plt.legend(handles=legend)
        plt.tight_layout(); plt.savefig(outdir / "field_coverage.png", dpi=150); plt.close()

    # 5. Metadata diversity: category / background / material
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field in zip(axes, ("product_category", "background_color", "material")):
        d = stats[field]
        ax.bar(list(d.keys()), [x["count"] for x in d.values()])
        ax.set_title(field); ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("images")
    plt.tight_layout(); plt.savefig(outdir / "metadata_distributions.png", dpi=150); plt.close()

    # 6. Spatial layout heatmap per entity type (token centroids on the 0-1000 canvas)
    if centroids:
        nrow, ncol = 3, 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 9))
        for i, t in enumerate(ENTITY_TYPES):
            ax = axes.flat[i]
            xs, ys = centroids.get(t, ([], []))
            if xs:
                ax.hist2d(xs, ys, bins=25, range=[[0, 1000], [0, 1000]], cmap="viridis")
            ax.set_title(f"{t} (n={len(xs)})", fontsize=8)
            ax.set_xlim(0, 1000); ax.set_ylim(1000, 0)  # image coords: y grows downward
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(ENTITY_TYPES), nrow * ncol):
            axes.flat[j].axis("off")
        fig.suptitle("Entity spatial layout (token-box centroids)")
        plt.tight_layout(); plt.savefig(outdir / "layout_heatmap.png", dpi=150); plt.close()
    return outdir


def print_report(stats):
    print(f"\n{'='*60}\nViFoodLabel — dataset statistics\n{'='*60}")
    print(f"Images: {stats['n_images']}   Tokens: {stats['totals']['tokens']:,}   "
          f"Entities: {stats['totals']['entities']:,}   Relations: {stats['totals']['relations']:,}")

    if "splits" in stats:
        print("\nSplits:")
        for name, d in stats["splits"].items():
            print(f"  {name:5s}: {d['images']:4d} imgs  {d['tokens']:7,d} tok  {d['relations']:5d} rel")

    tl = stats["token_length"]
    print(f"\nToken length/image: min {tl['min']}  median {tl['median']}  "
          f"mean {tl['mean']}  max {tl['max']}  (p90 {tl['p90']}, std {tl['std']})")

    print("\nEntity distribution (P=primary / a=auxiliary):")
    for t, d in stats["entity_distribution"].items():
        tag = "P" if d["group"] == "primary" else "a"
        tail = "  <- long tail" if t in stats["long_tail_entities"] else ""
        print(f"  [{tag}] {t:16s} {d['count']:6,d}  {d['pct']:5.1f}%{tail}")
    eg = stats["entity_groups"]
    print(f"  -> primary {eg['primary']['spans']:,} ({eg['primary']['pct']}%)  "
          f"auxiliary {eg['secondary']['spans']:,} ({eg['secondary']['pct']}%)")

    if "field_coverage" in stats:
        print("\nField coverage (images with the field present):")
        for f, d in stats["field_coverage"].items():
            tag = "P" if d["group"] == "primary" else "a"
            print(f"  [{tag}] {f:16s} {d['present']:4d}  {d['pct']:5.1f}%")

    r = stats["relation_per_image"]
    print(f"\nHAS_VALUE: total {r['total']}  per-image median {r['median']}  "
          f"mean {r['mean']}  max {r['max']}  (images with 0: {r['images_with_zero']})")

    for field in ("product_category", "background_color", "material"):
        print(f"\n{field}:")
        for k, d in stats[field].items():
            print(f"  {k:14s} {d['count']:4d}  {d['pct']:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/processed/dataset_meta.json")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--dataset", default="data/processed/dataset")
    ap.add_argument("--tokens", default="data/processed/train.json,data/processed/val.json")
    ap.add_argument("--output", default="data/processed/statistics.json")
    ap.add_argument("--figures", default="data/processed/figures")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    splits_path = Path(args.splits)
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else None
    dataset_dir = Path(args.dataset)
    records = [json.loads(p.read_text()) for p in sorted(dataset_dir.glob("*.json"))] \
        if dataset_dir.exists() else None
    token_files = [p for p in args.tokens.split(",") if p and Path(p).exists()]
    centroids = collect_centroids(token_files) if token_files else None

    stats = compute(meta, splits, records)
    print_report(stats)

    Path(args.output).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n-> {args.output}")
    figdir = save_figures(meta, stats, centroids, Path(args.figures))
    print(f"-> figures: {figdir}" if figdir else "-> figures skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
