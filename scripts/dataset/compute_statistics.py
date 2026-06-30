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
LONG_TAIL_PCT = 2.0


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


def compute(meta, splits):
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
        }
        for t in sorted(ENTITY_TYPES, key=lambda x: -entity_totals.get(x, 0))
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
    return stats


def save_figures(meta, stats, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist([m.get("token_count", 0) for m in meta.values()], bins=30)
    plt.xlabel("tokens per image"); plt.ylabel("images")
    plt.title("Token (sequence-length) distribution")
    plt.tight_layout(); plt.savefig(outdir / "token_length_hist.png", dpi=150); plt.close()

    ed = stats["entity_distribution"]
    plt.figure(figsize=(8, 4))
    plt.bar(list(ed.keys()), [d["count"] for d in ed.values()])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("spans")
    plt.title("Entity distribution")
    plt.tight_layout(); plt.savefig(outdir / "entity_distribution.png", dpi=150); plt.close()

    plt.figure()
    plt.hist([m.get("num_relations", 0) for m in meta.values()], bins=30)
    plt.xlabel("HAS_VALUE relations per image"); plt.ylabel("images")
    plt.title("Relation-count distribution")
    plt.tight_layout(); plt.savefig(outdir / "relation_per_image_hist.png", dpi=150); plt.close()

    pc = stats["product_category"]
    plt.figure(figsize=(7, 4))
    plt.bar(list(pc.keys()), [d["count"] for d in pc.values()])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("images")
    plt.title("Product-category distribution")
    plt.tight_layout(); plt.savefig(outdir / "product_category.png", dpi=150); plt.close()
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

    print("\nEntity distribution:")
    for t, d in stats["entity_distribution"].items():
        tail = "  <- long tail" if t in stats["long_tail_entities"] else ""
        print(f"  {t:16s} {d['count']:6,d}  {d['pct']:5.1f}%{tail}")

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
    ap.add_argument("--output", default="data/processed/statistics.json")
    ap.add_argument("--figures", default="data/processed/figures")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    splits_path = Path(args.splits)
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else None

    stats = compute(meta, splits)
    print_report(stats)

    Path(args.output).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n-> {args.output}")
    figdir = save_figures(meta, stats, Path(args.figures))
    print(f"-> figures: {figdir}" if figdir else "-> figures skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
