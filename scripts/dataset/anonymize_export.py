#!/usr/bin/env python3
"""Anonymize the Label Studio export for public release.

Removes annotator identity and infrastructure details from
``data/label_studio/data.json``:

  * ``completed_by`` user ids -> ``annotator_1 .. annotator_N`` (deterministic,
    by ascending id); the real id -> alias mapping is written to a private file
    that must NOT be published.
  * image/thumbnail URLs (which embed a server IP) -> relative paths.
  * editorial/PII fields dropped: upload filenames, comments, timestamps,
    ``updated_by`` / ``last_created_by`` user ids, per-annotation lead time.

The annotation content (boxes, transcriptions, BIO labels, relations) is left
untouched. After writing, the output is re-scanned for any leftover IP address or
email-like string and the run fails loudly if one is found.
"""
import argparse
import json
import re
from pathlib import Path

TASK_DROP = {
    "file_upload", "drafts", "predictions", "comment_authors", "comment_count",
    "unresolved_comment_count", "last_comment_updated_at", "updated_by",
    "created_at", "updated_at", "total_annotations", "cancelled_annotations",
    "total_predictions", "project", "inner_id", "allow_skip",
}
ANNOTATION_DROP = {
    "created_at", "updated_at", "draft_created_at", "lead_time", "updated_by",
    "last_created_by", "import_id", "unique_id", "was_cancelled", "ground_truth",
    "result_count", "bulk_created", "last_action", "parent_prediction",
    "parent_annotation", "prediction", "project", "task",
}
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def relativize(url):
    # "http://host/images/0001.jpeg" -> "images/0001.jpeg"
    return re.sub(r"^https?://[^/]+/", "", url) if isinstance(url, str) else url


def build_mapping(tasks):
    ids = sorted({
        a["completed_by"]
        for t in tasks for a in t.get("annotations", [])
        if isinstance(a.get("completed_by"), int)
    })
    return {i: f"annotator_{n}" for n, i in enumerate(ids, 1)}


def anonymize(tasks, mapping):
    out = []
    for t in tasks:
        task = {k: v for k, v in t.items() if k not in TASK_DROP}
        data = dict(task.get("data", {}))
        if "image" in data:
            data["image"] = relativize(data["image"])
        data.pop("thumbnail", None)  # thumbnails are not part of the release
        task["data"] = data
        anns = []
        for a in task.get("annotations", []):
            ann = {k: v for k, v in a.items() if k not in ANNOTATION_DROP}
            cb = a.get("completed_by")
            ann["completed_by"] = mapping.get(cb, "annotator_unknown")
            anns.append(ann)
        task["annotations"] = anns
        out.append(task)
    return out


def collect_strings(obj):
    out = []
    def walk(o):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return out


def metadata_view(tasks):
    """Everything except transcription/label *content* (result[].value), i.e. the
    fields where annotator identity could leak — distinct from the printed text on
    the labels themselves (which legitimately contains manufacturer emails)."""
    view = []
    for t in tasks:
        view.append({k: v for k, v in t.items() if k != "annotations"})
        for a in t.get("annotations", []):
            view.append({k: v for k, v in a.items() if k != "result"})
            for r in a.get("result", []):
                view.append({k: v for k, v in r.items() if k != "value"})
    return view


def scan_leaks(tasks):
    """Returns (blocking_leaks, content_email_count). Blocks on any IP address
    (infrastructure) or email in a metadata field (annotator identity). Emails in
    label transcriptions are dataset content and are reported, not blocked."""
    all_strings = collect_strings(tasks)
    blocking = [("ip", s[:80]) for s in all_strings if IP_RE.search(s)]
    blocking += [("email-in-metadata", s[:80])
                 for s in collect_strings(metadata_view(tasks)) if EMAIL_RE.search(s)]
    content_emails = sum(1 for s in all_strings if EMAIL_RE.search(s))
    return blocking, content_emails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/label_studio/data.json")
    ap.add_argument("--output", default="data/label_studio/data_anon.json")
    ap.add_argument("--mapping", default="data/label_studio/annotator_map.private.json")
    args = ap.parse_args()

    tasks = json.loads(Path(args.input).read_text())
    mapping = build_mapping(tasks)
    anon = anonymize(tasks, mapping)

    blocking, content_emails = scan_leaks(anon)
    if blocking:
        for kind, sample in blocking[:10]:
            print(f"  LEAK ({kind}): {sample}")
        raise SystemExit(f"Aborting: {len(blocking)} identity/infrastructure leak(s) in output")

    Path(args.output).write_text(json.dumps(anon, ensure_ascii=False, indent=1))
    Path(args.mapping).write_text(
        json.dumps({str(k): v for k, v in mapping.items()}, ensure_ascii=False, indent=2))

    print(f"Anonymized {len(anon)} tasks -> {args.output}")
    print(f"Annotators: {len(mapping)}  {dict(sorted((v, k) for k, v in mapping.items()))}")
    print(f"Private mapping -> {args.mapping}  (git-ignored, do NOT publish)")
    print(f"Leak scan: IP clean, no email in metadata. "
          f"{content_emails} emails remain in label transcriptions "
          f"(printed on packaging — kept as dataset content).")


if __name__ == "__main__":
    main()
