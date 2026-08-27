#!/usr/bin/env python
"""Stamp the WIYH native tier's quality tag onto every manifest record.

Adds record-level metadata (ClipManifestRecord.metadata, the DexCap-precedent
channel) so consumers see the caveat without reading descriptor.extra:
  finger_quality: "approximate_35_65px"
  finger_articulation_note: mask finger-level targets for pixel-tight work
  anchor_acceptance / anchor_session: provenance of the glove->eef solve

Usage: python scripts/build/wiyh_patch_manifest_metadata.py m1.jsonl [m2.jsonl ...]
"""
import json
import sys

NOTE = ("finger positions derive from a per-session vision-anchored glove->eef "
        "extrinsic (fit 15-45 px); wrist translation is sensor-locked. Mask "
        "finger-level targets for pixel-tight applications.")

for path in sys.argv[1:]:
    rows = []
    n = 0
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        ex = r.get("descriptor", {}).get("extra", {})
        md = r.get("metadata") or {}
        md.update({
            "finger_quality": "approximate_35_65px",
            "finger_articulation_note": NOTE,
            "anchor_acceptance": ex.get("anchor_acceptance", "unknown"),
            "anchor_session": ex.get("anchor_extrinsic", ""),
        })
        r["metadata"] = md
        rows.append(r)
        n += 1
    with open(path, "w") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {n} records tagged")
