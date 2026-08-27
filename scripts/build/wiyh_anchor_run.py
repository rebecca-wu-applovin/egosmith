#!/usr/bin/env python
"""WIYH anchor-pass driver: per-device-day glove->eef extrinsic solve workflow.

Wraps scripts/inspection/wiyh_anchor_pilot.py (prep/solve/render/apply) with the
census-driven block plan, streamed staging, and the extrinsics registry consumed
by the native converter (configs/keypoint_specs/wiyh_native.yaml).

Gate (pilot method): per hand fit_med < 25 px required; LOFO held-out med < 45 px
preferred — 45-70 px accepted only with a passing visual render gate (registered
with flag "lofo_soft", matching the pilot's accepted sessions).

Subcommands:
  plan                        rank locked device-days, choose anchor sessions
  fetch --dd D [--session S]  stage the anchor member (h5 + chest jpgs + masks)
  register --dd D --extr E    gate + write into anchors/extrinsics.json
  status                      registry + plan progress table
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ROOT = Path("/root/w7_native")
WORK = ROOT / "anchor_work"
REG = ROOT / "anchors" / "extrinsics.json"
PLAN = ROOT / "anchors" / "plan.json"
CENSUS = ROOT / "census" / "census.jsonl"
IDX = Path("/root/w7_full/wiyh/index")


def load_reg():
    return json.loads(REG.read_text()) if REG.exists() else {}


def save_reg(reg):
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=1))


def cmd_plan(a):
    from wiyh_gate_census import load_sessions
    rows = [json.loads(l) for l in open(CENSUS) if l.strip()]
    locked = [r for r in rows if r.get("locked")]
    by_dd = defaultdict(list)
    for r in locked:
        by_dd[f"{r['dev']}_{r['date']}"].append(r)
    sess = load_sessions(IDX)
    plan = {}
    for dd, rs in sorted(by_dd.items()):
        n_members = sum(len(sess.get(r["session"], [])) for r in rs)
        # anchor pick: strongest bimanual lock, then most gated frames
        def score(r):
            return (min(r["left"]["frac_lt30"], r["right"]["frac_lt30"]),
                    min(r["left"]["n"], r["right"]["n"]))
        rs.sort(key=score, reverse=True)
        plan[dd] = {
            "scene": rs[0]["scene"],
            "locked_sessions": [r["session"] for r in rs],
            "n_locked_sessions": len(rs),
            "n_members": n_members,
            "est_hours": round(n_members * 22 / 3600.0, 2),
            "anchor_session": rs[0]["session"],
            "anchor_member": rs[0]["member"],
            "backup_member": rs[1]["member"] if len(rs) > 1 else None,
        }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1))
    reg = load_reg()
    tot_h = sum(p["est_hours"] for p in plan.values())
    print(f"[plan] {len(plan)} device-day blocks, {sum(p['n_locked_sessions'] for p in plan.values())} "
          f"locked sessions, ~{tot_h:.1f} h source; {sum(1 for d in plan if d in reg)} already solved")
    for dd, p in sorted(plan.items(), key=lambda kv: -kv[1]["est_hours"]):
        mark = reg.get(dd, {}).get("status", "-")
        print(f"  {dd}  {p['scene']:12s} sess={p['n_locked_sessions']:3d} "
              f"~{p['est_hours']:5.2f}h  [{mark}]")


def cmd_fetch(a):
    import gcsfs
    from wiyh_gate_census import StreamedConcat, load_sessions
    import gzip
    import tarfile

    plan = json.loads(PLAN.read_text())
    p = plan[a.dd]
    member_base = a.member or p["anchor_member"]
    sess = load_sessions(IDX)
    m = next(m for v in sess.values() for m in v if m["base"] == member_base)
    parts = json.loads((IDX / f"{m['scene']}.parts.json").read_text())
    dest = WORK / a.dd / member_base
    if (dest / "dataset.hdf5").exists():
        print(f"already staged: {dest}")
        return
    print(f"staging {member_base} ({m['size']/1e6:.0f} MB gz)", flush=True)
    fs = gcsfs.GCSFileSystem()
    raw = StreamedConcat(fs, parts, int(m["offset"]), int(m["size"]))
    gz = gzip.GzipFile(fileobj=raw, mode="rb")
    kept = 0
    with tarfile.open(fileobj=gz, mode="r|") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            n = mem.name
            if not (n.endswith("dataset.hdf5") or "camera/lf_chest_fisheye/" in n
                    or "hand_masks/lf_chest_fisheye/" in n):
                continue
            rel = "/".join(n.split("/")[2:]) if n.count("/") >= 2 else n
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(mem) as src, open(out, "wb") as w:
                w.write(src.read())
            kept += 1
    print(f"STAGED {dest} ({kept} files)")


def cmd_register(a):
    ex = json.loads(Path(a.extr).read_text())
    reg = load_reg()
    entry = {"source": f"anchor_run {a.extr}", "solved_on": ex.get("sample_dir", "")}
    verdicts = []
    for side in ("left", "right"):
        if side not in ex:
            verdicts.append(f"{side}:MISSING")
            continue
        e = ex[side]
        fit, lofo = e["fit_med_px"], e.get("lofo_med_px")
        entry[side] = {"R": e["R"], "t": e["t"], "fit_med_px": round(fit, 1),
                       "lofo_med_px": round(lofo, 1) if lofo is not None else None,
                       "n_fit": e["n_fit"]}
        if fit >= 25.0:
            verdicts.append(f"{side}:FAIL_fit({fit:.0f})")
        elif lofo is not None and lofo >= 70.0:
            verdicts.append(f"{side}:FAIL_lofo({lofo:.0f})")
        elif lofo is not None and lofo >= 45.0:
            verdicts.append(f"{side}:SOFT_lofo({lofo:.0f})")
        else:
            verdicts.append(f"{side}:OK")
    hard_fail = any("FAIL" in v or "MISSING" in v for v in verdicts)
    soft = any("SOFT" in v for v in verdicts)
    if a.status:
        entry["status"] = a.status  # explicit override after visual gate
    else:
        entry["status"] = "fail" if hard_fail else ("needs_visual" if soft else "pass")
    if soft:
        entry["flags"] = ["lofo_soft"]
    entry["gate"] = verdicts
    reg[a.dd] = entry
    save_reg(reg)
    print(f"[register] {a.dd} -> {entry['status']} ({'; '.join(verdicts)})")


def cmd_status(a):
    plan = json.loads(PLAN.read_text()) if PLAN.exists() else {}
    reg = load_reg()
    n_pass = sum(1 for e in reg.values() if e.get("status") == "pass")
    hrs = sum(p["est_hours"] for dd, p in plan.items()
              if reg.get(dd, {}).get("status") == "pass")
    print(f"[status] blocks={len(plan)} solved_pass={n_pass} "
          f"pending={sum(1 for d in plan if d not in reg)} anchored_hours~{hrs:.1f}")
    for dd, p in sorted(plan.items()):
        e = reg.get(dd, {})
        print(f"  {dd} {p['scene']:12s} {e.get('status','-'):13s} {';'.join(e.get('gate', []))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    q = sub.add_parser("fetch"); q.add_argument("--dd", required=True); q.add_argument("--member", default=None)
    q = sub.add_parser("register"); q.add_argument("--dd", required=True)
    q.add_argument("--extr", required=True); q.add_argument("--status", default=None)
    sub.add_parser("status")
    a = ap.parse_args()
    {"plan": cmd_plan, "fetch": cmd_fetch, "register": cmd_register, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
