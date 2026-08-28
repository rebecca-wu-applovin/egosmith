#!/usr/bin/env python3
"""HumanTouch standing ship gate: >=100-clip stratified VISUAL alignment QA.

Adopted 2026-08-28 (tier remediation) as the ship gate for the humantouch tier,
replacing the original 18-episode propagation QA. Any manifest rewrite / re-ship
of the tier must re-run this gate on the post-change manifest and record the
verdict table (locked / marginal / off per stratum) next to the change.

Strata (same design as the 2026-08-28 strict audit): 10 clips per task
X001-X010, spread across anchor-assignment angle bins (near <2deg, mid 2-4.5deg,
far >=4.5deg — far is empty post-remediation and its quota folds into near/mid)
and across mount_blocks within each bin, one clip per episode.

Per clip it computes the strict-audit metrics (MANO GT projected through the
viewer render path vs a dark-glove mask: wrist_L/R, medjoint, chamfer bwd) and
writes one overlay render. The VERDICTS ARE VISUAL: a human (or agent) must read
every render (contact sheets emitted for that) and fill verdicts.json with
clip_id -> locked|marginal|off. `report` then prints the gate table.

Machinery is reused from the strict-audit tooling (default
/root/w7_full/humantouch/strict_audit/run_audit.py — kept verbatim as the
audit's authority); poses are the local final/outputs and tars stream from the
shipped bucket.

Usage:
  python humantouch_ship_gate_qa.py sample   [--n_per_task 10] [--seed S]
  python humantouch_ship_gate_qa.py run                       # metrics+renders
  python humantouch_ship_gate_qa.py sheets                    # contact sheets
  python humantouch_ship_gate_qa.py report --verdicts verdicts.json
All artifacts land under --work (default
/root/w7_full/humantouch/ship_gate_qa/<YYYYMMDD>/).
"""
import argparse
import datetime as _dt
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/w7_full/humantouch')
GCS_FRAMES = 'gs://foundational-research/hoi-dataset/egosmith_filtered/humantouch/frames'


def binof(a):
    if a is None:
        return 'unknown'
    return 'near' if a < 2.0 else ('mid' if a < 4.5 else 'far')


def load_clips(manifest, assignments):
    assign = json.load(open(assignments))
    clips = []
    for line in open(manifest):
        d = json.loads(line)
        cid = d['clip_id']
        ep = cid.rsplit('_iv', 1)[0]
        extra = d['descriptor']['extra']
        a = assign.get(ep, {})
        shard = Path(d['descriptor']['root_dir']).parent.name
        clips.append(dict(
            clip_id=cid, episode=ep, task=cid.split('_')[0],
            mount_block=extra.get('mount_block'),
            block_fit_median_px=extra.get('block_fit_median_px'),
            angle_deg=a.get('angle_deg'), angle_bin=binof(a.get('angle_deg')),
            n_frames=len(d['descriptor']['frame_names']),
            tar_uri=f'{GCS_FRAMES}/shard_{shard}/{cid}.tar'))
    return clips


def cmd_sample(args, work):
    clips = load_clips(args.manifest, args.assignments)
    rng = random.Random(args.seed)
    by_task = defaultdict(list)
    for c in clips:
        by_task[c['task']].append(c)
    target = {'near': 4, 'mid': 3, 'far': 3}
    sample = []
    for task in sorted(by_task):
        bybin = defaultdict(list)
        for c in by_task[task]:
            bybin[c['angle_bin']].append(c)
        want = dict(target)
        for b in ('far', 'mid'):        # fold empty-bin quota into near
            avail = len({c['episode'] for c in bybin[b]})
            if avail < want[b]:
                want['near'] += want[b] - avail
                want[b] = avail
        for b in ('far', 'mid', 'near'):
            byblock = defaultdict(list)
            for c in bybin[b]:
                byblock[c['mount_block']].append(c)
            blocks = sorted(byblock)
            rng.shuffle(blocks)
            got, used_eps, bi, tries = 0, set(), 0, 0
            while got < want[b] and tries < 10000 and blocks:
                blk = blocks[bi % len(blocks)]
                bi += 1
                tries += 1
                cs = [c for c in byblock[blk] if c['episode'] not in used_eps]
                if not cs:
                    continue
                c = rng.choice(cs)
                used_eps.add(c['episode'])
                sample.append(c)
                got += 1
    pop = Counter(c['angle_bin'] for c in clips)
    json.dump(dict(seed=args.seed, population_by_bin=dict(pop),
                   n_population=len(clips), n_sample=len(sample), sample=sample),
              open(work / 'sample.json', 'w'), indent=1)
    print(f'population={len(clips)} sample={len(sample)} '
          f'bins={Counter(c["angle_bin"] for c in sample)} '
          f'blocks={len({c["mount_block"] for c in sample})}')


def cmd_run(args, work):
    sys.path.insert(0, str(args.audit_tooling))
    import run_audit as ra
    ra.RENDERS = work / 'renders'
    ra.TARS = work / 'tars'
    ra.RENDERS.mkdir(exist_ok=True)
    ra.TARS.mkdir(exist_ok=True)
    sample = json.load(open(work / 'sample.json'))['sample']
    results = work / 'results.jsonl'
    done = set()
    if results.exists():
        done = {json.loads(l)['clip_id'] for l in open(results)}
    todo = [c for c in sample if c['clip_id'] not in done]
    print(f'{len(todo)} clips to run ({len(done)} done)', flush=True)
    with open(results, 'a') as f:
        for i, c in enumerate(todo):
            try:
                res = ra.audit_clip(c, c['tar_uri'])
                res['status'] = 'ok'
            except Exception as e:  # noqa: BLE001
                res = dict(c)
                res['status'] = f'error: {type(e).__name__}: {e}'
            res.pop('per_frame', None)
            f.write(json.dumps(res) + '\n')
            f.flush()
            (ra.TARS / f"{c['clip_id']}.tar").unlink(missing_ok=True)
            print(f'[{i + 1}/{len(todo)}] {c["clip_id"]} {res["status"][:40]}', flush=True)
    print('GATE_RUN_DONE')


def cmd_sheets(args, work):
    import cv2
    import numpy as np
    rows = [json.loads(l) for l in open(work / 'results.jsonl')]
    rows.sort(key=lambda r: (r['angle_bin'], r['clip_id']))
    (work / 'sheets').mkdir(exist_ok=True)
    COLS, ROWS_, TW, TH, BN = 3, 4, 632, 355, 22
    per = COLS * ROWS_
    for si in range(0, len(rows), per):
        chunk = rows[si:si + per]
        sheet = np.full((ROWS_ * (TH + BN), COLS * TW, 3), 16, np.uint8)
        for ti, it in enumerate(chunk):
            cid = it['clip_id']
            r, c = divmod(ti, COLS)
            y0, x0 = r * (TH + BN), c * TW
            cv2.putText(sheet, f'{si // per:02d}.{ti:02d} [{it["angle_bin"]}] {cid}',
                        (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
            p = work / 'renders' / f'{cid}.jpg'
            if p.exists():
                sheet[y0 + BN:y0 + BN + TH, x0:x0 + TW] = cv2.resize(
                    cv2.imread(str(p)), (TW, TH))
        out = work / 'sheets' / f'gate_{si // per:02d}.jpg'
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(out)


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def cmd_report(args, work):
    verdicts = json.load(open(args.verdicts))
    verdicts.pop('_comment', None)
    rows = [json.loads(l) for l in open(work / 'results.jsonl')]
    smp = json.load(open(work / 'sample.json'))
    popN = smp['n_population']
    pop = smp['population_by_bin']
    missing = [r['clip_id'] for r in rows if r['clip_id'] not in verdicts]
    if missing:
        print(f'GATE INCOMPLETE: {len(missing)} clips lack visual verdicts: {missing[:5]}')
        sys.exit(2)
    table, w_off, w_bad = [], 0.0, 0.0
    for b in ('near', 'mid', 'far'):
        sub = [r for r in rows if r['angle_bin'] == b]
        if not sub and not pop.get(b):
            continue
        n = len(sub)
        k_off = sum(1 for r in sub if verdicts[r['clip_id']] == 'off')
        k_marg = sum(1 for r in sub if verdicts[r['clip_id']] == 'marginal')
        pf = pop.get(b, 0) / popN
        w_off += wilson(k_off, n)[0] * pf if n else 0
        w_bad += wilson(k_off + k_marg, n)[0] * pf if n else 0
        table.append((b, n, k_off, k_marg, pop.get(b, 0), pf))
    print(f'{"bin":6s} {"n":>4s} {"off":>4s} {"marg":>5s} {"population":>10s} {"pop%":>6s}')
    for b, n, o, m, p, pf in table:
        print(f'{b:6s} {n:4d} {o:4d} {m:5d} {p:10d} {100 * pf:5.1f}%')
    print(f'\npopulation-weighted: off={100 * w_off:.2f}%  off|marginal={100 * w_bad:.2f}%')
    verdict = 'PASS' if w_off <= args.max_off_rate else 'FAIL'
    print(f'GATE {verdict} (weighted off-rate {100 * w_off:.2f}% vs '
          f'threshold {100 * args.max_off_rate:.1f}%)')
    json.dump(dict(table=[dict(zip(('bin', 'n', 'off', 'marginal', 'population',
                                    'pop_frac'), t)) for t in table],
                   weighted_off=w_off, weighted_off_or_marginal=w_bad,
                   threshold=args.max_off_rate, gate=verdict,
                   per_clip={r['clip_id']: dict(verdict=verdicts[r['clip_id']],
                                                wrist_worst=r.get('wrist_worst'),
                                                medjoint=r.get('medjoint'),
                                                bwd=r.get('bwd'),
                                                angle_bin=r['angle_bin'])
                             for r in rows}),
              open(work / 'gate_report.json', 'w'), indent=1)
    print(f'-> {work / "gate_report.json"}')
    sys.exit(0 if verdict == 'PASS' else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['sample', 'run', 'sheets', 'report'])
    ap.add_argument('--manifest', default=str(ROOT / 'final' / 'manifest_filtered.jsonl'))
    ap.add_argument('--assignments', default=str(ROOT / 'assignments_finalize.json'))
    ap.add_argument('--audit_tooling', type=Path, default=ROOT / 'strict_audit')
    ap.add_argument('--work', type=Path, default=None)
    ap.add_argument('--n_per_task', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20260828)
    ap.add_argument('--verdicts', default=None)
    ap.add_argument('--max_off_rate', type=float, default=0.02,
                    help='gate threshold on the population-weighted off rate')
    args = ap.parse_args()
    work = args.work or ROOT / 'ship_gate_qa' / _dt.date.today().strftime('%Y%m%d')
    work.mkdir(parents=True, exist_ok=True)
    {'sample': cmd_sample, 'run': cmd_run,
     'sheets': cmd_sheets, 'report': cmd_report}[args.cmd](args, work)


if __name__ == '__main__':
    main()
