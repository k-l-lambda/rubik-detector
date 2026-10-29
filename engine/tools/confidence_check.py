"""Why the state confidence stays where it does: the evidence of a video accumulated as the app does (state_tracker's
Tracker, every --step-th frame), then per facelet its observations classified against the final clusters, over time.

	python engine/tools/confidence_check.py VIDEO_DIR [--fx 1070] [--step 3] [--until SECONDS]

Prints the estimate's confidence distribution, and per uncertain facelet (confidence < 0.99) its votes in time order:
a switch from one colour to another (the cube was turned while the evidence accumulated), colours alternating
(ambiguous colour or misplaced views), or a few odd votes (outliers). And how often whole frames disagree with the
final state (a frame placed wrong, or the state changed): the share of each frame's samples on another colour.
"""
import argparse, os, sys
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from state_tracker import Tracker, estimate, FACES  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("video")
	ap.add_argument("--fx", type=float, default=1070)
	ap.add_argument("--step", type=int, default=3)
	ap.add_argument("--until", type=float, default=1e9)
	a = ap.parse_args()
	files = sorted(f for f in os.listdir(a.video) if f.startswith("frame") and f.endswith(".png"))
	img0 = cv2.imread(os.path.join(a.video, files[0]))
	h, w = img0.shape[:2]
	K = np.array([[a.fx, 0, w / 2], [0, a.fx, h / 2], [0, 0, 1]])
	args = SimpleNamespace(fallback_cost=0.15, evidence_cost=0.09, relative_cost=1.5, confirm=1, link_deg=20.0, agree_colour=12.0,
		unit="frame", outlier=0.0, legal_from=45)
	tr = Tracker(args, K)
	times = []
	for i in range(0, len(files), a.step):
		if i / 30 > a.until:
			break
		n = len(tr.frames)
		tr.detect(cv2.imread(os.path.join(a.video, files[i])), i)
		times += [i / 30] * (len(tr.frames) - n)
	e = estimate(tr.frames, tr.rotations, args)
	conf = np.array(e["confidence"]) if e["ok"] else np.zeros(54)
	print(f"{len(tr.frames)} evidence frames in {len({v['chain'] for v in tr.frames})} chains over {times[-1] if times else 0:.1f} s; "
		f"estimate legal {e['legal']}, {e['covered']}/54 observed")
	print(f"  confidence: min {conf.min():.3f}, facelets < 0.99: {int((conf < 0.99).sum())}, < 0.9: {int((conf < 0.9).sum())}, < 0.6: {int((conf < 0.6).sum())}")
	if not e["ok"] or not e["legal"]:
		return
	state = e["state"]
	# the final clusters: per letter the median Lab of the observations at facelets the state gives that letter
	obs = [[] for _ in range(54)]   # (time, lab, frame number)
	for n_, (v, g, t) in enumerate(zip(tr.frames, tr.rotations, times)):
		for s in v["samples"]:
			if s["valid"]:
				obs[cube.PERMS[g][s["facelet"]]].append((t, np.array(s["lab"]), n_))
	centre = {c: np.median([o[1] for k in range(54) if state[k] == c for o in obs[k]], axis=0) for c in FACES}
	feature = lambda lab: np.array([lab[0] * 0.5, lab[1], lab[2]])   # lightness counts half (shading)
	nearest = lambda lab: min(FACES, key=lambda c: np.linalg.norm(feature(lab) - feature(centre[c])))
	print("  cluster centres (L a b): " + "  ".join(f"{c} ({centre[c][0]:.0f} {centre[c][1]:.0f} {centre[c][2]:.0f})" for c in FACES))
	d = {(x, y): np.linalg.norm(feature(centre[x]) - feature(centre[y])) for x in FACES for y in FACES if x < y}
	(x, y), dmin = min(d.items(), key=lambda kv: kv[1])
	print(f"  closest clusters: {x}–{y} at {dmin:.1f}")
	# per frame: the share of its samples whose nearest cluster is not the state's letter there
	frame_bad = {}
	for k in range(54):
		for t, lab, n_ in obs[k]:
			b = frame_bad.setdefault(n_, [0, 0, t])
			b[1] += 1
			b[0] += nearest(lab) != state[k]
	shares = sorted((b[0] / b[1], b[2], n_) for n_, b in frame_bad.items())
	bad_frames = [(round(t, 1), round(sh, 2)) for sh, t, n_ in shares if sh > 0.3]
	print(f"  frames with > 30 % of their samples on another colour: {len(bad_frames)}/{len(shares)}: " + " ".join(f"{t}s:{sh}" for t, sh in sorted(bad_frames)[:20]))
	kinds = {"switch": 0, "alternating": 0, "outliers": 0}
	for k in np.argsort(conf):
		if conf[k] >= 0.99:
			break
		votes = [(t, nearest(lab)) for t, lab, _ in sorted(obs[k], key=lambda o: o[0])]
		seq = "".join(c for _, c in votes)
		changes = sum(1 for p, q in zip(seq, seq[1:]) if p != q)
		other = sum(1 for c in seq if c != state[k])
		kind = "outliers" if other <= max(1, len(seq) // 8) else ("switch" if changes <= 2 else "alternating")
		kinds[kind] += 1
		if sum(kinds.values()) <= 12:
			t_switch = next((t for (t, c), (_, c2) in zip(votes, votes[1:]) if c != c2), None)
			print(f"  facelet {k:2d} ({FACES[k // 9]}{k % 9}) state {state[k]} confidence {conf[k]:.2f}: {len(seq)} votes {seq}  → {kind}"
				+ (f" (first change at {t_switch:.1f} s)" if kind == "switch" and t_switch is not None else ""))
	print(f"  uncertain facelets by pattern: {kinds}")


if __name__ == "__main__":
	main()
