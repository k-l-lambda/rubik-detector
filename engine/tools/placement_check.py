"""Where to correct a misplaced evidence chain: a separate step before the state solve, or inside it? An experiment on
the rendered videos (true state and poses known), with wrong placements injected.

	python engine/tools/placement_check.py [--videos v103,v104] [--step 3] [--trials 30] [--block 0.5]

Evidence: the frames before the first twist (the state is fixed there), detected wide at the true focal length, kept
if they fit (cost ≤ 0.09, presence ≥ 0.5), each placed by its true pose; chains are runs of frames at most --gap s
apart and --length s long, reliability groups --block s of a chain (as the app). A trial misplaces k chains: their frames' placements are
taken through one wrong whole-cube rotation (kind: a quarter turn about an axis, as an aliased whole-cube turn reads,
or any other rotation). Methods:

	A  as given (no correction)
	B  separate step: each chain against a leave-one-out estimate from the others — its frames scored under the 24
	   rotations of their placements (rk_score_model); a rotation better by more than τ per sample is taken (all
	   chains at once), then the state solved
	C  inside the solve: estimate from all chains, re-score each chain under that model, move the one with the
	   largest gain over τ per sample, re-estimate; until none moves (at most 6 rounds)

Per method and k: exact states, wrong facelets, confidence of wrong states, chains put right / spoiled, estimate calls.
"""
import argparse, itertools, json, os, random, sys, time
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402

VIDEOS = os.path.expanduser("~/data/rubik-renderer/videos")
PERM_INDEX = {tuple(p): g for g, p in enumerate(cube.PERMS)}


def compose(a, b):
	"""the placement of 'first b, then a': facelet k → PERMS[a][PERMS[b][k]]"""
	return PERM_INDEX[tuple(cube.PERMS[a][cube.PERMS[b][k]] for k in range(54))]


def angle_of(g):
	return np.degrees(np.arccos(np.clip((np.trace(cube.ROTATIONS[g]) - 1) / 2, -1, 1)))


QUARTERS = [g for g in range(24) if abs(angle_of(g) - 90) < 1e-6 and sum(cube.ROTATIONS[g][i][i] == 1 for i in range(3)) == 1]
OTHERS = [g for g in range(24) if g != cube.IDENTITY and g not in QUARTERS]


INVERSE = {g: PERM_INDEX[tuple(np.argsort(cube.PERMS[g]))] for g in range(24)}


def true_placement(R_true, R_det):
	"""the placement (common frame = the labels' cube frame) of a detection with pose R_det of a cube at R_true: the
	renaming g with R_det ≈ R_true · ROTATIONS[g]ᵀ (best of 24) places the view by its inverse (checked: the true
	state exactly, 183 frames of v103)"""
	best = (1e9, None)
	for g in range(24):
		M = R_true @ cube.ROTATIONS[g].T
		a = np.degrees(np.arccos(np.clip((np.trace(M.T @ R_det) - 1) / 2, -1, 1)))
		best = min(best, (a, g))
	return best[0], INVERSE[best[1]]


def load_evidence(name, step, gap, length):
	d = json.load(open(os.path.join(VIDEOS, name, "labels.json")))
	views = d["views"]
	first_twist = d["video"]["twists"][0]["twist"][0] if d["video"]["twists"] else len(views)
	truth = views[0]["facelets"]
	fx = d["intrinsics"]["fx"]
	files = sorted(f for f in os.listdir(os.path.join(VIDEOS, name)) if f.startswith("frame") and f[-4:] in (".png", ".jpg"))
	det = re.Detector()
	frames = []
	for i in range(0, min(first_twist, len(files)), step):
		img = cv2.imread(os.path.join(VIDEOS, name, files[i]))
		h, w = img.shape[:2]
		K = [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]]
		v = det.detect(img, K)
		if not (v["ok"] and v["lattice_cost"] <= 0.09 and v["presence"] >= 0.5):
			continue
		R_true = np.array(views[i]["rotation"]).reshape(3, 3)
		angle, g = true_placement(R_true, v["R"])
		if angle > 10:
			continue
		frames.append(SimpleNamespace(view=v, time=i / d["video"]["fps"], g=g, n=sum(s["valid"] for s in v["samples"])))
	chains = []
	for f in frames:
		if chains and f.time - chains[-1][-1].time <= gap and f.time - chains[-1][0].time < length:
			chains[-1].append(f)
		else:
			chains.append([f])
	return truth, chains


def estimate(chains, placements, block):
	views, rots, groups = [], [], []
	for c, (chain, G) in enumerate(zip(chains, placements)):
		for f in chain:
			views.append(f.view)
			rots.append(compose(G, f.g))
			groups.append(c * 4096 + int(f.time / block))
	covered = len({cube.PERMS[r][s["facelet"]] for v, r in zip(views, rots) for s in v["samples"] if s["valid"]})
	e = re.estimate(views, rots, require_all_faces=False, legal=covered >= 45, partial=True, reliability=True, groups=groups)
	return e


def chain_gains(chain, G, model, state):
	"""per candidate extra rotation H (on top of G): the chain's cost under the model, H = identity first"""
	costs = []
	for H in range(24):
		GH = compose(H, G)
		costs.append(sum(re.score_model(model, f.view, compose(GH, f.g), [state])[0] for f in chain))
	return np.array(costs)


def method_B(chains, placements, block, tau):
	calls = 0
	new = list(placements)
	for c in range(len(chains)):
		rest = [i for i in range(len(chains)) if i != c]
		e = estimate([chains[i] for i in rest], [placements[i] for i in rest], block)
		calls += 1
		if not e["ok"]:
			continue
		costs = chain_gains(chains[c], placements[c], e["model"], e["state"])
		n = sum(f.n for f in chains[c])
		H = int(np.argmin(costs))
		if H != cube.IDENTITY and (costs[cube.IDENTITY] - costs[H]) / n > tau:
			new[c] = compose(H, placements[c])
	e = estimate(chains, new, block)
	return e, new, calls + 1


def method_C(chains, placements, block, tau, rounds=6):
	new = list(placements)
	calls = 0
	for _ in range(rounds):
		e = estimate(chains, new, block)
		calls += 1
		if not e["ok"]:
			return e, new, calls
		best = (0.0, None, None)
		for c in range(len(chains)):
			costs = chain_gains(chains[c], new[c], e["model"], e["state"])
			n = sum(f.n for f in chains[c])
			H = int(np.argmin(costs))
			gain = (costs[cube.IDENTITY] - costs[H]) / n
			if H != cube.IDENTITY and gain > tau and gain > best[0]:
				best = (gain, c, H)
		if best[1] is None:
			return e, new, calls
		new[best[1]] = compose(best[2], new[best[1]])
	e = estimate(chains, new, block)
	return e, new, calls + 1


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--videos", default="v103,v104")
	ap.add_argument("--step", type=int, default=3)
	ap.add_argument("--gap", type=float, default=0.25)
	ap.add_argument("--length", type=float, default=1.0, help="a chain at most this long (s): a hand-held cube never pauses")
	ap.add_argument("--block", type=float, default=0.5)
	ap.add_argument("--trials", type=int, default=30)
	ap.add_argument("--tau", type=float, default=1.0)
	ap.add_argument("--seed", type=int, default=1)
	a = ap.parse_args()
	rng = random.Random(a.seed)
	data = {}
	for name in a.videos.split(","):
		t0 = time.time()
		truth, chains = load_evidence(name, a.step, a.gap, a.length)
		ident = [cube.IDENTITY] * len(chains)
		e = estimate(chains, ident, a.block)
		wrong = cube.compare(e["state"], truth)[0] if e["ok"] else 54
		print(f"{name}: {sum(len(c) for c in chains)} frames in {len(chains)} chains (sizes {[len(c) for c in chains]}), "
			f"true placements → {wrong} wrong, min confidence {e['min_confidence']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
		data[name] = (truth, chains)
	# τ: the gains of correctly placed chains (leave-one-out, no misplacement) — no correction should fire on them
	for name, (truth, chains) in data.items():
		ident = [cube.IDENTITY] * len(chains)
		gains = []
		for c in range(len(chains)):
			rest = [i for i in range(len(chains)) if i != c]
			e = estimate([chains[i] for i in rest], [ident[i] for i in rest], a.block)
			if e["ok"]:
				costs = chain_gains(chains[c], ident[c], e["model"], e["state"])
				n = sum(f.n for f in chains[c])
				gains.append(round(float((costs[cube.IDENTITY] - costs.min()) / n), 3))
		print(f"  {name}: best gain per sample over the true placement, per chain (leave-one-out): {gains}", flush=True)
	results = {}
	for name, (truth, chains) in data.items():
		for k in (0, 1, 2):
			for kind in (["none"] if k == 0 else ["quarter", "other"]):
				for t in range(a.trials if k else 3):
					placements = [cube.IDENTITY] * len(chains)
					bad = rng.sample(range(len(chains)), k) if k else []
					for c in bad:
						placements[c] = rng.choice(QUARTERS if kind == "quarter" else OTHERS)
					for m, fn in (("A", None), ("B", method_B), ("C", method_C)):
						t0 = time.time()
						if fn is None:
							e, new, calls = estimate(chains, placements, a.block), placements, 1
						else:
							e, new, calls = fn(chains, placements, a.block, a.tau)
						ms = (time.time() - t0) * 1000
						wrong = cube.compare(e["state"], truth)[0] if e["ok"] and e["legal"] else 54
						fixed = sum(1 for c in bad if new[c] == cube.IDENTITY)
						spoiled = sum(1 for c in range(len(chains)) if c not in bad and new[c] != cube.IDENTITY)
						r = results.setdefault((name, k, kind, m), [])
						r.append((wrong, e["min_confidence"] if e["ok"] else 0.0, fixed, spoiled, calls, ms, k))
	print(f"\nτ = {a.tau} per sample; per video, misplaced chains k, kind, method: exact / trials, wrong facelets (mean), "
		f"wrong states with min confidence ≥ 0.99, misplaced put right, right ones spoiled, estimate calls, ms")
	for (name, k, kind, m), r in sorted(results.items()):
		exact = sum(1 for x in r if x[0] == 0)
		sure_wrong = sum(1 for x in r if x[0] > 0 and x[1] >= 0.99)
		print(f"  {name} k={k} {kind:7s} {m}: exact {exact:2d}/{len(r)}  wrong {np.mean([x[0] for x in r]):5.1f}  sure-but-wrong {sure_wrong:2d}  "
			f"fixed {sum(x[2] for x in r)}/{sum(x[6] for x in r)}  spoiled {sum(x[3] for x in r)}  calls {np.mean([x[4] for x in r]):4.1f}  {np.mean([x[5] for x in r]):6.0f} ms")


if __name__ == "__main__":
	main()
