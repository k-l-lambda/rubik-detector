"""Feasibility of optical-flow tracking between detections, on a real video with per-frame detections
(tools/real_video.py results.json) as the reference.

	python engine/tools/flow_check.py [--frames DIR] [--fx 1070] [--starts 40] [--horizon 30]

1. Flow tracking: from a start frame with a good detection, the visible lattice vertices (the gap crossings of the
   three faces turned to the camera, 37 points) are tracked frame to frame with pyramidal Lucas–Kanade (forward–
   backward check), and the pose solved from them each frame (solvePnPRansac seeded with the previous pose). Against
   the reference detections: sticker-centre error (cubie units, matched, so frame renamings don't matter) after h
   frames, per-frame time, and whether the tracker's own health (inliers, reprojection error) tells a bad pose.
2. ROI detection: the engine's local detection on a crop around the predicted cube (the start pose's projection
   + 30 %), against the same detection on the whole frame — time and result.
"""
import argparse, json, os, sys, time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))


def lattice_vertices():
	pts = set()
	for a in range(3):
		b, c = (a + 1) % 3, (a + 2) % 3
		for u in (-1.5, -0.5, 0.5, 1.5):
			for v in (-1.5, -0.5, 0.5, 1.5):
				p = [0.0, 0.0, 0.0]
				p[a], p[b], p[c] = 1.5, u, v
				pts.add(tuple(p))
	return np.array(sorted(pts))


def sticker_centres():
	pts = []
	for a in range(3):
		b, c = (a + 1) % 3, (a + 2) % 3
		for u in (-1, 0, 1):
			for v in (-1, 0, 1):
				p = [0.0, 0.0, 0.0]
				p[a], p[b], p[c] = 1.5, u, v
				pts.append(p)
	return np.array(pts)


def project(K, R, t, P):
	q = (R @ P.T).T + t
	return (K @ q.T).T[:, :2] / q[:, 2:3], q[:, 2]


def facing(R, t):
	"""the pose renamed so that the faces turned to the camera are +x, +y, +z (the view frame the sticker centres assume)"""
	c = -R.T @ t
	best = max((np.array(g) for g in _rotations()), key=lambda G: min(G.T @ c))
	return R @ best


def _rotations():
	from itertools import permutations, product
	out = []
	for p in permutations(range(3)):
		for s in product((-1, 1), repeat=3):
			M = np.zeros((3, 3))
			for i in range(3):
				M[i, p[i]] = s[i]
			if np.linalg.det(M) > 0:
				out.append(M)
	return out


def centre_error(K, Ra, ta, Rb, tb):
	"""mean distance between the two poses' visible sticker centres (each renamed to face the camera), matched, in
	units of the sticker pitch"""
	Ra, Rb = facing(Ra, ta), facing(Rb, tb)
	C = sticker_centres()
	pa, _ = project(K, Ra, ta, C)
	pb, _ = project(K, Rb, tb, C)
	d = np.linalg.norm(pa[:, None] - pb[None], axis=2)
	r, c = linear_sum_assignment(d)
	pitch = np.median(np.sort(np.linalg.norm(pb[:, None] - pb[None], axis=2), axis=1)[:, 1])
	return float(d[r, c].mean() / pitch)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--frames", default=os.path.expanduser("~/data/rubik-renderer/real/vid225115"))
	ap.add_argument("--fx", type=float, default=1070)
	ap.add_argument("--starts", type=int, default=40)
	ap.add_argument("--horizon", type=int, default=30)
	a = ap.parse_args()
	import rubik_engine as re
	R = json.load(open(os.path.join(a.frames, "eval", "results.json")))["frames"]
	n = len(R)
	img0 = cv2.imread(os.path.join(a.frames, "frame0001.png"))
	h, w = img0.shape[:2]
	K = np.array([[a.fx, 0, w / 2], [0, a.fx, h / 2], [0, 0, 1]])
	ref_ok = [r["ok"] and r["cost"] <= 0.08 for r in R]
	pose = lambda i: (np.array(R[i]["R"]), np.array(R[i]["t"]))
	gray = {}
	def frame(i):
		if i not in gray:
			gray[i] = cv2.cvtColor(cv2.imread(os.path.join(a.frames, f"frame{i + 1:04d}.png")), cv2.COLOR_BGR2GRAY)
		return gray[i]
	V = lattice_vertices()
	starts = [i for i in np.linspace(0, n - a.horizon - 2, a.starts * 3).astype(int) if ref_ok[i]][::3][:a.starts]
	lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
	errors = {k: [] for k in (1, 3, 5, 10, 15, 30) if k <= a.horizon}
	health = []          # (inliers, reprojection px, error) per tracked frame with a reference
	ms_flow, ms_pnp, lost = [], [], 0
	for s in starts:
		Rs, ts = pose(s)
		pts, depth = project(K, Rs, ts, V)
		keep = (depth > 0) & (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
		obj, p = V[keep], pts[keep].astype(np.float32).reshape(-1, 1, 2)
		rvec, _ = cv2.Rodrigues(Rs)
		tvec = ts.reshape(3, 1).copy()
		for k in range(1, a.horizon + 1):
			i = s + k
			t0 = time.perf_counter()
			p1, st, _ = cv2.calcOpticalFlowPyrLK(frame(i - 1), frame(i), p, None, **lk)
			p0b, st2, _ = cv2.calcOpticalFlowPyrLK(frame(i), frame(i - 1), p1, None, **lk)
			fb = np.linalg.norm((p0b - p).reshape(-1, 2), axis=1)
			good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
			t1 = time.perf_counter()
			if good.sum() < 6:
				lost += 1
				break
			okp, rvec2, tvec2, inl = cv2.solvePnPRansac(obj[good], p1[good].reshape(-1, 2), K, None, rvec.copy(), tvec.copy(), useExtrinsicGuess=True,
				iterationsCount=50, reprojectionError=4.0, flags=cv2.SOLVEPNP_ITERATIVE)
			t2 = time.perf_counter()
			ms_flow.append((t1 - t0) * 1000)
			ms_pnp.append((t2 - t1) * 1000)
			if not okp or inl is None or len(inl) < 6:
				lost += 1
				break
			rvec, tvec = rvec2, tvec2
			Rt, _ = cv2.Rodrigues(rvec)
			proj, _ = cv2.projectPoints(obj[good][inl.ravel()], rvec, tvec, K, None)
			rep = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - p1[good][inl.ravel()].reshape(-1, 2), axis=1)))
			# keep tracking only the points that survived (as a tracker would until the next detection re-seeds)
			obj, p = obj[good], p1[good]
			if ref_ok[i]:
				e = centre_error(K, Rt, tvec.ravel(), *pose(i))
				health.append((len(inl), rep, e))
				if k in errors:
					errors[k].append(e)
	print(f"flow tracking from {len(starts)} start frames (good detections), Lucas–Kanade on the lattice vertices + solvePnPRansac:")
	print(f"  per frame: flow {np.mean(ms_flow):.1f} ms (forward + backward), PnP {np.mean(ms_pnp):.1f} ms; tracks lost before {a.horizon} frames: {lost}")
	for k, e in errors.items():
		if e:
			e = np.array(e)
			print(f"  after {k:2d} frames ({k / 30:.2f} s): sticker-centre error vs the detection median {np.median(e):.2f}, p90 {np.percentile(e, 90):.2f} cubie, "
				f"< 0.3 cubie on {np.mean(e < 0.3):.0%} ({len(e)} with a reference)")
	H = np.array(health)
	if len(H):
		for lo, hi in ((0, 12), (12, 20), (20, 99)):
			m = (H[:, 0] >= lo) & (H[:, 0] < hi)
			if m.any():
				print(f"  inliers {lo:2d}–{hi:2d}: {m.sum():4d} frames, error median {np.median(H[m, 2]):.2f}, > 0.3 cubie {np.mean(H[m, 2] > 0.3):.0%}")
		for lo, hi in ((0, 1), (1, 2), (2, 99)):
			m = (H[:, 1] >= lo) & (H[:, 1] < hi)
			if m.any():
				print(f"  reprojection {lo}–{hi} px: {m.sum():4d} frames, error median {np.median(H[m, 2]):.2f}, > 0.3 cubie {np.mean(H[m, 2] > 0.3):.0%}")

	# ROI detection vs whole-frame local detection, from the previous frame's pose
	det = re.Detector()
	rows = []
	for s in starts:
		i = s + 1
		if not ref_ok[i]:
			continue
		img = cv2.imread(os.path.join(a.frames, f"frame{i + 1:04d}.png"))
		Rs, ts = pose(s)
		corners = np.array([[x, y, z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
		uv, _ = project(K, Rs, ts, corners)
		(u0, v0), (u1, v1) = uv.min(0), uv.max(0)
		mu, mv = 0.3 * (u1 - u0), 0.3 * (v1 - v0)
		x0, y0 = int(max(0, u0 - mu)), int(max(0, v0 - mv))
		x1, y1 = int(min(w, u1 + mu)), int(min(h, v1 + mv))
		crop = np.ascontiguousarray(img[y0:y1, x0:x1])
		Kc = K.copy()
		Kc[0, 2] -= x0
		Kc[1, 2] -= y0
		t0 = time.perf_counter(); full = det.detect(img, K, prior=(Rs, ts)); t1 = time.perf_counter()
		roi = det.detect(crop, Kc, prior=(Rs, ts)); t2 = time.perf_counter()
		e_full = centre_error(K, full["R"], full["t"], *pose(i)) if full["ok"] else 9
		e_roi = centre_error(K, roi["R"], roi["t"], *pose(i)) if roi["ok"] else 9
		rows.append(((t1 - t0) * 1000, (t2 - t1) * 1000, full["lattice_cost"], roi["lattice_cost"], e_full, e_roi, crop.shape[1] * crop.shape[0] / (w * h),
			roi["ms"]["segments"], full["ms"]["segments"]))
	X = np.array(rows)
	print(f"\nlocal detection on a crop around the predicted cube ({len(X)} frames, crop = {np.mean(X[:, 6]):.0%} of the frame's pixels):")
	print(f"  whole frame: {np.mean(X[:, 0]):.0f} ms (segments {np.mean(X[:, 8]):.0f}), cost median {np.median(X[:, 2]):.3f}, error vs the reference median {np.median(X[:, 4]):.2f} cubie")
	print(f"  crop       : {np.mean(X[:, 1]):.0f} ms (segments {np.mean(X[:, 7]):.0f}), cost median {np.median(X[:, 3]):.3f}, error vs the reference median {np.median(X[:, 5]):.2f} cubie; "
		f"good (cost ≤ 0.15) {np.mean(X[:, 3] <= 0.15):.0%} vs whole frame {np.mean(X[:, 2] <= 0.15):.0%}")


if __name__ == "__main__":
	main()
