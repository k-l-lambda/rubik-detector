"""The engine's FrameTracker on a video with per-frame reference detections (tools/real_video.py results.json, or a
rendered video's labels): from start frames with a good detection, tracked for --horizon frames; sticker-centre error
against the reference after h frames, time per frame, lost tracks, and how the tracker's own health (inliers,
reprojection) relates to the error.

	python engine/tools/track_check.py [--frames DIR] [--fx 1070] [--starts 40] [--horizon 30]
"""
import argparse, json, os, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from flow_check import centre_error  # noqa: E402


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
	load = lambda i: cv2.imread(os.path.join(a.frames, f"frame{i + 1:04d}.png"))
	starts = [i for i in np.linspace(0, n - a.horizon - 2, a.starts * 3).astype(int) if ref_ok[i]][::3][:a.starts]
	errors = {k: [] for k in (1, 3, 5, 10, 15, 30) if k <= a.horizon}
	health, times, lost, survived = [], [], 0, []
	tr = re.Tracker()
	for s in starts:
		tr.seed(load(s), K, *pose(s))
		for k in range(1, a.horizon + 1):
			i = s + k
			t0 = time.perf_counter()
			r = tr.track(load(i))
			times.append((r["ms"]["pyramid"], r["ms"]["flow"], r["ms"]["pose"]))
			if not r["ok"]:
				lost += 1
				survived.append(k - 1)
				break
			if ref_ok[i]:
				e = centre_error(K, r["R"], r["t"], *pose(i))
				health.append((r["inliers"], r["reprojection"], e))
				if k in errors:
					errors[k].append(e)
		else:
			survived.append(a.horizon)
	T = np.array(times)
	print(f"engine FrameTracker from {len(starts)} start frames: per frame pyramid {T[:, 0].mean():.1f} ms, flow {T[:, 1].mean():.1f} ms, pose {T[:, 2].mean():.2f} ms "
		f"(total {T.sum(1).mean():.1f}); lost before {a.horizon} frames: {lost}, frames survived median {np.median(survived):.0f}")
	for k, e in errors.items():
		if e:
			e = np.array(e)
			print(f"  after {k:2d} frames: error vs the detection median {np.median(e):.3f}, p90 {np.percentile(e, 90):.3f}, max {e.max():.2f} cubie, "
				f"< 0.3 on {np.mean(e < 0.3):.0%} ({len(e)})")
	H = np.array(health)
	for lo, hi in ((0, 12), (12, 24), (24, 99)):
		m = (H[:, 0] >= lo) & (H[:, 0] < hi)
		if m.any():
			print(f"  inliers {lo:2d}–{hi:2d}: {m.sum():4d} frames, error median {np.median(H[m, 2]):.3f}, > 0.3 cubie {np.mean(H[m, 2] > 0.3):.0%}")


if __name__ == "__main__":
	main()
