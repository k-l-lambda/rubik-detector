"""Per view: where the visible stickers are.

`oracle(view)` uses the truth quads (to test the colour and fusion stages alone).

`LineGeometry` finds the cube from line segments, knowing the camera intrinsics K (a phone exposes them):
	1. segments: LSD on gray and on Lab L*, a*, b* (colour edges between equally bright stickers); weight = length ×
	   colour contrast across the segment
	2. cube axes: every segment is the image of a 3D line, whose direction lies in the segment's interpretation plane
	   (normal n = K⁻¹p₁ × K⁻¹p₂). The cube's edges run along three orthogonal directions; RANSAC over segment
	   triples (two give an axis d₁ = n₁ × n₂, a third with d₁ fixes d₂ = d₁ × n₃) and scoring by weighted n·d ≈ 0
	   finds them; refined by alternating assignment and smallest-eigenvector fits, re-orthogonalized
	3. signs: each axis is pointed so that its face turns to the camera; this names the visible faces R, U, F of the
	   view's own frame (which physical face is which is left to fusion)
	4. position: the cube centre t = Z·K⁻¹[u, v, 1]. Every segment of axis family a must lie on one of the two
	   visible faces containing a; back-projected onto that face plane, its perpendicular coordinate should sit on a
	   cubie boundary (…, -0.5, 0.5, …) and its extent inside the face. A robust cost over (u, v, log Z), grid search
	   then Nelder–Mead, then a joint refinement with a small rotation
	5. stickers: the 27 visible facelets projected
"""
import time

import cv2
import numpy as np
from scipy.optimize import minimize

from . import colors, cube


SAMPLE = 0.5	# sampled part of a sticker (side share)


class GeometryError(Exception):
	pass


def oracle(view, sample=SAMPLE):
	"""Truth quads, shrunk to the sampled part."""
	return {"quads": {k: colors.shrink(q, sample) for k, q in view.quads.items()}, "debug": {}}


def project(K, R, t, points):
	P = points @ R.T + t
	return np.c_[K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]]


# visible faces in a view's own frame, and their facelets
VISIBLE = [cube.FACES.index(f) for f in "RUF"]
VISIBLE_FACELETS = [k for k in range(54) if cube.FACE_OF[k] in VISIBLE]


def facelet_quads(K, R, t, half=SAMPLE / 2 * 0.92, plane=1.5):
	"""Image quads (sampled part) of the R, U, F facelets of a cube at pose (R, t) in its own frame."""
	pts = cube.facelet_points(plane, ((-half, -half), (half, -half), (half, half), (-half, half)))
	out = {}
	for k in VISIBLE_FACELETS:
		out[k] = project(K, R, t, pts[k])
	return out


# --- line segments -------------------------------------------------------------------------------------------------

def segments(image, min_length=0.012):
	"""LSD segments over gray and Lab channels: (N, 4) x1 y1 x2 y2 and weights."""
	h, w = image.shape[:2]
	lab = colors.to_lab(image)
	chans = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
		np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8),
		np.clip(lab[..., 1] * 2 + 128, 0, 255).astype(np.uint8),
		np.clip(lab[..., 2] * 2 + 128, 0, 255).astype(np.uint8)]
	lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
	segs = []
	for c in chans:
		found = lsd.detect(c)[0]
		if found is not None:
			segs.append(found.reshape(-1, 4))
	if not segs:
		return np.zeros((0, 4)), np.zeros(0)
	segs = np.concatenate(segs).astype(float)
	length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
	segs = segs[length >= min_length * max(h, w)]
	length = length[length >= min_length * max(h, w)]

	# contrast: Lab difference 2 px to either side, sampled along the segment
	d = (segs[:, 2:] - segs[:, :2]) / length[:, None]
	nrm = np.c_[-d[:, 1], d[:, 0]]
	ts = np.linspace(0.15, 0.85, 7)
	pts = segs[:, None, :2] + ts[None, :, None] * (segs[:, None, 2:] - segs[:, None, :2])

	def at(p):
		x = np.clip(np.rint(p[..., 0]).astype(int), 0, w - 1)
		y = np.clip(np.rint(p[..., 1]).astype(int), 0, h - 1)
		return lab[y, x]

	a, b = at(pts + 2.5 * nrm[:, None]), at(pts - 2.5 * nrm[:, None])
	contrast = np.median(np.linalg.norm(colors.weigh(a) - colors.weigh(b), axis=-1), axis=1)
	return segs, length * contrast


def plane_normals(K, segs):
	Kinv = np.linalg.inv(K)
	p1 = np.c_[segs[:, :2], np.ones(len(segs))] @ Kinv.T
	p2 = np.c_[segs[:, 2:], np.ones(len(segs))] @ Kinv.T
	n = np.cross(p1, p2)
	return n / np.linalg.norm(n, axis=1, keepdims=True)


# --- cube axes -----------------------------------------------------------------------------------------------------

def _orthonormal(D):
	U, _, Vt = np.linalg.svd(D)
	return U @ Vt


def axes(normals, weights, rng, hypotheses=3000, tol=np.sin(np.radians(1.5))):
	"""Three orthogonal directions (rows) maximizing the weighted support |n·d| < tol, and the family of each
	segment (-1: none)."""
	n = len(normals)
	if n < 6:
		raise GeometryError(f"only {n} segments")
	p = weights / weights.sum()
	idx = rng.choice(n, size=(hypotheses, 3), p=p)
	d1 = np.cross(normals[idx[:, 0]], normals[idx[:, 1]])
	d2 = np.cross(d1, normals[idx[:, 2]])
	ok = (np.linalg.norm(d1, axis=1) > 1e-6) & (np.linalg.norm(d2, axis=1) > 1e-6)
	d1, d2 = d1[ok], d2[ok]
	d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
	d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
	d3 = np.cross(d1, d2)
	D = np.stack([d1, d2, d3], axis=1)	# (H, 3, 3)

	def score(D):
		r = np.abs(np.einsum("nk,hak->hna", normals, D))	# (H, N, 3)
		s = np.clip(1 - (r / tol) ** 2, 0, None)
		return (s.max(axis=2) * weights).sum(axis=1)

	scores = np.concatenate([score(D[i:i + 500]) for i in range(0, len(D), 500)])
	D = D[np.argmax(scores)]

	for tol_ in (tol * 2, tol * 1.5, tol):
		for _ in range(4):
			r = np.abs(normals @ D.T)
			fam = np.where(r.min(axis=1) < tol_, r.argmin(axis=1), -1)
			for a in range(3):
				m = fam == a
				if m.sum() >= 2:
					M = (normals[m] * weights[m, None]).T @ normals[m]
					D[a] = np.linalg.eigh(M)[1][:, 0]
			D = _orthonormal(D)
	r = np.abs(normals @ D.T)
	fam = np.where(r.min(axis=1) < tol, r.argmin(axis=1), -1)
	return D, fam


# --- position ------------------------------------------------------------------------------------------------------

def _rays(K, pts):
	r = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T
	return r


def lattice_cost(R, t, rays1, rays2, fam, w, outside=1.0, tol=0.35):
	"""Mean per-segment cost of segments (endpoint rays rays1, rays2; family fam = cube axis 0..2) against a cube
	(cube→camera R, centre t). A segment of axis a lies on face x_b = +1.5 for one of the two other axes b; the
	cheaper is taken. On the face: the perpendicular coordinate (axis c, the third) should be near a cubie boundary
	(…, -0.5, 0.5, …, outer edges ±1.5 included) and the segment within the face."""
	cost = np.full(len(fam), outside)
	# the camera must be in front of every visible face (normal·t < -1.5 with a margin): on a face plane every ray
	# meets it at the camera itself, a degenerate zero-cost pose
	if (R.T @ t).max() > -1.5 - 0.3:
		return float(outside)
	for a in range(3):
		m = fam == a
		if not m.any():
			continue
		best = np.full(m.sum(), outside)
		for b in range(3):
			if b == a:
				continue
			c = 3 - a - b
			nb = R[:, b]	# face normal (camera frame)
			# ray s·r hits the plane nb·(x - t) = 1.5 at s = (1.5 + nb·t) / (nb·r)
			hit = []
			for rays in (rays1[m], rays2[m]):
				den = rays @ nb
				s = (1.5 + nb @ t) / np.where(np.abs(den) < 1e-9, 1e-9, den)
				X = s[:, None] * rays - t	# camera frame, relative to the centre
				hit.append((X @ R, s))	# cube frame
			(P1, s1), (P2, s2) = hit
			valid = (s1 > 0) & (s2 > 0)
			pc = (P1[:, c] + P2[:, c]) / 2	# perpendicular coordinate
			lo, hi = np.minimum(P1[:, a], P2[:, a]), np.maximum(P1[:, a], P2[:, a])
			inside = valid & (np.abs(pc) <= 1.6) & (lo >= -1.65) & (hi <= 1.65)
			phase = np.abs((pc % 1.0) - 0.5)	# distance to the nearest half-integer: 0 on a boundary, 0.5 mid-sticker
			cst = np.where(inside, np.clip(phase / tol, 0, 1) ** 2, outside)
			best = np.minimum(best, cst)
		cost[m] = best
	return float((cost * w).sum() / w.sum())


def position(K, R, segs, fam, w, rng):
	"""Cube centre t for axes R by minimizing lattice_cost over (u, v, log Z): grid, then Nelder–Mead."""
	m = fam >= 0
	segs, fam, w = segs[m], fam[m], w[m]
	rays1, rays2 = _rays(K, segs[:, :2]), _rays(K, segs[:, 2:])
	ends = np.r_[segs[:, :2], segs[:, 2:]]
	ww = np.r_[w, w]

	def wq(x, q):
		o = np.argsort(x)
		c = np.cumsum(ww[o]) / ww.sum()
		return x[o][np.searchsorted(c, q)]

	lo = np.array([wq(ends[:, 0], 0.03), wq(ends[:, 1], 0.03)])
	hi = np.array([wq(ends[:, 0], 0.97), wq(ends[:, 1], 0.97)])
	centre, extent = (lo + hi) / 2, max(hi - lo)
	f = K[0, 0]
	Z0 = f * 4.9 / max(extent, 1)	# corner view: the outline spans ~4.9 cubie pitches

	def t_of(x):
		u, v, lz = x
		return np.exp(lz) * np.array([(u - K[0, 2]) / f, (v - K[1, 2]) / K[1, 1], 1.0])

	def cost(x, Rm=R, tol=0.35):
		return lattice_cost(Rm, t_of(x), rays1, rays2, fam, w, tol=tol)

	# coarse: a wide phase tolerance makes the basin of the true pose wider than the grid step
	grid = []
	for lz in np.log(Z0) + np.linspace(-0.45, 0.45, 13):
		for du in np.linspace(-0.25, 0.25, 9) * extent:
			for dv in np.linspace(-0.25, 0.25, 9) * extent:
				x = np.array([centre[0] + du, centre[1] + dv, lz])
				grid.append((cost(x, tol=0.5), x))
	grid.sort(key=lambda g: g[0])

	# refine the best few, distinct, starts: tolerance 0.5 → 0.35
	starts, fits = [], []
	for c, x in grid:
		if all(np.linalg.norm((x[:2] - y[:2]) / extent) > 0.06 or abs(x[2] - y[2]) > 0.05 for y in starts):
			starts.append(x)
		if len(starts) == 5:
			break
	for x in starts:
		for tol in (0.5, 0.35):
			x = minimize(lambda y: cost(y, tol=tol), x, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-5}).x
		fits.append((cost(x), x))
	c, x = min(fits, key=lambda f: f[0])

	# the lattice repeats every cubie: try the pose shifted by ±1 cubie along each axis (cube frame)
	def shifted(x, d):
		t = t_of(x) + R @ d
		return np.array([K[0, 0] * t[0] / t[2] + K[0, 2], K[1, 1] * t[1] / t[2] + K[1, 2], np.log(t[2])])
	moved = True
	while moved:
		moved = False
		for a in range(3):
			for sgn in (-1, 1):
				d = np.zeros(3)
				d[a] = sgn
				y = minimize(cost, shifted(x, d), method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-5}).x
				cy = cost(y)
				if cy < c - 1e-3:
					c, x, moved = cy, y, True

	# joint refinement with a small rotation (rotation vector in the camera frame)
	def cost6(y):
		Rm = cv2.Rodrigues(y[3:])[0] @ R
		return cost(y[:3], Rm)
	res6 = minimize(cost6, np.r_[x, 0, 0, 0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000})
	R6 = cv2.Rodrigues(res6.x[3:])[0] @ R
	return R6, t_of(res6.x[:3]), {"grid_cost": grid[0][0], "cost": float(res6.fun), "extent": float(extent)}


# --- the stage -----------------------------------------------------------------------------------------------------

class LineGeometry:
	def __init__(self, seed=0):
		self.seed = seed

	def __call__(self, view):
		rng = np.random.default_rng(self.seed)
		times = {}
		t0 = time.perf_counter()
		segs, w = segments(view.image)
		times["segments"] = time.perf_counter() - t0
		if len(segs) < 12:
			raise GeometryError(f"{len(segs)} segments")
		normals = plane_normals(view.K, segs)

		t0 = time.perf_counter()
		D, fam = axes(normals, w, rng)
		times["axes"] = time.perf_counter() - t0

		# signs: the visible face of each axis points back to the camera; the cube is roughly at the weighted centroid
		# of the family segments
		m = fam >= 0
		mid = (segs[m, :2] + segs[m, 2:]) / 2
		c = (mid * w[m, None]).sum(0) / w[m].sum()
		ray = np.linalg.inv(view.K) @ np.r_[c, 1]
		for a in range(3):
			if D[a] @ ray > 0:
				D[a] = -D[a]
		R = D.T	# columns: cube axes in the camera frame
		if np.linalg.det(R) < 0:
			R[:, [0, 1]] = R[:, [1, 0]]
			fam = np.where(fam == 0, 1, np.where(fam == 1, 0, fam))

		t0 = time.perf_counter()
		R, t, info = position(view.K, R, segs, fam, w, rng)
		times["position"] = time.perf_counter() - t0
		return {"quads": facelet_quads(view.K, R, t), "R": R, "t": t,
			"debug": {"segments": segs, "weights": w, "families": fam, "times": times, **info}}
