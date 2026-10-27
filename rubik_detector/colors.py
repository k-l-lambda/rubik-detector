"""Sticker colours: sampling, the colour feature, and the final 9-per-colour classification.

No colour scheme is assumed: the six colours are whatever the six centre stickers show, and the 54 facelets are
clustered around them (constrained to 9 each). What the feature encodes is only how photos change a colour:
shading and per-face lighting scale lightness and chroma (Lab C*) together and leave the hue angle nearly alone
(measured on the renders: red stays at hue ≈ 30° and orange ≈ 45–54° from lit to deep shade, while their a*b*
distance shrinks below the lit-shade gap). So the feature is
	[hue · w(C), log C*, L*]  — hue as a 2D unit vector scaled by HUE_SCALE · min(C*/C_SAT, 1) (an unsaturated colour
	has no reliable hue), log chroma (a shading factor becomes a small shift), lightness weighed down.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import cube


HUE_SCALE = 60.0	# feature units per radian of hue at full saturation
C_SAT = 25.0		# chroma from which the hue counts fully
LOGC_SCALE = 18.0	# feature units per e-fold of chroma
L_WEIGHT = 0.3


def to_lab(image):
	return cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def sample(lab, quad):
	"""Median Lab inside a quad (pixels), or None if it covers no pixel."""
	q = np.asarray(quad, np.float32)
	h, w = lab.shape[:2]
	x0, y0 = np.floor(q.min(0)).astype(int)
	x1, y1 = np.ceil(q.max(0)).astype(int) + 1
	x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
	if x1 - x0 < 1 or y1 - y0 < 1:
		return None
	mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
	cv2.fillConvexPoly(mask, np.round((q - [x0, y0]) * 4).astype(np.int32), 1, shift=2)
	px = lab[y0:y1, x0:x1][mask > 0]
	if len(px) < 3:
		c = q.mean(0).astype(int)
		return lab[np.clip(c[1], 0, h - 1), np.clip(c[0], 0, w - 1)].astype(float) if 0 <= c[0] < w and 0 <= c[1] < h else None
	return np.median(px, axis=0).astype(float)


def shrink(quad, factor):
	q = np.asarray(quad, float)
	c = q.mean(0)
	return c + (q - c) * factor


def weigh(lab):
	"""Colour feature (see the module doc) of one Lab colour or an (N, 3) array."""
	lab = np.asarray(lab, float)
	L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
	C = np.hypot(a, b)
	h = np.arctan2(b, a)
	sat = np.minimum(C / C_SAT, 1.0)
	return np.stack([HUE_SCALE * sat * np.cos(h), HUE_SCALE * sat * np.sin(h), LOGC_SCALE * np.log1p(C), L_WEIGHT * L], axis=-1)


# dimensions a face's lighting moves: log chroma and lightness
LIGHT_DIMS = [2, 3]


def classify(observations, iterations=6):
	"""observations: facelet → list of (view, view-frame face, Lab), all in one common cube frame. Returns 54 labels
	(face letters), exactly 9 of each, every centre carrying its own face.

	Clustering around the six centres: alternate a balanced assignment (Hungarian over 9 slots per colour, centres
	pinned) and re-estimation of the colour means and of one lighting offset per photographed face (a face of one
	photo is lit alike; the offset only touches log chroma and lightness, shrunk towards 0)."""
	keys = sorted(observations)
	obs = [(k, v, f, weigh(lab)) for k in keys for v, f, lab in observations[k]]
	faces = sorted({(v, f) for _, v, f, _ in obs})
	offset = {vf: np.zeros(4) for vf in faces}

	def corrected(v, f, x):
		return x - offset[(v, f)]

	means = np.array([np.mean([corrected(v, f, x) for k, v, f, x in obs if k == c], axis=0) for c in cube.CENTRE_FACELETS])
	labels = None
	for _ in range(iterations):
		cost = np.zeros((54, 6))
		count = np.zeros(54)
		for k, v, f, x in obs:
			cost[k] += np.linalg.norm(means - corrected(v, f, x), axis=1)
			count[k] += 1
		cost /= count[:, None]
		big = cost.max() * 10 + 1
		for c, k in enumerate(cube.CENTRE_FACELETS):
			cost[k] = big
			cost[k, c] = 0
		rows, cols = linear_sum_assignment(np.repeat(cost, 9, axis=1))
		labels = np.empty(54, int)
		labels[rows] = cols // 9

		for vf in faces:
			res = [x - means[labels[k]] for k, v, f, x in obs if (v, f) == vf]
			med = np.median(res, axis=0)
			o = np.zeros(4)
			o[LIGHT_DIMS] = med[LIGHT_DIMS] * len(res) / (len(res) + 2)
			offset[vf] = o
		means = np.array([np.mean([corrected(v, f, x) for k, v, f, x in obs if labels[k] == c], axis=0) for c in range(6)])
	return "".join(cube.FACES[c] for c in labels)


T_DOF = 4.0		# Student-t degrees of freedom: a finger, a highlight or a wrong sample stays a bounded cost
VAR_FLOOR = 4.0		# feature units², per dimension


def likelihood(observations, labels, offset_shrink=2.0, var_shrink=6.0):
	"""Per facelet and colour, the negative log-likelihood of its observations, from a model fitted to `labels`:
	every colour a mean and a diagonal variance (shrunk towards the pooled one), every photographed face a lighting
	offset on log chroma and lightness (shrunk towards 0); observations Student-t. Returns (cost (54, 6), model)."""
	obs = [(k, v, f, weigh(lab)) for k in sorted(observations) for v, f, lab in observations[k]]
	faces = sorted({(v, f) for _, v, f, _ in obs})
	offset = {vf: np.zeros(4) for vf in faces}
	X = np.array([x for _, _, _, x in obs])
	K = np.array([k for k, _, _, _ in obs])
	VF = [(v, f) for _, v, f, _ in obs]
	for _ in range(3):
		Xc = X - np.array([offset[vf] for vf in VF])
		means = np.array([Xc[labels[K] == c].mean(axis=0) for c in range(6)])
		for vf in faces:
			m = np.array([q == vf for q in VF])
			res = X[m] - means[labels[K[m]]]
			o = np.zeros(4)
			o[LIGHT_DIMS] = np.median(res, axis=0)[LIGHT_DIMS] * m.sum() / (m.sum() + offset_shrink)
			offset[vf] = o
	Xc = X - np.array([offset[vf] for vf in VF])
	means = np.array([Xc[labels[K] == c].mean(axis=0) for c in range(6)])
	pooled = np.mean([((Xc[labels[K] == c] - means[c]) ** 2).mean(axis=0) for c in range(6)], axis=0)
	var = np.array([(((Xc[labels[K] == c] - means[c]) ** 2).sum(axis=0) + var_shrink * pooled) / ((labels[K] == c).sum() + var_shrink)
		for c in range(6)])
	var = np.maximum(var, VAR_FLOOR)

	d = Xc.shape[1]
	maha = (((Xc[:, None, :] - means[None]) ** 2) / var[None]).sum(axis=2)	# (N, 6)
	nll = 0.5 * np.log(var).sum(axis=1)[None] + 0.5 * (T_DOF + d) * np.log1p(maha / T_DOF)
	cost = np.zeros((54, 6))
	np.add.at(cost, K, nll)
	return cost, {"means": means, "var": var, "offset": offset}


def posterior(cost):
	"""Per facelet independent colour probabilities from a cost (negative log-likelihood) matrix."""
	z = -(cost - cost.min(axis=1, keepdims=True))
	p = np.exp(z)
	return p / p.sum(axis=1, keepdims=True)


def _fit_cost(observations, labels):
	"""Self-consistent cost of a labelling: Σ of its own negative log-likelihood under the model fitted to it."""
	cost, _ = likelihood(observations, labels)
	return float(cost[np.arange(54), labels].sum()), cost


def classify_legal(observations, iterations=5, swap_rounds=3):
	"""The most probable legal state.

	1. start from the balanced clustering (`classify`); alternate fitting the colour model to the labels and taking
	   the legal MAP state under it (legal.best_legal) until it settles;
	2. a model fitted to wrong labels favours them (the truth can score worse under it than under its own fit), so
	   neighbours are scored by their own fit too: each of the least sure facelets is forbidden its colour, the
	   legal MAP re-solved, and the result kept if its self-consistent cost is lower.
	Returns (state, info): info has the cost, the per facelet posterior of the chosen colour (independent of
	legality: a confidence), the facelets legality moved off their independent best colour."""
	from . import legal
	labels = np.array([cube.FACES.index(c) for c in classify(observations)])
	history = []
	for _ in range(iterations):
		cost, _ = likelihood(observations, labels)
		new, total, exact = legal.best_legal(cost)
		history.append(total)
		if (new == labels).all():
			break
		labels = new
	best, cost = _fit_cost(observations, labels)
	tried = 0
	for _ in range(swap_rounds):
		improved = False
		post = posterior(cost)
		# the least sure facelets, each pushed to its runner-up colour, one at a time
		order = np.argsort(post[np.arange(54), labels])
		for k in order[:12]:
			if k in cube.CENTRE_FACELETS:
				continue
			alt = np.argsort(cost[k])
			second = alt[1] if alt[0] == labels[k] else alt[0]
			forced = cost.copy()
			forced[k, labels[k]] += 1e6
			cand, _, _ = legal.best_legal(forced)
			tried += 1
			c, cc = _fit_cost(observations, cand)
			if c < best - 1e-6:
				best, labels, cost, improved = c, cand, cc, True
				break
		if not improved:
			break
	post = posterior(cost)
	free = cost.argmin(axis=1)
	info = {"cost": best, "exact": bool(exact), "iterations": len(history), "neighbours_tried": tried,
		"min_posterior": float(post[np.arange(54), labels].min()),
		"low_confidence": [int(k) for k in np.flatnonzero(post[np.arange(54), labels] < 0.9)],
		"corrected": [int(k) for k in np.flatnonzero(free != labels)]}
	return "".join(cube.FACES[c] for c in labels), info
