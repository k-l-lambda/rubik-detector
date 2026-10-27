"""Legal cube states: which cubies a 3×3 can have given its six centre colours, and the most probable legal state.

No colour scheme is assumed: which colours share a cubie follows from the centres (colour c is the colour of face c's
centre; a corner cubie carries the colours of the three faces around its home corner in their clockwise order, an edge
the colours of its two faces). A state is legal when
	every corner and edge slot holds one of these cubies, each cubie exactly once;
	corner twists sum to 0 mod 3, edge flips to 0 mod 2, corner and edge permutations have equal parity.
Any consistent choice of reference stickers works for the twist and flip sums: moving a slot's reference changes the
twist of the slot's occupant and of the slot's own cubie by opposite amounts.

best_legal(cost) minimizes Σ cost[facelet, colour] over legal states, exactly: corner and edge permutations are
enumerated in order of their orientation-relaxed cost (Murty's k-best assignments), each one's orientations solved
under the twist / flip constraint by a small DP, until the relaxed cost passes the best found for both parities;
then the parities are matched.
"""
import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import cube


def _slots():
	pos = np.rint(np.where(np.abs(cube.CENTRES) > 1.4, np.sign(cube.CENTRES), cube.CENTRES)).astype(int)
	groups = {}
	for k, p in enumerate(map(tuple, pos)):
		groups.setdefault(p, []).append(k)
	axis = lambda k: int(np.flatnonzero(cube.FACE_NORMALS[k])[0])
	corners, edges = [], []
	for p, ks in sorted(groups.items()):
		if len(ks) == 3:
			ref = next(k for k in ks if axis(k) == 1)
			a, b = [k for k in ks if k != ref]
			if np.linalg.det(np.array([cube.FACE_NORMALS[ref], cube.FACE_NORMALS[a], cube.FACE_NORMALS[b]])) < 0:
				a, b = b, a
			corners.append((ref, a, b))
		elif len(ks) == 2:
			ref = next((k for k in ks if axis(k) == 1), None)
			if ref is None:
				ref = next(k for k in ks if axis(k) == 2)
			edges.append((ref, next(k for k in ks if k != ref)))
	return np.array(corners), np.array(edges)


CORNERS, EDGES = _slots()		# (8, 3), (12, 2) facelets, reference first, corners in one handedness
CORNER_COLOURS = cube.FACE_OF[CORNERS]	# colours of each cubie in its solved order (cubie i = home slot i)
EDGE_COLOURS = cube.FACE_OF[EDGES]


def parity(perm):
	perm = list(perm)
	return sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm))) % 2


def to_labels(state):
	return np.array([cube.FACES.index(c) for c in state])


def decompose(labels):
	"""(corner perm, twists, edge perm, flips) of a state given as 54 colour indices, or None if some slot holds no
	possible cubie."""
	labels = np.asarray(labels)
	out = []
	for slots, cubies, n in ((CORNERS, CORNER_COLOURS, 3), (EDGES, EDGE_COLOURS, 2)):
		perm, orient = [], []
		for fs in slots:
			col = labels[fs]
			hit = [(c, o) for c, cc in enumerate(cubies) for o in range(n) if all(col[i] == cc[(i - o) % n] for i in range(n))]
			if not hit:
				return None
			perm.append(hit[0][0])
			orient.append(hit[0][1])
		out += [perm, orient]
	return tuple(out)


def is_legal(labels):
	d = decompose(labels)
	if d is None:
		return False
	cp, co, ep, eo = d
	return (sorted(cp) == list(range(8)) and sorted(ep) == list(range(12)) and sum(co) % 3 == 0 and sum(eo) % 2 == 0
		and parity(cp) == parity(ep) and all(labels[c] == f for f, c in enumerate(cube.CENTRE_FACELETS)))


def _placement_costs(cost, slots, cubies, n):
	"""T[s, c, o]: cost of cubie c in slot s with orientation o (slot facelet i gets cubie colour (i - o) mod n)."""
	T = np.zeros((len(slots), len(cubies), n))
	for o in range(n):
		for i in range(n):
			T[:, :, o] += cost[slots[:, i]][:, cubies[:, (i - o) % n]]
	return T


def _orient(costs, mod):
	"""Orientations minimizing Σ costs[i, o_i] with Σ o_i ≡ 0 (mod): (total, orientations)."""
	n = len(costs)
	best = np.full((n + 1, mod), np.inf)
	back = np.zeros((n, mod), int)
	best[0, 0] = 0
	for i in range(n):
		for r in range(mod):
			for o in range(mod):
				v = best[i, (r - o) % mod] + costs[i, o]
				if v < best[i + 1, r]:
					best[i + 1, r] = v
					back[i, r] = o
	orient, r = [], 0
	for i in reversed(range(n)):
		o = back[i, r]
		orient.append(o)
		r = (r - o) % mod
	return float(best[n, 0]), orient[::-1]


def _kbest(C, limit):
	"""Assignments (row r → column perm[r]) in nondecreasing cost (Murty)."""
	n = len(C)
	BIG = 1e12

	def solve(fixed, forbidden):
		M = C.copy()
		for r, c in forbidden:
			M[r, c] = BIG
		for r, c in fixed:
			v = C[r, c]
			M[r, :] = BIG
			M[:, c] = BIG
			M[r, c] = v
		rows, cols = linear_sum_assignment(M)
		total = M[rows, cols].sum()
		return (total, cols) if total < BIG / 2 else None

	first = solve((), ())
	heap, counter = [(first[0], 0, tuple(first[1]), (), ())], 1
	for _ in range(limit):
		if not heap:
			return
		total, _, perm, fixed, forbidden = heapq.heappop(heap)
		yield total, np.array(perm)
		f = list(fixed)
		fixed_rows = {r for r, _ in fixed}
		for r in range(n):
			if r in fixed_rows:
				continue
			sub_forbidden = forbidden + ((r, perm[r]),)
			sub = solve(tuple(f), sub_forbidden)
			if sub is not None:
				heapq.heappush(heap, (sub[0], counter, tuple(sub[1]), tuple(f), sub_forbidden))
				counter += 1
			f.append((r, perm[r]))


def best_legal(cost, limit=20000):
	"""The legal state of least Σ cost[facelet, colour] (cost: (54, 6), e.g. negative log-likelihoods); centres are
	pinned to their own colour. Returns (labels, total, exact) — exact is False if an enumeration hit `limit`."""
	cost = np.asarray(cost, float)
	exact = True

	def side(slots, cubies, n):
		nonlocal exact
		T = _placement_costs(cost, slots, cubies, n)
		relaxed = T.min(axis=2)
		best = {0: (np.inf, None), 1: (np.inf, None)}
		count = 0
		for rc, perm in _kbest(relaxed, limit):
			count += 1
			if rc >= best[0][0] and rc >= best[1][0]:
				break
			c, orient = _orient(T[np.arange(len(slots)), perm], n)
			p = parity(perm)
			if c < best[p][0]:
				best[p] = (c, (perm, orient))
		else:
			exact = exact and count < limit
		return best

	bc = side(CORNERS, CORNER_COLOURS, 3)
	be = side(EDGES, EDGE_COLOURS, 2)
	p = min((0, 1), key=lambda p: bc[p][0] + be[p][0])
	labels = np.empty(54, int)
	for f, k in enumerate(cube.CENTRE_FACELETS):
		labels[k] = f
	for (_, (perm, orient)), slots, cubies, n in ((bc[p], CORNERS, CORNER_COLOURS, 3), (be[p], EDGES, EDGE_COLOURS, 2)):
		for s, (c, o) in enumerate(zip(perm, orient)):
			for i in range(n):
				labels[slots[s, i]] = cubies[c, (i - o) % n]
	total = float(cost[np.arange(54), labels].sum())
	return labels, total, exact
