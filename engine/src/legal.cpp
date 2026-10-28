// Port of rubik_detector/legal.py: legal states from the six centre colours, and the most probable legal state.
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <tuple>

#include "internal.hpp"

namespace rubik {

namespace {

struct Slots {
	std::vector<std::array<int, 3>> corners;   // facelets, reference first, one handedness
	std::vector<std::array<int, 2>> edges;
	std::vector<std::array<int, 3>> cornerColours;
	std::vector<std::array<int, 2>> edgeColours;
};

int axisOf (int k) {
	const V3& n = tables().normal[k];
	return n.x != 0 ? 0 : (n.y != 0 ? 1 : 2);
}

Slots buildSlots () {
	const CubeTables& T = tables();
	std::map<std::array<int, 3>, std::vector<int>> groups;
	for (int k = 0; k < 54; k++) {
		std::array<int, 3> p{};
		for (int i = 0; i < 3; i++) {
			double c = T.centre[k][i];
			p[i] = int(std::nearbyint(std::fabs(c) > 1.4 ? (c > 0 ? 1 : -1) : c));
		}
		groups[p].push_back(k);
	}
	Slots S;
	for (const auto& entry : groups) {
		const auto& ks = entry.second;
		if (ks.size() == 3) {
			int ref = *std::find_if(ks.begin(), ks.end(), [](int k) { return axisOf(k) == 1; });
			std::vector<int> rest;
			for (int k : ks)
				if (k != ref)
					rest.push_back(k);
			int a = rest[0], b = rest[1];
			V3 n0 = T.normal[ref], na = T.normal[a], nb = T.normal[b];
			if (dot(n0, cross(na, nb)) < 0)
				std::swap(a, b);
			S.corners.push_back({ref, a, b});
		}
		else if (ks.size() == 2) {
			auto it = std::find_if(ks.begin(), ks.end(), [](int k) { return axisOf(k) == 1; });
			if (it == ks.end())
				it = std::find_if(ks.begin(), ks.end(), [](int k) { return axisOf(k) == 2; });
			int ref = *it, other = ks[0] == ref ? ks[1] : ks[0];
			S.edges.push_back({ref, other});
		}
	}
	for (const auto& c : S.corners)
		S.cornerColours.push_back({T.faceOf[c[0]], T.faceOf[c[1]], T.faceOf[c[2]]});
	for (const auto& e : S.edges)
		S.edgeColours.push_back({T.faceOf[e[0]], T.faceOf[e[1]]});
	return S;
}

const Slots& slots () {
	static const Slots S = buildSlots();
	return S;
}

int parity (const std::vector<int>& perm) {
	int inv = 0;
	for (size_t i = 0; i < perm.size(); i++)
		for (size_t j = i + 1; j < perm.size(); j++)
			inv += perm[i] > perm[j];
	return inv % 2;
}

// orientations minimizing Σ costs[i][o_i] with Σ o_i ≡ 0 (mod)
double orient (const std::vector<std::vector<double>>& costs, int mod, std::vector<int>& orientation) {
	const size_t n = costs.size();
	const double INF = std::numeric_limits<double>::infinity();
	std::vector<std::vector<double>> best(n + 1, std::vector<double>(mod, INF));
	std::vector<std::vector<int>> back(n, std::vector<int>(mod, 0));
	best[0][0] = 0;
	for (size_t i = 0; i < n; i++)
		for (int r = 0; r < mod; r++)
			for (int o = 0; o < mod; o++) {
				double v = best[i][((r - o) % mod + mod) % mod] + costs[i][o];
				if (v < best[i + 1][r]) {
					best[i + 1][r] = v;
					back[i][r] = o;
				}
			}
	orientation.assign(n, 0);
	int r = 0;
	for (size_t j = 0; j < n; j++) {
		size_t i = n - 1 - j;
		int o = back[i][r];
		orientation[i] = o;
		r = ((r - o) % mod + mod) % mod;
	}
	return best[n][0];
}

// Murty's k-best assignments, in nondecreasing cost; calls visit(total, perm) until it returns false or `limit`
template <typename Visit>
int kbest (const std::vector<std::vector<double>>& C, int limit, Visit&& visit) {
	const int n = int(C.size());
	const double BIG = 1e12;
	using Pairs = std::vector<std::pair<int, int>>;
	auto solve = [&](const Pairs& fixed, const Pairs& forbidden, double& total, std::vector<int>& perm) {
		auto M = C;
		for (auto [r, c] : forbidden)
			M[r][c] = BIG;
		for (auto [r, c] : fixed) {
			double v = C[r][c];
			for (int j = 0; j < n; j++)
				M[r][j] = BIG;
			for (int i = 0; i < n; i++)
				M[i][c] = BIG;
			M[r][c] = v;
		}
		perm = assignment(M);
		total = 0;
		for (int r = 0; r < n; r++)
			total += M[r][perm[r]];
		return total < BIG / 2;
	};
	struct Node {
		double total;
		long counter;
		std::vector<int> perm;
		Pairs fixed, forbidden;
		bool operator> (const Node& o) const { return std::tie(total, counter) > std::tie(o.total, o.counter); }
	};
	std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;
	double total;
	std::vector<int> perm;
	if (!solve({}, {}, total, perm))
		return 0;
	heap.push({total, 0, perm, {}, {}});
	long counter = 1;
	int count = 0;
	for (int it = 0; it < limit; it++) {
		if (heap.empty())
			return count;
		Node node = heap.top();
		heap.pop();
		count++;
		if (!visit(node.total, node.perm))
			return -count;   // stopped by the visitor
		Pairs f = node.fixed;
		std::vector<bool> fixedRow(n, false);
		for (auto [r, c] : node.fixed)
			fixedRow[r] = true;
		for (int r = 0; r < n; r++) {
			if (fixedRow[r])
				continue;
			Pairs sub = node.forbidden;
			sub.push_back({r, node.perm[r]});
			if (solve(f, sub, total, perm))
				heap.push({total, counter++, perm, f, sub});
			f.push_back({r, node.perm[r]});
		}
	}
	return count;
}

}

std::vector<int> assignment (const std::vector<std::vector<double>>& C) {
	// Hungarian method (shortest augmenting paths, potentials), square matrix
	const int n = int(C.size());
	const double INF = std::numeric_limits<double>::infinity();
	std::vector<double> u(n + 1, 0), v(n + 1, 0);
	std::vector<int> p(n + 1, 0), way(n + 1, 0);
	for (int i = 1; i <= n; i++) {
		p[0] = i;
		int j0 = 0;
		std::vector<double> minv(n + 1, INF);
		std::vector<bool> used(n + 1, false);
		do {
			used[j0] = true;
			int i0 = p[j0], j1 = 0;
			double delta = INF;
			for (int j = 1; j <= n; j++) {
				if (used[j])
					continue;
				double cur = C[i0 - 1][j - 1] - u[i0] - v[j];
				if (cur < minv[j]) {
					minv[j] = cur;
					way[j] = j0;
				}
				if (minv[j] < delta) {
					delta = minv[j];
					j1 = j;
				}
			}
			for (int j = 0; j <= n; j++) {
				if (used[j]) {
					u[p[j]] += delta;
					v[j] -= delta;
				}
				else
					minv[j] -= delta;
			}
			j0 = j1;
		} while (p[j0] != 0);
		do {
			int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		} while (j0);
	}
	std::vector<int> perm(n);
	for (int j = 1; j <= n; j++)
		perm[p[j] - 1] = j - 1;
	return perm;
}

LegalResult bestLegal (const std::array<std::array<double, 6>, 54>& cost, int limit) {
	const Slots& S = slots();
	bool exact = true;
	struct Best {
		double cost = std::numeric_limits<double>::infinity();
		std::vector<int> perm, orientation;
	};
	auto side = [&](auto slotList, auto colours, int n) {
		const int m = int(slotList.size());
		// T[s][c][o]: cubie c in slot s with orientation o (slot facelet i gets cubie colour (i - o) mod n)
		std::vector<std::vector<std::vector<double>>> T(m, std::vector<std::vector<double>>(m, std::vector<double>(n, 0)));
		for (int s = 0; s < m; s++)
			for (int c = 0; c < m; c++)
				for (int o = 0; o < n; o++)
					for (int i = 0; i < n; i++)
						T[s][c][o] += cost[slotList[s][i]][colours[c][((i - o) % n + n) % n]];
		std::vector<std::vector<double>> relaxed(m, std::vector<double>(m));
		for (int s = 0; s < m; s++)
			for (int c = 0; c < m; c++)
				relaxed[s][c] = *std::min_element(T[s][c].begin(), T[s][c].end());
		std::array<Best, 2> best;
		int count = kbest(relaxed, limit, [&](double rc, const std::vector<int>& perm) {
			if (rc >= best[0].cost && rc >= best[1].cost)
				return false;
			std::vector<std::vector<double>> oc(m);
			for (int s = 0; s < m; s++)
				oc[s] = T[s][perm[s]];
			std::vector<int> orientation;
			double c = orient(oc, n, orientation);
			int p = parity(perm);
			if (c < best[p].cost)
				best[p] = {c, perm, orientation};
			return true;
		});
		if (count >= 0)   // not stopped by the bound: the enumeration ran out (complete) or hit the limit
			exact = exact && count < limit;
		return best;
	};
	auto bc = side(S.corners, S.cornerColours, 3);
	auto be = side(S.edges, S.edgeColours, 2);
	int p = bc[0].cost + be[0].cost <= bc[1].cost + be[1].cost ? 0 : 1;
	LegalResult out{};
	for (int f = 0; f < 6; f++)
		out.labels[CENTRE_FACELETS[f]] = f;
	for (int s = 0; s < int(S.corners.size()); s++)
		for (int i = 0; i < 3; i++)
			out.labels[S.corners[s][i]] = S.cornerColours[bc[p].perm[s]][((i - bc[p].orientation[s]) % 3 + 3) % 3];
	for (int s = 0; s < int(S.edges.size()); s++)
		for (int i = 0; i < 2; i++)
			out.labels[S.edges[s][i]] = S.edgeColours[be[p].perm[s]][((i - be[p].orientation[s]) % 2 + 2) % 2];
	out.total = 0;
	for (int k = 0; k < 54; k++)
		out.total += cost[k][out.labels[k]];
	out.exact = exact;
	return out;
}

bool isLegal (const std::array<int, 54>& labels) {
	const Slots& S = slots();
	for (int f = 0; f < 6; f++)
		if (labels[CENTRE_FACELETS[f]] != f)
			return false;
	auto decompose = [&](const auto& slotList, const auto& colours, int n, std::vector<int>& perm, std::vector<int>& ori) {
		for (const auto& fs : slotList) {
			bool found = false;
			for (int c = 0; c < int(colours.size()) && !found; c++)
				for (int o = 0; o < n && !found; o++) {
					bool ok = true;
					for (int i = 0; i < n; i++)
						ok = ok && labels[fs[i]] == colours[c][((i - o) % n + n) % n];
					if (ok) {
						perm.push_back(c);
						ori.push_back(o);
						found = true;
					}
				}
			if (!found)
				return false;
		}
		return true;
	};
	std::vector<int> cp, co, ep, eo;
	if (!decompose(S.corners, S.cornerColours, 3, cp, co) || !decompose(S.edges, S.edgeColours, 2, ep, eo))
		return false;
	auto isPerm = [](std::vector<int> p) {
		std::sort(p.begin(), p.end());
		for (int i = 0; i < int(p.size()); i++)
			if (p[i] != i)
				return false;
		return true;
	};
	int twist = 0, flip = 0;
	for (int o : co)
		twist += o;
	for (int o : eo)
		flip += o;
	return isPerm(cp) && isPerm(ep) && twist % 3 == 0 && flip % 2 == 0 && parity(cp) == parity(ep);
}

}
