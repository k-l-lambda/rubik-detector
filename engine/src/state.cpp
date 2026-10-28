// Evidence across views (port of rubik_detector/fusion.py, extended to any number of views and known rotations),
// state estimation (colors.classify_legal) and scoring of candidate states.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "internal.hpp"

namespace rubik {

// --- rotations ---------------------------------------------------------------------------------------------------

namespace {

int rotationIndex (const std::array<int, 9>& m) {
	for (int g = 0; g < 24; g++)
		if (tables().rotations[g] == m)
			return g;
	throw std::logic_error("not a cube rotation");
}

}

int composeRotations (int a, int b) {
	const auto& A = tables().rotations.at(a);
	const auto& B = tables().rotations.at(b);
	std::array<int, 9> c{};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				c[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
	return rotationIndex(c);
}

int inverseRotation (int g) {
	const auto& m = tables().rotations.at(g);
	return rotationIndex({m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]});
}

int relativeRotation (const Pose& from, const Pose& to) {
	// a point at position p in `to`'s frame is at g·p in `from`'s frame: R_from·g ≈ R_to, so g ≈ R_fromᵀ·R_to
	M3 F, T;
	std::copy(from.R.begin(), from.R.end(), F.begin());
	std::copy(to.R.begin(), to.R.end(), T.begin());
	M3 Ft = {F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8]};
	M3 G = matmul(Ft, T);
	int best = 0;
	double bestScore = -1e300;
	for (int g = 0; g < 24; g++) {
		double s = 0;
		for (int i = 0; i < 9; i++)
			s += tables().rotations[g][i] * G[i];
		if (s > bestScore) {
			bestScore = s;
			best = g;
		}
	}
	return best;
}

// --- evidence ----------------------------------------------------------------------------------------------------

int Evidence::addView (const std::vector<FaceletSample>& samples, int rotation) {
	if (rotation >= 24)
		throw std::invalid_argument("rotation out of range");
	list.push_back({nextId, samples, rotation, rotation >= 0});
	return nextId++;
}

void Evidence::removeView (int id) {
	list.erase(std::remove_if(list.begin(), list.end(), [id](const View& v) { return v.id == id; }), list.end());
}

void Evidence::clear () {
	list.clear();
}

int Evidence::views () const { return int(list.size()); }

int Evidence::rotation (int view) const {
	for (const View& v : list)
		if (v.id == view)
			return v.rotation;
	return -1;
}

bool Evidence::complete () const {
	auto obs = observations();
	return std::all_of(obs.begin(), obs.end(), [](const auto& o) { return !o.empty(); });
}

std::vector<std::vector<Observation>> Evidence::observations () const {
	std::vector<std::vector<Observation>> out(54);
	for (const View& v : list) {
		if (v.rotation < 0)
			continue;
		for (const FaceletSample& s : v.samples)
			if (s.valid)
				out[tables().perms[v.rotation][s.facelet]].push_back({v.id, tables().faceOf[s.facelet], s.lab});
	}
	return out;
}

bool Evidence::fuse (double* costOut, bool requireAllFaces) {
	const CubeTables& T = tables();
	if (list.empty())
		return false;
	// per view: features with each face's mean taken out (the face's pattern survives any lighting)
	struct Feat {
		std::map<int, std::array<double, 4>> f;   // view-frame facelet → feature
		std::set<int> faces;
	};
	std::vector<Feat> feats(list.size());
	for (size_t v = 0; v < list.size(); v++) {
		std::map<int, std::pair<std::array<double, 4>, int>> faceSum;
		for (const FaceletSample& s : list[v].samples) {
			if (!s.valid)
				continue;
			auto x = weigh(s.lab.data());
			feats[v].f[s.facelet] = x;
			auto& fs = faceSum[T.faceOf[s.facelet]];
			for (int d = 0; d < 4; d++)
				fs.first[d] += x[d];
			fs.second++;
		}
		for (auto& [k, x] : feats[v].f) {
			const auto& fs = faceSum[T.faceOf[k]];
			for (int d = 0; d < 4; d++)
				x[d] -= fs.first[d] / fs.second;
		}
		for (const auto& entry : faceSum)
			feats[v].faces.insert(entry.first);
	}
	// the common frame: views with a given rotation, else the first one
	std::vector<int> rot(list.size(), -1), todo;
	for (size_t v = 0; v < list.size(); v++)
		if (list[v].fixed)
			rot[v] = list[v].rotation;
	if (std::none_of(rot.begin(), rot.end(), [](int g) { return g >= 0; }))
		rot[0] = T.identity;
	for (size_t v = 0; v < list.size(); v++)
		if (rot[v] < 0)
			todo.push_back(int(v));

	auto mapped = [&](int v, int g) {
		std::map<int, const std::array<double, 4>*> m;
		for (const auto& [k, x] : feats[v].f)
			m[T.perms[g][k]] = &x;
		return m;
	};
	auto covered = [&](const std::vector<int>& r) {
		std::set<int> faces;
		for (size_t v = 0; v < r.size(); v++)
			if (r[v] >= 0)
				for (int f : feats[v].faces)
					faces.insert(T.faceOf[T.perms[r[v]][9 * f + 4]]);
		return faces.size();
	};
	// mean distance over overlapping facelets of every pair of placed views
	auto overlap = [&](const std::vector<int>& r, int& n) {
		std::vector<std::map<int, const std::array<double, 4>*>> maps;
		for (size_t v = 0; v < r.size(); v++)
			if (r[v] >= 0)
				maps.push_back(mapped(int(v), r[v]));
		double total = 0;
		n = 0;
		for (size_t i = 0; i < maps.size(); i++)
			for (size_t j = i + 1; j < maps.size(); j++)
				for (const auto& [k, x] : maps[i]) {
					auto it = maps[j].find(k);
					if (it == maps[j].end())
						continue;
					double s = 0;
					for (int d = 0; d < 4; d++)
						s += ((*x)[d] - (*it->second)[d]) * ((*x)[d] - (*it->second)[d]);
					total += std::sqrt(s);
					n++;
				}
		return n ? total / n : std::numeric_limits<double>::infinity();
	};

	double bestCost = std::numeric_limits<double>::infinity();
	std::vector<int> best;
	if (todo.empty()) {
		// nothing to place (one view, or all rotations given): only the coverage can fail
		if (requireAllFaces && covered(rot) < 6)
			return false;
		int n;
		double cost = overlap(rot, n);
		for (size_t v = 0; v < list.size(); v++)
			list[v].rotation = rot[v];
		if (costOut)
			*costOut = n ? cost : 0.0;
		return true;
	}
	if (todo.size() <= 2) {
		// every combination (itertools.product order: the first view's rotation outermost)
		int combos = 1;
		for (size_t i = 0; i < todo.size(); i++)
			combos *= 24;
		for (int c = 0; c < combos; c++) {
			std::vector<int> r = rot;
			int code = c;
			for (size_t i = todo.size(); i-- > 0;) {
				r[todo[i]] = code % 24;
				code /= 24;
			}
			if (requireAllFaces && covered(r) < 6)
				continue;
			int n;
			double cost = overlap(r, n);
			if (n >= 9 && cost < bestCost) {
				bestCost = cost;
				best = r;
			}
		}
	}
	else {
		std::vector<int> r = rot;
		for (int v : todo) {
			double vb = std::numeric_limits<double>::infinity();
			int gb = -1;
			for (int g = 0; g < 24; g++) {
				r[v] = g;
				int n;
				double cost = overlap(r, n);
				if (n >= 9 && cost < vb) {
					vb = cost;
					gb = g;
				}
			}
			r[v] = gb;
			if (gb < 0)
				return false;
		}
		if (!requireAllFaces || covered(r) == 6) {
			int n;
			bestCost = overlap(r, n);
			best = r;
		}
	}
	if (best.empty())
		return false;
	for (size_t v = 0; v < list.size(); v++)
		list[v].rotation = best[v];
	if (costOut)
		*costOut = bestCost;
	return true;
}

// --- state -------------------------------------------------------------------------------------------------------

namespace {

std::vector<Obs> toObs (const Evidence& evidence) {
	std::vector<Obs> obs;
	auto all = evidence.observations();
	for (int k = 0; k < 54; k++)
		for (const Observation& o : all[k])
			obs.push_back({k, o.view, o.viewFace, weigh(o.lab.data()), 1.0});
	return obs;
}

double selfCost (const Likelihood& l, const std::array<int, 54>& labels) {
	double s = 0;
	for (int k = 0; k < 54; k++)
		s += l.cost[k][labels[k]];
	return s;
}

}

StateEstimate estimateState (const Evidence& evidence, const EstimateOptions& options) {
	StateEstimate out;
	if (!evidence.complete()) {
		out.error = "the placed views don't cover all 54 facelets";
		return out;
	}
	std::vector<Obs> obs = toObs(evidence);
	std::array<int, 54> labels = classifyBalanced(obs);
	Likelihood fit;
	bool exact = true;
	if (options.legal) {
		// alternate: model fitted to the labels, legal MAP state under it, until it settles
		for (int it = 0; it < options.iterations; it++) {
			Likelihood l = likelihood(obs, labels);
			LegalResult r = bestLegal(l.cost);
			exact = r.exact;
			if (r.labels == labels)
				break;
			labels = r.labels;
		}
		fit = likelihood(obs, labels);
		double best = selfCost(fit, labels);
		// a model fitted to wrong labels favours them: the least sure facelets are each forbidden their colour and
		// the result kept if its own fit is better
		for (int round = 0; round < options.swapRounds; round++) {
			bool improved = false;
			auto post = posterior(fit.cost);
			std::vector<int> order(54);
			for (int k = 0; k < 54; k++)
				order[k] = k;
			std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return post[a][labels[a]] < post[b][labels[b]]; });
			for (int i = 0; i < 12; i++) {
				int k = order[i];
				if (std::find(std::begin(CENTRE_FACELETS), std::end(CENTRE_FACELETS), k) != std::end(CENTRE_FACELETS))
					continue;
				auto forced = fit.cost;
				forced[k][labels[k]] += 1e6;
				LegalResult cand = bestLegal(forced);
				Likelihood l = likelihood(obs, cand.labels);
				double c = selfCost(l, cand.labels);
				if (c < best - 1e-6) {
					best = c;
					labels = cand.labels;
					fit = l;
					improved = true;
					break;
				}
			}
			if (!improved)
				break;
		}
		out.cost = best;
	}
	else {
		fit = likelihood(obs, labels);
		out.cost = selfCost(fit, labels);
	}
	auto post = posterior(fit.cost);
	out.state.resize(54);
	out.minPosterior = 1;
	for (int k = 0; k < 54; k++) {
		out.state[k] = FACES[labels[k]];
		out.posterior[k] = post[k][labels[k]];
		out.minPosterior = std::min(out.minPosterior, out.posterior[k]);
		int free = int(std::min_element(fit.cost[k].begin(), fit.cost[k].end()) - fit.cost[k].begin());
		if (free != labels[k])
			out.corrected.push_back(k);
	}
	out.nll = fit.cost;
	out.model = fit.model;
	out.exact = exact;
	out.legal = isLegal(labels);
	out.ok = true;
	return out;
}

std::vector<double> scoreStates (const ColourModel& model, const std::vector<FaceletSample>& samples, int rotation,
	const std::vector<std::string>& states) {
	const CubeTables& T = tables();
	const double T_DOF = 4.0, OFFSET_SHRINK = 2.0;
	std::array<double, 6> logdet{};
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			logdet[c] += std::log(model.var[c][d]);
	std::vector<std::pair<int, std::array<double, 4>>> xs;   // (view facelet, feature)
	for (const FaceletSample& s : samples)
		if (s.valid)
			xs.push_back({s.facelet, weigh(s.lab.data())});
	std::vector<double> out;
	for (const std::string& state : states) {
		if (state.size() != 54)
			throw std::invalid_argument("a state has 54 facelets");
		std::vector<int> colour(xs.size());
		for (size_t i = 0; i < xs.size(); i++) {
			const char* p = std::strchr(FACES, state[T.perms[rotation][xs[i].first]]);
			if (!p)
				throw std::invalid_argument("state letters must be L R D U B F");
			colour[i] = int(p - FACES);
		}
		// a lighting offset per face of this view, fitted under this candidate
		std::map<int, std::array<double, 4>> offset;
		for (int f = 0; f < 6; f++) {
			std::vector<double> res[2];
			for (size_t i = 0; i < xs.size(); i++)
				if (T.faceOf[xs[i].first] == f)
					for (int j = 0; j < 2; j++)
						res[j].push_back(xs[i].second[2 + j] - model.mean[colour[i]][2 + j]);
			if (res[0].empty())
				continue;
			std::array<double, 4> o{0, 0, 0, 0};
			double n = double(res[0].size());
			for (int j = 0; j < 2; j++) {
				std::vector<double>& v = res[j];
				std::sort(v.begin(), v.end());
				double med = v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
				o[2 + j] = med * n / (n + OFFSET_SHRINK);
			}
			offset[f] = o;
		}
		double total = 0;
		for (size_t i = 0; i < xs.size(); i++) {
			int c = colour[i];
			const auto& o = offset[T.faceOf[xs[i].first]];
			double maha = 0;
			for (int d = 0; d < 4; d++) {
				double r = xs[i].second[d] - o[d] - model.mean[c][d];
				maha += r * r / model.var[c][d];
			}
			total += 0.5 * logdet[c] + 0.5 * (T_DOF + 4) * std::log1p(maha / T_DOF);
		}
		out.push_back(total);
	}
	return out;
}

}
