// Port of the classification parts of rubik_detector/colors.py: balanced clustering, the colour likelihood model.
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "internal.hpp"

namespace rubik {

namespace {

const int LIGHT_DIMS[2] = {2, 3};   // log chroma and lightness: what a face's lighting moves
const double T_DOF = 4.0, VAR_FLOOR = 4.0;

double median (std::vector<double> v) {
	size_t n = v.size(), h = n / 2;
	std::nth_element(v.begin(), v.begin() + h, v.end());
	double hi = v[h];
	if (n % 2)
		return hi;
	double lo = *std::max_element(v.begin(), v.begin() + h);
	return (lo + hi) / 2;
}

using Group = std::pair<int, int>;   // (view, face)

}

namespace {

// observations grouped by (view, face): the lighting groups
struct Groups {
	std::vector<int> of;                       // per observation its group
	std::vector<std::vector<int>> members;
	explicit Groups (const std::vector<Obs>& obs) {
		std::map<Group, int> index;
		for (const Obs& o : obs) {
			auto it = index.emplace(Group{o.view, o.face}, int(members.size())).first;
			if (it->second == int(members.size()))
				members.emplace_back();
			members[it->second].push_back(int(of.size()));
			of.push_back(it->second);
		}
	}
	// per group: median residual on log chroma and lightness, shrunk towards 0
	template <class Residual>
	std::vector<std::array<double, 4>> offsets (Residual residual, double shrink) const {
		std::vector<std::array<double, 4>> out(members.size(), {0, 0, 0, 0});
		for (size_t g = 0; g < members.size(); g++) {
			double n = double(members[g].size());
			for (int d : LIGHT_DIMS) {
				std::vector<double> r;
				for (int i : members[g])
					r.push_back(residual(i, d));
				out[g][d] = median(r) * n / (n + shrink);
			}
		}
		return out;
	}
};

}

std::array<int, 54> classifyBalanced (const std::vector<Obs>& obs, int iterations) {
	const size_t N = obs.size();
	Groups groups(obs);
	std::vector<std::array<double, 4>> offset(groups.members.size(), {0, 0, 0, 0});
	auto corrected = [&](size_t i) {
		const auto& off = offset[groups.of[i]];
		return std::array<double, 4>{obs[i].x[0] - off[0], obs[i].x[1] - off[1], obs[i].x[2] - off[2], obs[i].x[3] - off[3]};
	};
	std::array<bool, 54> observed{};
	for (const Obs& o : obs)
		observed[o.facelet] = true;
	// seeds: the centres' colours; a colour whose centre is not observed (partial evidence) by the observation
	// farthest from the seeds so far
	std::array<std::array<double, 4>, 6> means{};
	std::array<bool, 6> seeded{};
	for (int c = 0; c < 6; c++) {
		double w = 0;
		for (size_t i = 0; i < N; i++)
			if (obs[i].facelet == CENTRE_FACELETS[c]) {
				auto x = corrected(i);
				for (int d = 0; d < 4; d++)
					means[c][d] += obs[i].weight * x[d];
				w += obs[i].weight;
			}
		if (w > 0) {
			for (int d = 0; d < 4; d++)
				means[c][d] /= w;
			seeded[c] = true;
		}
	}
	for (int c = 0; c < 6; c++) {
		if (seeded[c])
			continue;
		double far = -1;
		for (size_t i = 0; i < N; i++) {
			double near = std::numeric_limits<double>::infinity();
			for (int j = 0; j < 6; j++)
				if (seeded[j]) {
					double s = 0;
					for (int d = 0; d < 4; d++)
						s += (means[j][d] - obs[i].x[d]) * (means[j][d] - obs[i].x[d]);
					near = std::min(near, s);
				}
			if (near > far) {
				far = near;
				means[c] = obs[i].x;
			}
		}
		seeded[c] = true;
	}
	std::array<int, 54> labels{};
	for (int it = 0; it < iterations; it++) {
		std::vector<std::vector<double>> cost(54, std::vector<double>(6, 0));
		std::array<double, 54> weight{};
		for (size_t i = 0; i < N; i++) {
			auto x = corrected(i);
			for (int c = 0; c < 6; c++) {
				double s = 0;
				for (int d = 0; d < 4; d++)
					s += (means[c][d] - x[d]) * (means[c][d] - x[d]);
				cost[obs[i].facelet][c] += obs[i].weight * std::sqrt(s);
			}
			weight[obs[i].facelet] += obs[i].weight;
		}
		double maxCost = 0;
		for (int k = 0; k < 54; k++)
			for (int c = 0; c < 6; c++) {
				if (weight[k] > 0)
					cost[k][c] /= weight[k];
				maxCost = std::max(maxCost, cost[k][c]);
			}
		double big = maxCost * 10 + 1;
		for (int c = 0; c < 6; c++)
			for (int j = 0; j < 6; j++)
				cost[CENTRE_FACELETS[c]][j] = j == c ? 0 : big;
		// nine slots per colour; facelets not observed cost nothing anywhere (they take the slots left over)
		std::vector<std::vector<double>> C(54, std::vector<double>(54));
		for (int k = 0; k < 54; k++)
			for (int j = 0; j < 54; j++)
				C[k][j] = observed[k] || k == CENTRE_FACELETS[k / 9] ? cost[k][j / 9] : 0;
		std::vector<int> perm = assignment(C);
		for (int k = 0; k < 54; k++)
			labels[k] = perm[k] / 9;
		offset = groups.offsets([&](int i, int d) { return obs[i].x[d] - means[labels[obs[i].facelet]][d]; }, 2);
		for (int c = 0; c < 6; c++) {
			std::array<double, 4> sum{};
			double w = 0;
			for (size_t i = 0; i < N; i++)
				if (labels[obs[i].facelet] == c) {
					auto x = corrected(i);
					for (int d = 0; d < 4; d++)
						sum[d] += obs[i].weight * x[d];
					w += obs[i].weight;
				}
			if (w > 0)
				for (int d = 0; d < 4; d++)
					means[c][d] = sum[d] / w;
		}
	}
	return labels;
}

Likelihood likelihood (const std::vector<Obs>& obs, const std::array<int, 54>& labels, double outlier) {
	const double OFFSET_SHRINK = 2.0, VAR_SHRINK = 6.0;
	const size_t N = obs.size();
	Groups groups(obs);
	std::vector<std::array<double, 4>> offset(groups.members.size(), {0, 0, 0, 0});
	std::vector<std::array<double, 4>> Xc(N);
	std::array<std::array<double, 4>, 6> means{};
	std::array<double, 6> counts{};
	auto correct = [&]() {
		for (size_t i = 0; i < N; i++) {
			const auto& off = offset[groups.of[i]];
			for (int d = 0; d < 4; d++)
				Xc[i][d] = obs[i].x[d] - off[d];
		}
	};
	auto fitMeans = [&]() {
		for (int c = 0; c < 6; c++) {
			std::array<double, 4> sum{};
			double w = 0;
			for (size_t i = 0; i < N; i++)
				if (labels[obs[i].facelet] == c) {
					for (int d = 0; d < 4; d++)
						sum[d] += obs[i].weight * Xc[i][d];
					w += obs[i].weight;
				}
			counts[c] = w;
			if (w > 0)
				for (int d = 0; d < 4; d++)
					means[c][d] = sum[d] / w;
		}
	};
	for (int it = 0; it < 3; it++) {
		correct();
		fitMeans();
		offset = groups.offsets([&](int i, int d) { return obs[i].x[d] - means[labels[obs[i].facelet]][d]; }, OFFSET_SHRINK);
	}
	correct();
	fitMeans();
	// a colour with no observation (partial evidence: all its facelets hidden) has no model: it costs every
	// observed facelet more than its worst modelled colour
	std::array<double, 4> pooled{};
	std::array<std::array<double, 4>, 6> ss{};
	int modelled = 0;
	for (size_t i = 0; i < N; i++) {
		int c = labels[obs[i].facelet];
		for (int d = 0; d < 4; d++)
			ss[c][d] += obs[i].weight * (Xc[i][d] - means[c][d]) * (Xc[i][d] - means[c][d]);
	}
	for (int c = 0; c < 6; c++)
		if (counts[c] > 0) {
			modelled++;
			for (int d = 0; d < 4; d++)
				pooled[d] += ss[c][d] / counts[c];
		}
	for (int d = 0; d < 4; d++)
		pooled[d] /= modelled;
	Likelihood out{};
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			out.model.var[c][d] = std::max((ss[c][d] + VAR_SHRINK * pooled[d]) / (counts[c] + VAR_SHRINK), VAR_FLOOR);
	out.model.mean = means;
	{
		std::map<Group, std::array<double, 4>> sorted;
		for (size_t i = 0; i < N; i++)
			sorted[{obs[i].view, obs[i].face}] = offset[groups.of[i]];
		out.model.offsets.assign(sorted.begin(), sorted.end());
	}
	std::array<double, 6> logdet{};
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			logdet[c] += std::log(out.model.var[c][d]);
	for (auto& r : out.cost)
		r.fill(0);
	// outlier: each observation is, with this probability, unrelated to its facelet's colour (a mis-located or
	// mis-placed frame), drawn from a broad density: the Student-t density 5 standard deviations out of the mean
	// colour, under the mean variance. A bad frame then costs every colour about the same instead of vetoing one.
	double logdetMean = 0;
	for (int c = 0; c < 6; c++)
		if (counts[c] > 0)
			logdetMean += logdet[c] / modelled;
	const double broad = 0.5 * logdetMean + 0.5 * (T_DOF + 4) * std::log1p(4 * 25.0 / T_DOF);
	out.obsCost.assign(N, {0, 0, 0, 0, 0, 0});
	for (size_t i = 0; i < N; i++) {
		for (int c = 0; c < 6; c++) {
			if (counts[c] <= 0)
				continue;
			double maha = 0;
			for (int d = 0; d < 4; d++)
				maha += (Xc[i][d] - means[c][d]) * (Xc[i][d] - means[c][d]) / out.model.var[c][d];
			double nll = 0.5 * logdet[c] + 0.5 * (T_DOF + 4) * std::log1p(maha / T_DOF);
			if (outlier > 0) {
				// -log((1 - ε) e^-nll + ε e^-broad)
				double m = std::min(nll, broad);
				nll = m - std::log((1 - outlier) * std::exp(m - nll) + outlier * std::exp(m - broad));
			}
			out.cost[obs[i].facelet][c] += obs[i].weight * nll;
			out.obsCost[i][c] = nll;
		}
	}
	std::array<bool, 54> observed{};
	for (const Obs& o : obs)
		observed[o.facelet] = true;
	for (int k = 0; k < 54; k++)
		if (observed[k]) {
			double worst = -std::numeric_limits<double>::infinity();
			for (int c = 0; c < 6; c++)
				if (counts[c] > 0)
					worst = std::max(worst, out.cost[k][c]);
			for (int c = 0; c < 6; c++)
				if (counts[c] <= 0)
					out.cost[k][c] = worst + 10;
		}
	for (size_t i = 0; i < N; i++) {
		double worst = -std::numeric_limits<double>::infinity();
		for (int c = 0; c < 6; c++)
			if (counts[c] > 0)
				worst = std::max(worst, out.obsCost[i][c]);
		for (int c = 0; c < 6; c++)
			if (counts[c] <= 0)
				out.obsCost[i][c] = worst + 10;
	}
	return out;
}

std::array<std::array<double, 6>, 54> posterior (const std::array<std::array<double, 6>, 54>& cost) {
	std::array<std::array<double, 6>, 54> p{};
	for (int k = 0; k < 54; k++) {
		double m = *std::min_element(cost[k].begin(), cost[k].end()), s = 0;
		for (int c = 0; c < 6; c++)
			s += p[k][c] = std::exp(-(cost[k][c] - m));
		for (int c = 0; c < 6; c++)
			p[k][c] /= s;
	}
	return p;
}

}
