// Port of the classification parts of rubik_detector/colors.py: balanced clustering, the colour likelihood model.
#include <algorithm>
#include <cmath>
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

std::array<int, 54> classifyBalanced (const std::vector<Obs>& obs, int iterations) {
	std::map<Group, std::array<double, 4>> offset;
	for (const Obs& o : obs)
		offset[{o.view, o.face}] = {0, 0, 0, 0};
	auto corrected = [&](const Obs& o) {
		const auto& off = offset[{o.view, o.face}];
		return std::array<double, 4>{o.x[0] - off[0], o.x[1] - off[1], o.x[2] - off[2], o.x[3] - off[3]};
	};
	std::array<std::array<double, 4>, 6> means{};
	for (int c = 0; c < 6; c++) {
		int n = 0;
		for (const Obs& o : obs)
			if (o.facelet == CENTRE_FACELETS[c]) {
				auto x = corrected(o);
				for (int d = 0; d < 4; d++)
					means[c][d] += x[d];
				n++;
			}
		for (int d = 0; d < 4; d++)
			means[c][d] /= n;
	}
	std::array<int, 54> labels{};
	for (int it = 0; it < iterations; it++) {
		std::vector<std::vector<double>> cost(54, std::vector<double>(6, 0));
		std::array<int, 54> count{};
		for (const Obs& o : obs) {
			auto x = corrected(o);
			for (int c = 0; c < 6; c++) {
				double s = 0;
				for (int d = 0; d < 4; d++)
					s += (means[c][d] - x[d]) * (means[c][d] - x[d]);
				cost[o.facelet][c] += std::sqrt(s);
			}
			count[o.facelet]++;
		}
		double maxCost = 0;
		for (int k = 0; k < 54; k++)
			for (int c = 0; c < 6; c++) {
				cost[k][c] /= count[k];
				maxCost = std::max(maxCost, cost[k][c]);
			}
		double big = maxCost * 10 + 1;
		for (int c = 0; c < 6; c++)
			for (int j = 0; j < 6; j++)
				cost[CENTRE_FACELETS[c]][j] = j == c ? 0 : big;
		// nine slots per colour
		std::vector<std::vector<double>> C(54, std::vector<double>(54));
		for (int k = 0; k < 54; k++)
			for (int j = 0; j < 54; j++)
				C[k][j] = cost[k][j / 9];
		std::vector<int> perm = assignment(C);
		for (int k = 0; k < 54; k++)
			labels[k] = perm[k] / 9;
		// lighting offset per photographed face: median residual on log chroma and lightness, shrunk towards 0
		for (auto& entry : offset) {
			std::vector<double> res[4];
			for (const Obs& o : obs)
				if (Group{o.view, o.face} == entry.first)
					for (int d = 0; d < 4; d++)
						res[d].push_back(o.x[d] - means[labels[o.facelet]][d]);
			double n = double(res[0].size());
			std::array<double, 4> off{0, 0, 0, 0};
			for (int d : LIGHT_DIMS)
				off[d] = median(res[d]) * n / (n + 2);
			entry.second = off;
		}
		for (int c = 0; c < 6; c++) {
			std::array<double, 4> sum{};
			int n = 0;
			for (const Obs& o : obs)
				if (labels[o.facelet] == c) {
					auto x = corrected(o);
					for (int d = 0; d < 4; d++)
						sum[d] += x[d];
					n++;
				}
			for (int d = 0; d < 4; d++)
				means[c][d] = sum[d] / n;
		}
	}
	return labels;
}

Likelihood likelihood (const std::vector<Obs>& obs, const std::array<int, 54>& labels) {
	const double OFFSET_SHRINK = 2.0, VAR_SHRINK = 6.0;
	std::map<Group, std::array<double, 4>> offset;
	for (const Obs& o : obs)
		offset[{o.view, o.face}] = {0, 0, 0, 0};
	const size_t N = obs.size();
	std::vector<std::array<double, 4>> Xc(N);
	std::array<std::array<double, 4>, 6> means{};
	auto correct = [&]() {
		for (size_t i = 0; i < N; i++) {
			const auto& off = offset[{obs[i].view, obs[i].face}];
			for (int d = 0; d < 4; d++)
				Xc[i][d] = obs[i].x[d] - off[d];
		}
	};
	auto fitMeans = [&]() {
		for (int c = 0; c < 6; c++) {
			std::array<double, 4> sum{};
			int n = 0;
			for (size_t i = 0; i < N; i++)
				if (labels[obs[i].facelet] == c) {
					for (int d = 0; d < 4; d++)
						sum[d] += Xc[i][d];
					n++;
				}
			for (int d = 0; d < 4; d++)
				means[c][d] = sum[d] / n;
		}
	};
	for (int it = 0; it < 3; it++) {
		correct();
		fitMeans();
		for (auto& entry : offset) {
			std::vector<double> res[4];
			for (size_t i = 0; i < N; i++)
				if (Group{obs[i].view, obs[i].face} == entry.first)
					for (int d = 0; d < 4; d++)
						res[d].push_back(obs[i].x[d] - means[labels[obs[i].facelet]][d]);
			double n = double(res[0].size());
			std::array<double, 4> off{0, 0, 0, 0};
			for (int d : LIGHT_DIMS)
				off[d] = median(res[d]) * n / (n + OFFSET_SHRINK);
			entry.second = off;
		}
	}
	correct();
	fitMeans();
	std::array<double, 4> pooled{};
	std::array<std::array<double, 4>, 6> ss{};
	std::array<int, 6> counts{};
	for (size_t i = 0; i < N; i++) {
		int c = labels[obs[i].facelet];
		counts[c]++;
		for (int d = 0; d < 4; d++)
			ss[c][d] += (Xc[i][d] - means[c][d]) * (Xc[i][d] - means[c][d]);
	}
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			pooled[d] += ss[c][d] / counts[c] / 6;
	Likelihood out{};
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			out.model.var[c][d] = std::max((ss[c][d] + VAR_SHRINK * pooled[d]) / (counts[c] + VAR_SHRINK), VAR_FLOOR);
	out.model.mean = means;
	for (const auto& entry : offset)
		out.model.offsets.push_back(entry);
	std::array<double, 6> logdet{};
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++)
			logdet[c] += std::log(out.model.var[c][d]);
	for (auto& r : out.cost)
		r.fill(0);
	for (size_t i = 0; i < N; i++) {
		for (int c = 0; c < 6; c++) {
			double maha = 0;
			for (int d = 0; d < 4; d++)
				maha += (Xc[i][d] - means[c][d]) * (Xc[i][d] - means[c][d]) / out.model.var[c][d];
			out.cost[obs[i].facelet][c] += 0.5 * logdet[c] + 0.5 * (T_DOF + 4) * std::log1p(maha / T_DOF);
		}
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
