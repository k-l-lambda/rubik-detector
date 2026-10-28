// scipy.optimize.minimize(method="Nelder-Mead") ported step for step (non-adaptive coefficients), so searches follow
// the Python reference: same initial simplex rule, same convergence test, same maxiter / maxfev defaults.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace rubik {

using Vec = std::vector<double>;

struct MinimizeResult {
	Vec x;
	double fun = 0;
	int nfev = 0;
	int nit = 0;
};

// scipy's default simplex: x0 plus 5 % along each coordinate (0.00025 for a zero coordinate).
inline std::vector<Vec> defaultSimplex (const Vec& x0) {
	std::vector<Vec> sim(x0.size() + 1, x0);
	for (size_t k = 0; k < x0.size(); k++)
		sim[k + 1][k] = sim[k + 1][k] != 0 ? 1.05 * sim[k + 1][k] : 0.00025;
	return sim;
}

// maxiter / maxfev <= 0 mean "not given": as scipy, both default to 200·N when neither is given, and the other is
// unbounded when one is.
template <typename F>
MinimizeResult nelderMead (F&& func, std::vector<Vec> sim, double xatol, double fatol, int maxiter = 0, int maxfev = 0) {
	const double rho = 1, chi = 2, psi = 0.5, sigma = 0.5;
	const size_t n = sim[0].size();
	const int inf = std::numeric_limits<int>::max();
	if (maxiter <= 0 && maxfev <= 0)
		maxiter = maxfev = int(200 * n);
	else if (maxiter <= 0)
		maxiter = inf;
	else if (maxfev <= 0)
		maxfev = inf;
	int fcalls = 0;
	auto f = [&](const Vec& x) {
		fcalls++;
		return double(func(x));
	};
	Vec fsim(n + 1);
	for (size_t k = 0; k <= n; k++)
		fsim[k] = f(sim[k]);
	auto sortSimplex = [&]() {
		std::vector<size_t> order(n + 1);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return fsim[a] < fsim[b]; });
		std::vector<Vec> s2(n + 1);
		Vec f2(n + 1);
		for (size_t i = 0; i <= n; i++) {
			s2[i] = sim[order[i]];
			f2[i] = fsim[order[i]];
		}
		sim.swap(s2);
		fsim.swap(f2);
	};
	sortSimplex();
	Vec xbar(n), xr(n), xe(n), xc(n);
	int iterations = 1;
	while (fcalls < maxfev && iterations < maxiter) {
		double xspread = 0, fspread = 0;
		for (size_t j = 1; j <= n; j++) {
			for (size_t i = 0; i < n; i++)
				xspread = std::max(xspread, std::fabs(sim[j][i] - sim[0][i]));
			fspread = std::max(fspread, std::fabs(fsim[0] - fsim[j]));
		}
		if (xspread <= xatol && fspread <= fatol)
			break;
		for (size_t i = 0; i < n; i++) {
			xbar[i] = 0;
			for (size_t j = 0; j < n; j++)
				xbar[i] += sim[j][i];
			xbar[i] /= double(n);
		}
		for (size_t i = 0; i < n; i++)
			xr[i] = (1 + rho) * xbar[i] - rho * sim[n][i];
		double fxr = f(xr);
		bool shrink = false;
		if (fxr < fsim[0]) {
			for (size_t i = 0; i < n; i++)
				xe[i] = (1 + rho * chi) * xbar[i] - rho * chi * sim[n][i];
			double fxe = f(xe);
			if (fxe < fxr) {
				sim[n] = xe;
				fsim[n] = fxe;
			}
			else {
				sim[n] = xr;
				fsim[n] = fxr;
			}
		}
		else if (fxr < fsim[n - 1]) {
			sim[n] = xr;
			fsim[n] = fxr;
		}
		else if (fxr < fsim[n]) {
			for (size_t i = 0; i < n; i++)
				xc[i] = (1 + psi * rho) * xbar[i] - psi * rho * sim[n][i];
			double fxc = f(xc);
			if (fxc <= fxr) {
				sim[n] = xc;
				fsim[n] = fxc;
			}
			else
				shrink = true;
		}
		else {
			for (size_t i = 0; i < n; i++)
				xc[i] = (1 - psi) * xbar[i] + psi * sim[n][i];
			double fxcc = f(xc);
			if (fxcc < fsim[n]) {
				sim[n] = xc;
				fsim[n] = fxcc;
			}
			else
				shrink = true;
		}
		if (shrink) {
			for (size_t j = 1; j <= n; j++) {
				for (size_t i = 0; i < n; i++)
					sim[j][i] = sim[0][i] + sigma * (sim[j][i] - sim[0][i]);
				fsim[j] = f(sim[j]);
			}
		}
		sortSimplex();
		iterations++;
	}
	return {sim[0], fsim[0], fcalls, iterations};
}

template <typename F>
MinimizeResult nelderMead (F&& func, const Vec& x0, double xatol, double fatol, int maxiter = 0, int maxfev = 0) {
	return nelderMead(func, defaultSimplex(x0), xatol, fatol, maxiter, maxfev);
}

}
