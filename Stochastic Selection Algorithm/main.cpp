#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>

#include "hrc_profiler.hpp"

#include <boost/random.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using quad = boost::multiprecision::cpp_bin_float_quad;

hrc_profiler profiler1("Non-batched data generation");
hrc_profiler profiler2("Non-batched selection");
hrc_profiler profiler3("Batched data generation");
hrc_profiler profiler4("Batched selection (sort)");
hrc_profiler profiler5("Batched selection (merge)");
hrc_profiler profiler6("Batched check");

struct slct_data_view_t
{
	auto operator<=>(const slct_data_view_t& rhs) const
	{
		return *curr_it <=> *(rhs.curr_it);
	}

	std::vector<double>::const_iterator curr_it;
	std::vector<double>::const_iterator end_it;
	bool success_flag;
};

template <class T, class S, class C> S& get_priority_queue_container(std::priority_queue<T, S, C>& q)
{
	struct priority_queue_view_t : private std::priority_queue<T, S, C>
	{
		static S& get_priority_queue_container(std::priority_queue<T, S, C>& q)
		{
			return q.*(&priority_queue_view_t::c);
		}
	};
	return priority_queue_view_t::get_priority_queue_container(q);
}

int64_t opt_radius(int64_t N, int64_t m, int64_t K, quad tgt_err, double& best_err)
{
	quad p = quad(K) / N;
	int64_t k = (m * K - 1) / N + 1;

	quad q_l = 0.;
	for (int64_t i = 0; i < k; i++)
	{
		q_l += log(quad(m - i)) - log(quad(i + 1));
	}
	q_l += k * log(p) + (m - k) * log(1. - p);
	quad q_u = q_l;

	quad best_err_m = 1. - exp(q_l);
	quad tgt_err_m = 1. - pow(1. - tgt_err, static_cast<quad>(m) / N);

	int64_t l = 0;
	while (best_err_m > tgt_err_m)
	{
		l++;

		if (k - l >= 0)
		{
			q_l -= log(quad(m - k + l)) - log(quad(k - l + 1)) + log(p) - log(1. - p);
			best_err_m -= exp(q_l);
		}
		if (k + l <= m)
		{
			q_u += log(quad(m - k - l + 1)) - log(quad(k + l)) + log(p) - log(1. - p);
			best_err_m -= exp(q_u);
		}
	}

	best_err = static_cast<double>(1. - pow(1. - best_err_m, static_cast<quad>(N) / m));

	return l;
}

double mem_rate(int64_t N, int64_t M, int64_t L, int64_t K)
{
	double p = static_cast<double>(K) / N;
	int64_t k = ceil(M * p);
	int64_t k_l = std::max(k - L, 1LL);
	int64_t k_u = std::min(k + L, M);
	double rate = static_cast<double>(k_u - k_l + 1) / M + static_cast<double>(M) / N;

	return rate;
}

int64_t opt_batch_size(int64_t N, int64_t K, double tgt_err)
{
	for (int64_t M_l = 1, M_u = N;;)
	{
		int64_t M1 = (2 * M_l + M_u) / 3;
		int64_t M2 = (M_l + 2 * M_u) / 3;

		double best_err;
		int64_t L1 = opt_radius(N, M1, K, tgt_err, best_err);
		double rate1 = mem_rate(N, M1, L1, K);
		int64_t L2 = opt_radius(N, M2, K, tgt_err, best_err);
		double rate2 = mem_rate(N, M2, L2, K);

		if (rate1 > rate2)
		{
			if (M_u <= M_l + 2)
			{
				return M2;
			}
			M_l = M1;
		}
		else
		{
			if (M_u <= M_l + 2)
			{
				return M1;
			}
			M_u = M2;
		}
	}
}

template<class C> double non_batched_ec(int64_t test_num, int64_t N, int64_t K, C cmp, std::vector<double>& __restrict data)
{
	boost::mt19937_64 rng_engine;
	rng_engine.seed(test_num);
	boost::normal_distribution dist;

	profiler1.start();
	for (auto& val : data)
	{
		val = dist(rng_engine);
	}
	profiler1.stop();

	profiler2.start();
	std::partial_sort(data.begin(), data.begin() + K, data.end(), cmp);
	profiler2.stop();

	return data[K - 1];
}

template<class C> double batched_ec(int64_t test_num, int64_t N, int64_t M, int64_t L, int64_t K, C cmp, std::vector<double>& __restrict data, std::vector<std::vector<double>>& __restrict slct_data, std::vector<slct_data_view_t>& __restrict slct_data_view, bool& success_flag, std::vector<double>& __restrict nb_data)
{
	const int64_t num_batches = (N - 1) / M + 1;

	boost::mt19937_64 rng_engine;
	rng_engine.seed(test_num);
	boost::normal_distribution dist;

	int64_t n = 1;

	for (int64_t i = 0; i < num_batches; i++)
	{
		const int64_t m = std::min(N - M * i, M);
		const int64_t k = (m * K - 1) / N + 1;
		const int64_t k_l = std::max(k - L, 1LL);
		const int64_t k_u = std::min(k + L, m);

		n += k_l - 1;

		profiler3.start();
		for (int64_t j = 0; j < m; j++)
		{
			data[j] = dist(rng_engine);
		}
		profiler3.stop();

		profiler4.start();
		std::partial_sort(data.begin(), data.begin() + k_u, data.begin() + m, cmp);

		std::copy(data.begin() + k_l - 1, data.begin() + k_u, slct_data[i].begin());

		slct_data_view[i].curr_it = slct_data[i].begin();
		slct_data_view[i].end_it = slct_data[i].end();
		slct_data_view[i].success_flag = k_l == 1 || k_u == m;
		profiler4.stop();
	}

	auto merge_and_check = [&]<class T, class S, class C>(std::priority_queue<T, S, C> && pqueue)
	{
		profiler5.start();
		for (; n < K; n++)
		{
			auto top_elem = pqueue.top();
			pqueue.pop();
			++(top_elem.curr_it);
			if (top_elem.curr_it != top_elem.end_it)
			{
				top_elem.success_flag = true;
				pqueue.push(top_elem);
			}
		}
		double q = *(pqueue.top().curr_it);
		profiler5.stop();

		profiler6.start();
		success_flag = pqueue.size() == num_batches;
		if (success_flag)
		{
			const auto& pqueue_container = get_priority_queue_container(pqueue);
			for (const auto& slct_data_batch : pqueue_container)
			{
				if (!slct_data_batch.success_flag)
				{
					success_flag = false;
					break;
				}
			}
		}
		profiler6.stop();

		return q;
	};

	double q;
	if (std::is_same_v<C, std::less<void>>)
	{
		q = merge_and_check(std::priority_queue(slct_data_view.begin(), slct_data_view.end(), std::greater()));
	}
	else
	{
		q = merge_and_check(std::priority_queue(slct_data_view.begin(), slct_data_view.end(), std::less()));
	}

	return q;
}

int main()
{
	std::cout << "Input\n";

	std::cout << "Number of tests: ";
	int64_t num_tests;
	std::cin >> num_tests;

	std::cout << "N: ";
	int64_t N;
	std::cin >> N;

	std::cout << "K: ";
	int64_t K;
	std::cin >> K;
	bool inv_cmp = false;
	if (2 * K > N)
	{
		K = N - K + 1;
		inv_cmp = true;
	}

	std::cout << "Target mismatch rate: ";
	double tgt_err;
	std::cin >> tgt_err;

	std::cout << "Use optimal batch size (y/n): ";
	char find_opt_batch_size;
	std::cin >> find_opt_batch_size;
	int64_t M;
	if (find_opt_batch_size != 'y')
	{
		std::cout << "Batch size: ";
		std::cin >> M;
	}
	else
	{
		M = opt_batch_size(N, K, tgt_err);
	}

	double best_err;
	const int64_t L = opt_radius(N, M, K, tgt_err, best_err);

	const int64_t num_batches = (N - 1) / M + 1;

	std::vector<double> non_batched_data(N);
	std::vector<double> batched_data(M);
	std::vector<std::vector<double>> slct_data(num_batches);
	std::vector<slct_data_view_t> slct_data_view(num_batches);

	for (int64_t i = 0; i < num_batches; i++)
	{
		const int64_t m = std::min(N - M * i, M);
		const int64_t k = (m * K - 1) / N + 1;
		const int64_t k_l = std::max(k - L, 1LL);
		const int64_t k_u = std::min(k + L, m);

		slct_data[i].resize(k_u - k_l + 1);
	}

	std::cout << "\nOutput\n";
	if (find_opt_batch_size == 'y')
	{
		std::cout << "Optimal batch size: " << M << '\n';
	}
	std::cout << "Best error rate: " << best_err << '\n';
	std::cout << "Memory reduction rate: " << mem_rate(N, M, L, K) << '\n';

	int64_t num_warnings = 0;
	int64_t num_missmatches = 0;
	int64_t num_missmatches_wo_warning = 0;

	for (int64_t i = 0; i < num_tests; i++)
	{
		double non_batched_result;
		double batched_result;
		bool success_flag;
		if (inv_cmp)
		{
			non_batched_result = non_batched_ec(i, N, K, std::greater(), non_batched_data);
			batched_result = batched_ec(i, N, M, L, K, std::greater(), batched_data, slct_data, slct_data_view, success_flag, non_batched_data);
		}
		else
		{
			non_batched_result = non_batched_ec(i, N, K, std::less(), non_batched_data);
			batched_result = batched_ec(i, N, M, L, K, std::less(), batched_data, slct_data, slct_data_view, success_flag, non_batched_data);
		}

		if (!success_flag)
		{
			num_warnings++;
		}
		if (non_batched_result != batched_result)
		{
			num_missmatches++;
			if (success_flag)
			{
				num_missmatches_wo_warning++;
			}
		}

		if ((i + 1) * 100 % num_tests == 0)
		{
			std::cout << "\x1b[2K";
			std::cout << "\x1b[G";
			std::cout << (i + 1) * 100 / num_tests << "%, W: " << num_warnings << ", M: " << num_missmatches << ", M w/o W: " << num_missmatches_wo_warning;
		}
	}
	std::cout << "\x1b[2K";
	std::cout << "\x1b[G";

	std::cout << "Number of warnings: " << num_warnings << '\n';
	std::cout << "Number of mismatches: " << num_missmatches << '\n';
	std::cout << "Number of mismatches without warning: " << num_missmatches_wo_warning << '\n';
	std::cout << "\nRun times:\n";
	std::cout << profiler1.print_time();
	std::cout << profiler2.print_time();
	std::cout << profiler3.print_time();
	std::cout << profiler4.print_time();
	std::cout << profiler5.print_time();
	std::cout << profiler6.print_time();

	system("pause");
	return 0;
}
