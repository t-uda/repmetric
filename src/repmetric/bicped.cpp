#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <limits>
#include <thread>
#include <numeric>

// --- Core CPED DP Table Computation ---
// This is adapted from cped.cpp but returns the full DP table.
std::vector<double> _compute_cped_dp_table_cpp(const std::string& X, const std::string& Y, int max_copy_len = 20) {
    const size_t n = X.length();
    const size_t m = Y.length();
    const double INF = std::numeric_limits<double>::infinity();

    auto index = [m_val = m](size_t i, size_t j) { return i * (m_val + 1) + j; };
    std::vector<double> dp( (n + 1) * (m + 1), INF);

    dp[index(0, 0)] = 0.0;

    std::vector<double> col_mins(m + 1, INF);
    col_mins[0] = 1.0;

    std::string_view y_sv(Y);

    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            if (i == 0 && j == 0) {
                col_mins[j] = std::min(col_mins[j], dp[index(i,j)] + 1.0);
                continue;
            }
            double current_min = INF;
            if (i > 0 && j > 0) {
                double cost = (X[i - 1] == Y[j - 1]) ? 0.0 : 1.0;
                current_min = std::min(current_min, dp[index(i - 1, j - 1)] + cost);
            }
            if (j > 0) {
                current_min = std::min(current_min, dp[index(i, j - 1)] + 1.0);
            }
            if (i > 0) {
                current_min = std::min(current_min, col_mins[j]);
            }
            if (j > 1) {
                size_t max_len = std::min(j / 2, static_cast<size_t>(max_copy_len));
                for (size_t length = max_len; length > 0; --length) {
                    std::string_view substring = y_sv.substr(j - length, length);
                    if (y_sv.substr(0, j - length).find(substring) != std::string_view::npos) {
                        double cost = dp[index(i, j - length)] + 1.0;
                        if (cost < current_min) {
                            current_min = cost;
                        }
                    }
                }
            }
            dp[index(i, j)] = current_min;
            col_mins[j] = std::min(col_mins[j], dp[index(i, j)] + 1.0);
        }
    }
    return dp;
}


// --- BICPED Core Implementation ---
int _calculate_bicped_cpp(const std::string& X, const std::string& Y) {
    const size_t n = X.length();
    const size_t m = Y.length();
    const int max_copy_len = 20;
    const double INF = std::numeric_limits<double>::infinity();
    auto index = [m_val = m](size_t i, size_t j) { return i * (m_val + 1) + j; };

    // Forward DP
    std::vector<double> forward_dp = _compute_cped_dp_table_cpp(X, Y, max_copy_len);

    // Backward DP
    std::string X_rev = X;
    std::string Y_rev = Y;
    std::reverse(X_rev.begin(), X_rev.end());
    std::reverse(Y_rev.begin(), Y_rev.end());
    std::vector<double> reverse_dp_flat = _compute_cped_dp_table_cpp(X_rev, Y_rev, max_copy_len);

    std::vector<double> backward_dp((n + 1) * (m + 1));
    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            backward_dp[index(i, j)] = reverse_dp_flat[index(n - i, m - j)];
        }
    }

    double best = forward_dp[index(n, m)];

    // Refinement Step
    std::vector<std::vector<int>> future_repeats(m + 1);
    std::string_view y_sv(Y);
    for (size_t j = 0; j < m; ++j) {
        size_t limit = std::min(m - j, static_cast<size_t>(max_copy_len));
        for (size_t length = 1; length <= limit; ++length) {
            std::string_view substring = y_sv.substr(j, length);
            if (y_sv.find(substring, j + 1) != std::string_view::npos) {
                future_repeats[j].push_back(length);
            }
        }
    }

    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            if (forward_dp[index(i, j)] == INF) continue;
            for (int length : future_repeats[j]) {
                if (j + length > m) continue;
                if (backward_dp[index(i, j + length)] <= static_cast<double>(length)) continue;

                double candidate = forward_dp[index(i, j)] + 1.0 + backward_dp[index(i, j + length)];
                if (candidate < best) {
                    best = candidate;
                }
            }
        }
    }

    return (best != INF) ? static_cast<int>(best) : -1;
}

// --- C-style Interface for Python ---
extern "C" {
    int calculate_bicped_cpp_int(const char* X_str, const char* Y_str) {
        return _calculate_bicped_cpp(std::string(X_str), std::string(Y_str));
    }

    void calculate_bicped_distance_matrix_cpp_int(const char** sequences, int n, int* dist_matrix, bool parallel) {
        std::vector<std::string> seqs(n);
        for (int i = 0; i < n; ++i) {
            seqs[i] = std::string(sequences[i]);
        }

        auto worker = [&](int start_row, int step) {
            for (int i = start_row; i < n; i += step) {
                for (int j = 0; j < n; ++j) {
                    dist_matrix[i * n + j] = _calculate_bicped_cpp(seqs[i], seqs[j]);
                }
            }
        };

        if (parallel) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
            std::vector<std::thread> threads;
            for (unsigned int i = 0; i < num_threads; ++i) {
                threads.emplace_back(worker, i, num_threads);
            }
            for (auto& t : threads) {
                t.join();
            }
        } else {
            worker(0, 1); // Sequential execution
        }
    }
}
