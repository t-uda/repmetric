#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <limits>
#include <thread>

// --- Template-based Core Implementation ---

template <typename T>
T _calculate_cped_template(const std::string& X, const std::string& Y) {
    const size_t n = X.length();
    const size_t m = Y.length();
    const T INF = std::numeric_limits<T>::max() / 2;

    std::vector<T> dp((n + 1) * (m + 1), INF);
    auto index = [m_val = m](size_t i, size_t j) { return i * (m_val + 1) + j; };

    std::vector<T> col_mins(m + 1, INF);

    dp[index(0, 0)] = 0;
    col_mins[0] = 1;

    std::string_view y_sv(Y);

    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }

            T current_min = INF;

            if (i > 0 && j > 0) {
                T cost = (X[i - 1] == Y[j - 1]) ? 0 : 1;
                current_min = std::min(current_min, dp[index(i - 1, j - 1)] + cost);
            }
            if (j > 0) {
                current_min = std::min(current_min, dp[index(i, j - 1)] + 1);
            }
            if (i > 0) {
                current_min = std::min(current_min, col_mins[j]);
            }
            if (j > 1) {
                size_t max_len = std::min(j / 2, static_cast<size_t>(20));
                for (size_t length = max_len; length > 0; --length) {
                    std::string_view substring = y_sv.substr(j - length, length);
                    std::string_view search_space = y_sv.substr(0, j - length);
                    if (search_space.find(substring) != std::string_view::npos) {
                        T cost = dp[index(i, j - length)] + 1;
                        if (cost < current_min) {
                            current_min = cost;
                        }
                    }
                }
            }

            dp[index(i, j)] = current_min;
            col_mins[j] = std::min(col_mins[j], dp[index(i, j)] + 1);
        }
    }

    T result = dp[index(n, m)];
    return (result >= INF) ? -1 : result;
}

// --- C-style Interface for Python ---

extern "C" {
    // Expose the integer version for performance
    int calculate_cped_cpp_int(const char* X_str, const char* Y_str) {
        return _calculate_cped_template<int>(std::string(X_str), std::string(Y_str));
    }

    // Expose the double version for future weighted extensions
    double calculate_cped_cpp_double(const char* X_str, const char* Y_str) {
        return _calculate_cped_template<double>(std::string(X_str), std::string(Y_str));
    }

    // Integer version of the distance matrix calculation
    void calculate_cped_distance_matrix_cpp_int(const char** sequences, int n, int* dist_matrix, bool parallel) {
        std::vector<std::string> seqs(n);
        for (int i = 0; i < n; ++i) {
            seqs[i] = std::string(sequences[i]);
        }

        auto worker = [&](int start_row, int step) {
            for (int i = start_row; i < n; i += step) {
                for (int j = 0; j < n; ++j) {
                    dist_matrix[i * n + j] = _calculate_cped_template<int>(seqs[i], seqs[j]);
                }
            }
        };

        if (parallel) {
            std::vector<std::thread> threads;
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
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
