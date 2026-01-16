#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

unsigned int resolve_num_threads() {
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 1;
    }

    const char* env_value = std::getenv("REPMETRIC_NUM_THREADS");
    if (!env_value || *env_value == '\0') {
        return num_threads;
    }

    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(env_value, &end, 10);
    if (errno == ERANGE || end == env_value || *end != '\0' || parsed <= 0) {
        return num_threads;
    }

    if (parsed > static_cast<long>(num_threads)) {
        return num_threads;
    }

    return static_cast<unsigned int>(parsed);
}

class DPTable {
   public:
    DPTable(size_t rows, size_t cols, double initial_value)
        : cols_(cols), data_((rows + 1) * (cols + 1), initial_value) {}

    inline double& at(size_t i, size_t j) {
        return data_[i * (cols_ + 1) + j];
    }

    inline double at(size_t i, size_t j) const {
        return data_[i * (cols_ + 1) + j];
    }

   private:
    size_t cols_;
    std::vector<double> data_;
};

DPTable compute_cped_dp_table(const std::string& X, const std::string& Y, size_t max_copy_len) {
    const size_t n = X.length();
    const size_t m = Y.length();
    const double INF = std::numeric_limits<double>::infinity();

    DPTable dp(n, m, INF);
    std::vector<double> col_mins(m + 1, INF);

    dp.at(0, 0) = 0.0;
    col_mins[0] = 1.0;

    std::string_view y_sv(Y);

    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            if (i == 0 && j == 0) {
                col_mins[j] = std::min(col_mins[j], dp.at(i, j) + 1.0);
                continue;
            }

            double current_min = INF;

            if (i > 0 && j > 0) {
                double cost = (X[i - 1] == Y[j - 1]) ? 0.0 : 1.0;
                current_min = std::min(current_min, dp.at(i - 1, j - 1) + cost);
            }
            if (j > 0) {
                current_min = std::min(current_min, dp.at(i, j - 1) + 1.0);
            }
            if (i > 0) {
                current_min = std::min(current_min, col_mins[j]);
            }
            if (j > 1) {
                size_t max_len = std::min(j / 2, max_copy_len);
                for (size_t length = max_len; length > 0; --length) {
                    std::string_view substring = y_sv.substr(j - length, length);
                    std::string_view search_space = y_sv.substr(0, j - length);
                    if (search_space.find(substring) != std::string_view::npos) {
                        double cost = dp.at(i, j - length) + 1.0;
                        if (cost < current_min) {
                            current_min = cost;
                        }
                    }
                }
            }

            dp.at(i, j) = current_min;
            col_mins[j] = std::min(col_mins[j], dp.at(i, j) + 1.0);
        }
    }

    return dp;
}

int calculate_bicped_core(const std::string& X, const std::string& Y) {
    constexpr size_t max_copy_len = 20;

    DPTable forward_dp = compute_cped_dp_table(X, Y, max_copy_len);

    std::string X_rev(X.rbegin(), X.rend());
    std::string Y_rev(Y.rbegin(), Y.rend());
    DPTable reverse_dp = compute_cped_dp_table(X_rev, Y_rev, max_copy_len);

    const size_t n = X.length();
    const size_t m = Y.length();

    double best = forward_dp.at(n, m);

    std::vector<std::vector<int>> future_repeats(m + 1);
    std::string_view y_sv(Y);

    for (size_t j = 0; j < m; ++j) {
        size_t limit = std::min(m - j, max_copy_len);
        std::string_view remaining = y_sv.substr(j + 1);
        for (size_t length = 1; length <= limit; ++length) {
            std::string_view substring = y_sv.substr(j, length);
            if (remaining.find(substring) != std::string_view::npos) {
                future_repeats[j].push_back(static_cast<int>(length));
            }
        }
    }

    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            double forward_value = forward_dp.at(i, j);
            if (!std::isfinite(forward_value)) {
                continue;
            }
            for (int length : future_repeats[j]) {
                size_t next_j = j + static_cast<size_t>(length);
                if (next_j > m) {
                    continue;
                }
                double backward_value = reverse_dp.at(n - i, m - next_j);
                if (backward_value <= static_cast<double>(length)) {
                    continue;
                }
                double candidate = forward_value + 1.0 + backward_value;
                if (candidate < best) {
                    best = candidate;
                }
            }
        }
    }

    if (!std::isfinite(best)) {
        return -1;
    }
    return static_cast<int>(best);
}

}  // namespace

extern "C" {

int calculate_bicped_cpp_int(const char* X_str, const char* Y_str) {
    return calculate_bicped_core(std::string(X_str ? X_str : ""),
                                 std::string(Y_str ? Y_str : ""));
}

void calculate_bicped_distance_matrix_cpp_int(const char** sequences,
                                              int n,
                                              int* dist_matrix,
                                              bool parallel) {
    std::vector<std::string> seqs(n);
    for (int i = 0; i < n; ++i) {
        seqs[i] = std::string(sequences[i] ? sequences[i] : "");
    }

    auto worker = [&](int start_row, int step) {
        for (int i = start_row; i < n; i += step) {
            for (int j = 0; j < n; ++j) {
                dist_matrix[i * n + j] = calculate_bicped_core(seqs[i], seqs[j]);
            }
        }
    };

    if (parallel) {
        unsigned int num_threads = resolve_num_threads();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker, static_cast<int>(i), static_cast<int>(num_threads));
        }
        for (auto& t : threads) {
            t.join();
        }
    } else {
        worker(0, 1);
    }
}

}
