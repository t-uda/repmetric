#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <thread>
#include <cstring> // for strcpy

// --- Template-based Core Implementation for Levenshtein Distance ---

template <typename T>
T _calculate_levd_template(const std::string& s1, const std::string& s2) {
    const size_t n = s1.length();
    const size_t m = s2.length();

    if (n == 0) return m;
    if (m == 0) return n;

    std::vector<T> p(m + 1);
    std::vector<T> d(m + 1);

    for (size_t j = 0; j <= m; ++j) {
        p[j] = j;
    }

    for (size_t i = 1; i <= n; ++i) {
        d[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            T cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            d[j] = std::min({p[j] + 1, d[j - 1] + 1, p[j - 1] + cost});
        }
        p = d;
    }

    return p[m];
}

std::pair<int, std::string> _calculate_levd_geodesic_impl(const std::string& s1, const std::string& s2) {
    const size_t n = s1.length();
    const size_t m = s2.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

    for (size_t i = 0; i <= n; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= m; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
        }
    }

    std::string path;
    size_t i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            if (dp[i][j] == dp[i - 1][j - 1] + cost) {
                path += (cost == 0 ? 'M' : 'S'); // Match or Substitute
                i--; j--;
                continue;
            }
        }
        if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
            path += 'D'; // Deletion
            i--;
            continue;
        }
        if (j > 0 && dp[i][j] == dp[i][j - 1] + 1) {
            path += 'I'; // Insertion
            j--;
            continue;
        }
    }
    std::reverse(path.begin(), path.end());
    return {dp[n][m], path};
}

// --- C-style Interface for Python ---

extern "C" {
    // Expose the integer version for performance
    int calculate_levd_cpp_int(const char* s1_str, const char* s2_str) {
        return _calculate_levd_template<int>(std::string(s1_str), std::string(s2_str));
    }

    // Integer version of the distance matrix calculation
    void calculate_levd_distance_matrix_cpp_int(const char** sequences, int n, int* dist_matrix, bool parallel) {
        std::vector<std::string> seqs(n);
        for (int i = 0; i < n; ++i) {
            seqs[i] = std::string(sequences[i]);
        }

        auto worker = [&](int start_row, int step) {
            for (int i = start_row; i < n; i += step) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        dist_matrix[i * n + j] = 0;
                    } else if (i < j) {
                        int dist = _calculate_levd_template<int>(seqs[i], seqs[j]);
                        dist_matrix[i * n + j] = dist;
                        dist_matrix[j * n + i] = dist; // Levenshtein is symmetric
                    }
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
            worker(0, 1); // Sequential execution
        }
    }

    // Geodesic calculation
    int calculate_levd_geodesic_cpp(const char* s1_str, const char* s2_str, char* buffer, int buffer_len) {
        std::pair<int, std::string> result = _calculate_levd_geodesic_impl(std::string(s1_str), std::string(s2_str));
        if (result.second.length() >= static_cast<size_t>(buffer_len)) {
            return -1; // Buffer too small
        }
        std::strcpy(buffer, result.second.c_str());
        return result.first;
    }
}
