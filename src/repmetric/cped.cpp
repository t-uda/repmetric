#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// --- Template-based Core Implementation ---

template <typename T>
T _calculate_cped_template(const std::string &X, const std::string &Y) {
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

std::pair<int, std::string>
_calculate_cped_geodesic_impl(const std::string &X, const std::string &Y) {
  const size_t n = X.length();
  const size_t m = Y.length();
  const int INF = std::numeric_limits<int>::max() / 2;

  std::vector<int> dp((n + 1) * (m + 1), INF);
  auto index = [m_val = m](size_t i, size_t j) { return i * (m_val + 1) + j; };

  std::vector<int> col_mins(m + 1, INF);

  dp[index(0, 0)] = 0;
  col_mins[0] = 1;

  std::string_view y_sv(Y);

  // Forward pass to fill DP table
  for (size_t i = 0; i <= n; ++i) {
    for (size_t j = 0; j <= m; ++j) {
      if (i == 0 && j == 0)
        continue;

      int current_min = INF;

      if (i > 0 && j > 0) {
        int cost = (X[i - 1] == Y[j - 1]) ? 0 : 1;
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
            int cost = dp[index(i, j - length)] + 1;
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

  if (dp[index(n, m)] >= INF) {
    return {-1, ""};
  }

  // Backtracking
  std::vector<std::string> path_ops;
  size_t i = n;
  size_t j = m;

  while (i > 0 || j > 0) {
    int current_val = dp[index(i, j)];

    // Check Match/Sub
    if (i > 0 && j > 0) {
      int cost = (X[i - 1] == Y[j - 1]) ? 0 : 1;
      if (current_val == dp[index(i - 1, j - 1)] + cost) {
        path_ops.push_back(cost == 0 ? "M" : "S");
        i--;
        j--;
        continue;
      }
    }

    // Check Insert
    if (j > 0) {
      if (current_val == dp[index(i, j - 1)] + 1) {
        path_ops.push_back("I");
        j--;
        continue;
      }
    }

    // Check Copy
    bool found_copy = false;
    if (j > 1) {
      size_t max_len = std::min(j / 2, static_cast<size_t>(20));
      for (size_t length = max_len; length > 0; --length) {
        std::string_view substring = y_sv.substr(j - length, length);
        std::string_view search_space = y_sv.substr(0, j - length);
        if (search_space.find(substring) != std::string_view::npos) {
          if (current_val == dp[index(i, j - length)] + 1) {
            path_ops.push_back("C:" + std::to_string(length));
            j -= length;
            found_copy = true;
            break;
          }
        }
      }
    }
    if (found_copy)
      continue;

    // Check Block Delete
    // We need to find k < i such that dp[k][j] + 1 == current_val
    if (i > 0) {
      bool found_delete = false;
      for (size_t k = 0; k < i; ++k) {
        if (dp[index(k, j)] + 1 == current_val) {
          path_ops.push_back("D:" + std::to_string(i - k));
          i = k;
          found_delete = true;
          break;
        }
      }
      if (found_delete)
        continue;
    }

    // Should not reach here if path exists
    break;
  }

  std::string path_str;
  for (auto it = path_ops.rbegin(); it != path_ops.rend(); ++it) {
    path_str += *it;
    if (it + 1 != path_ops.rend()) {
      path_str += ","; // Use comma separator
    }
  }

  return {dp[index(n, m)], path_str};
}

// --- C-style Interface for Python ---

extern "C" {
// Expose the integer version for performance
int calculate_cped_cpp_int(const char *X_str, const char *Y_str) {
  return _calculate_cped_template<int>(std::string(X_str), std::string(Y_str));
}

// Expose the double version for future weighted extensions
double calculate_cped_cpp_double(const char *X_str, const char *Y_str) {
  return _calculate_cped_template<double>(std::string(X_str),
                                          std::string(Y_str));
}

// Integer version of the distance matrix calculation
void calculate_cped_distance_matrix_cpp_int(const char **sequences, int n,
                                            int *dist_matrix, bool parallel) {
  std::vector<std::string> seqs(n);
  for (int i = 0; i < n; ++i) {
    seqs[i] = std::string(sequences[i]);
  }

  auto worker = [&](int start_row, int step) {
    for (int i = start_row; i < n; i += step) {
      for (int j = 0; j < n; ++j) {
        dist_matrix[i * n + j] =
            _calculate_cped_template<int>(seqs[i], seqs[j]);
      }
    }
  };

  if (parallel) {
    std::vector<std::thread> threads;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 1;
    for (unsigned int i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker, i, num_threads);
    }

    for (auto &t : threads) {
      t.join();
    }
  } else {
    worker(0, 1); // Sequential execution
  }
}

// Geodesic calculation
int calculate_cped_geodesic_cpp(const char *X_str, const char *Y_str,
                                char *buffer, int buffer_len) {
  std::pair<int, std::string> result =
      _calculate_cped_geodesic_impl(std::string(X_str), std::string(Y_str));
  if (result.second.length() >= static_cast<size_t>(buffer_len)) {
    return -1; // Buffer too small
  }
  std::strcpy(buffer, result.second.c_str());
  return result.first;
}
}
