#pragma once
// qs_math_pch.hpp — Pre-compiled header
// Contains ONLY C++ standard library headers.
// qs_math domain headers are included per translation unit in core/src/*/
// This ensures the PCH always compiles successfully regardless of
// header dependency ordering.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <array>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <type_traits>
#include <limits>
#include <numeric>
#include <iterator>
#include <initializer_list>
#include <optional>
#include <variant>
#include <span>
#include <bit>
#include <numbers>
#include <concepts>
#include <ranges>
#include <compare>
#include <tuple>
#include <string>
#include <string_view>
#include <complex>
#include <random>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
