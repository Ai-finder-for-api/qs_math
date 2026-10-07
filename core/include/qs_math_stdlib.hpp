#pragma once
// qs_math_pch.hpp — Pre-compiled header (stdlib only)
// qs_build compiles this with /Yc"qs_math_pch.hpp" and passes
// /Yu"qs_math_pch.hpp" to every TU.  Keep this to pure stdlib so
// the PCH never depends on include-path ordering.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <climits>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
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
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <complex>
#include <numeric>
