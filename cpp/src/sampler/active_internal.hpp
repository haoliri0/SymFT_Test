#pragma once

#include "core/internal.hpp"
#include "sampler/active.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace symft::detail {

inline constexpr std::size_t kNoSource = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kSimdPairRotationThreshold = 4096;

#if defined(__clang__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SINGLE_SIMD_LOOP
#endif

inline Complex phase_factor(int phase) {
    switch (phase & 3) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, 1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, -1.0};
    }
}

inline std::size_t active_length(int k) {
    if (k < 0 || k >= 62) {
        fail("active qubit count is too large for machine basis indices");
    }
    return std::size_t{1} << k;
}

inline std::size_t insert_zero_bit(std::size_t packed, int bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

inline std::uint64_t active_mask_x(const PauliString& pauli) {
    return pauli.x.empty() ? 0 : pauli.x[0];
}

inline std::uint64_t active_mask_z(const PauliString& pauli) {
    return pauli.z.empty() ? 0 : pauli.z[0];
}

inline bool active_action_phase_odd(const ActivePauliAction& action, std::size_t basis) {
    return is_odd_popcount(static_cast<std::uint64_t>(basis) & action.zmask);
}

inline Complex active_action_phase(const ActivePauliAction& action, std::size_t basis) {
    return active_action_phase_odd(action, basis) ? action.odd_phase : action.even_phase;
}

inline bool is_near_zero(double value) {
    return std::abs(value) < 1e-14;
}

inline bool can_rotate_real_pair_flip(const ActivePauliAction& action) {
    if (action.zmask == 0 || !action.xz_overlap_odd) {
        return false;
    }
    return is_near_zero(action.even_phase.real()) && !is_near_zero(action.even_phase.imag());
}

} // namespace symft::detail
