#pragma once

#include "factored/factored.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace symft::detail {


inline constexpr int kWordBits = 64;
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr std::size_t kNoSource = std::numeric_limits<std::size_t>::max();
inline constexpr double kLowProbabilitySampleThreshold = 0.02;
inline constexpr std::size_t kSimdPairRotationThreshold = 4096;

#if defined(__clang__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SINGLE_SIMD_LOOP
#endif

#define SYMFT_RESTRICT

#if defined(__AVX2__) && defined(__FMA__)
#define SYMFT_INLINE_AVX2 1
#else
#define SYMFT_INLINE_AVX2 0
#endif

[[noreturn]] inline void fail(const std::string& message) {
    throw Error(message);
}

inline int check_qubit(int nqubits, int q) {
    if (q < 0 || q >= nqubits) {
        fail("qubit index out of range");
    }
    return q;
}

inline void check_same_nqubits(const PauliString& a, const PauliString& b) {
    if (a.nqubits != b.nqubits) {
        fail("Pauli strings act on different numbers of qubits");
    }
}

inline std::uint64_t bit_mask(int q) {
    return std::uint64_t{1} << (q & 63);
}

inline int word_index(int q) {
    return q >> 6;
}

inline int popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

inline std::uint64_t symbol_bit_mask(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return std::uint64_t{1} << ((condition - 1) & 63);
}

inline std::size_t symbol_word_index(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return static_cast<std::size_t>((condition - 1) >> 6);
}

inline std::size_t symbol_word_count(int nsymbols) {
    if (nsymbols <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((nsymbols + 63) >> 6);
}

inline int trailing_zeros64(std::uint64_t value) {
    if (value == 0) {
        fail("trailing_zeros64 called with zero");
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
#endif
}

inline bool is_odd_popcount(std::uint64_t value) {
    return (popcount64(value) & 1) != 0;
}

inline double check_probability(double probability) {
    if (!(probability >= 0.0 && probability <= 1.0)) {
        fail("probability must be between 0 and 1");
    }
    return probability;
}

inline void toggle_condition(std::vector<int>& conditions, int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    auto it = std::lower_bound(conditions.begin(), conditions.end(), condition);
    if (it != conditions.end() && *it == condition) {
        conditions.erase(it);
    } else {
        conditions.insert(it, condition);
    }
}

inline std::vector<int> normalize_conditions(const std::vector<int>& conditions) {
    std::vector<int> out;
    for (int condition : conditions) {
        toggle_condition(out, condition);
    }
    return out;
}

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

inline PauliString project_pauli_body(const PauliString& pauli, int qstart, int qcount) {
    PauliString out(qcount);
    int y_count = 0;
    for (int q = 0; q < qcount; ++q) {
        const int src = qstart + q;
        const bool xb = pauli.xbit(src);
        const bool zb = pauli.zbit(src);
        if (xb) {
            out.set_xbit(q);
        }
        if (zb) {
            out.set_zbit(q);
        }
        if (xb && zb) {
            ++y_count;
        }
    }
    out.set_phase(y_count);
    return out;
}

inline PauliString embed_active_pauli(int n, const PauliString& active_body) {
    PauliString out(n);
    for (int q = 0; q < active_body.nqubits; ++q) {
        if (active_body.xbit(q)) {
            out.set_xbit(q);
        }
        if (active_body.zbit(q)) {
            out.set_zbit(q);
        }
    }
    out.set_phase(active_body.phase_exponent());
    return out;
}

inline bool optional_equal(const std::optional<int>& lhs, const std::optional<int>& rhs) {
    return lhs.has_value() == rhs.has_value() && (!lhs || *lhs == *rhs);
}

inline int max_condition(const SymbolicPauliString& pauli) {
    return pauli.sign.max_condition();
}

inline int max_condition(const PendingPauliRotation& rotation) {
    return max_condition(rotation.pauli);
}

inline int max_condition(const PendingPauliMeasurement& measurement) {
    int out = max_condition(measurement.pauli);
    if (measurement.record_condition) {
        out = std::max(out, *measurement.record_condition);
    }
    return out;
}

template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

inline int max_condition(const PendingOperation& operation) {
    return std::visit(Overloaded{
                          [](const PendingPauliRotation& op) { return max_condition(op); },
                          [](const PendingPauliMeasurement& op) { return max_condition(op); },
                      },
                      operation);
}

inline int max_condition(const ApplyPrecomputedActivePauliRotation& instruction) {
    return instruction.sign.max_condition();
}

inline int max_condition(const ApplyActiveBasisChange&) {
    return 0;
}

inline int max_condition(const PromoteDormantRotation& instruction) {
    return instruction.sign.max_condition();
}

inline int max_condition(const RecordMeasurement& instruction) {
    int out = instruction.outcome.max_condition();
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const MeasureActiveLastZ& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const MeasurePrecomputedActivePauli& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const IntroduceDormantMeasurementBranch& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return max_condition(inst); }, instruction);
}

} // namespace symft::detail
