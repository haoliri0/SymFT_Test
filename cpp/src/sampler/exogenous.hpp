#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous {
    int nshots = 0;
    int nsymbols = 0;
    std::size_t nwords = 0;
    std::uint64_t next_rng_state = 0;
    std::vector<std::uint64_t> exogenous_assigned_words;
    std::vector<std::uint64_t> value_words;
};

PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);

} // namespace symft
