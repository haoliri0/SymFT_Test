#include "symft_internal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <variant>

namespace symft {
using namespace detail;

namespace {

std::string trim(const std::string& input) {
    std::size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) {
        ++first;
    }
    std::size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }
    return input.substr(first, last - first);
}

std::vector<std::string> split_ws(const std::string& input) {
    std::istringstream stream(input);
    std::vector<std::string> out;
    std::string item;
    while (stream >> item) {
        out.push_back(item);
    }
    return out;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string upper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool all_digits(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

std::string strip_stim_comment(const std::string& line) {
    bool in_brackets = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '[') {
            in_brackets = true;
        } else if (ch == ']') {
            in_brackets = false;
        } else if (ch == '#' && !in_brackets) {
            return trim(line.substr(0, i));
        }
    }
    return trim(line);
}

struct StimInstruction {
    std::string op;
    std::optional<std::string> tag;
    std::vector<double> parens;
    std::vector<std::string> targets;
    int line = 0;
};

struct StimRepeatBlock;
using StimNode = std::variant<StimInstruction, std::shared_ptr<StimRepeatBlock>>;

struct StimRepeatBlock {
    int count = 0;
    std::vector<StimNode> body;
    int line = 0;
};

StimInstruction parse_instruction(const std::string& body, int line) {
    std::size_t idx = 0;
    while (idx < body.size() && (std::isalpha(static_cast<unsigned char>(body[idx])) ||
                                 (idx > 0 && (std::isdigit(static_cast<unsigned char>(body[idx])) || body[idx] == '_')))) {
        ++idx;
    }
    if (idx == 0) {
        fail("invalid Stim instruction");
    }
    StimInstruction inst;
    inst.op = upper(body.substr(0, idx));
    inst.line = line;
    if (idx < body.size() && body[idx] == '[') {
        const std::size_t close = body.find(']', idx + 1);
        if (close == std::string::npos) {
            fail("unterminated Stim tag");
        }
        inst.tag = body.substr(idx + 1, close - idx - 1);
        idx = close + 1;
    }
    if (idx < body.size() && body[idx] == '(') {
        const std::size_t close = body.find(')', idx + 1);
        if (close == std::string::npos) {
            fail("unterminated Stim argument list");
        }
        const std::string args = trim(body.substr(idx + 1, close - idx - 1));
        if (!args.empty()) {
            std::size_t start = 0;
            while (start <= args.size()) {
                const std::size_t comma = args.find(',', start);
                const std::string piece = trim(args.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
                inst.parens.push_back(std::stod(piece));
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        }
        idx = close + 1;
    }
    if (idx < body.size()) {
        const std::string rest = trim(body.substr(idx));
        inst.targets = split_ws(rest);
    }
    return inst;
}

std::vector<StimNode> parse_nodes(const std::vector<std::string>& lines, std::size_t& pos, bool in_block) {
    std::vector<StimNode> nodes;
    while (pos < lines.size()) {
        std::string body = strip_stim_comment(lines[pos]);
        const int line = static_cast<int>(pos + 1);
        ++pos;
        if (body.empty()) {
            continue;
        }
        if (body == "}") {
            if (!in_block) {
                fail("unmatched Stim block terminator");
            }
            return nodes;
        }
        bool block_start = false;
        if (ends_with(body, "{")) {
            block_start = true;
            body = trim(body.substr(0, body.size() - 1));
        }
        StimInstruction inst = parse_instruction(body, line);
        if (block_start) {
            if (inst.op != "REPEAT" || inst.targets.size() != 1 || !all_digits(inst.targets[0])) {
                fail("invalid Stim REPEAT block");
            }
            auto block = std::make_shared<StimRepeatBlock>();
            block->count = std::stoi(inst.targets[0]);
            block->line = line;
            block->body = parse_nodes(lines, pos, true);
            nodes.push_back(block);
        } else {
            if (inst.op == "REPEAT") {
                fail("REPEAT must start a block");
            }
            nodes.push_back(inst);
        }
    }
    if (in_block) {
        fail("unterminated Stim block");
    }
    return nodes;
}

std::pair<int, bool> stim_qubit_target(const std::string& target, int line) {
    bool inverted = false;
    std::string body = target;
    if (starts_with(body, "!")) {
        inverted = true;
        body = body.substr(1);
    }
    if (!all_digits(body)) {
        (void)line;
        fail("invalid qubit target");
    }
    return {std::stoi(body), inverted};
}

std::vector<int> stim_qubit_targets(const StimInstruction& inst) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        auto [q, inverted] = stim_qubit_target(target, inst.line);
        if (inverted) {
            fail("operation does not accept inverted targets");
        }
        out.push_back(q);
    }
    return out;
}

std::vector<std::pair<int, bool>> stim_measurement_targets(const StimInstruction& inst) {
    std::vector<std::pair<int, bool>> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_qubit_target(target, inst.line));
    }
    return out;
}

int stim_record_index(const std::string& target, int nrecords) {
    if (!starts_with(target, "rec[") || !ends_with(target, "]")) {
        fail("invalid measurement record target");
    }
    const int offset = std::stoi(target.substr(4, target.size() - 5));
    const int idx = nrecords + offset + 1;
    if (idx <= 0 || idx > nrecords) {
        fail("measurement record target out of range");
    }
    return idx;
}

std::vector<int> stim_record_indices(const StimInstruction& inst, const std::vector<SymbolicBool>& records) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_record_index(target, static_cast<int>(records.size())));
    }
    return out;
}

double stim_paren_probability(const StimInstruction& inst, double default_probability = 0.0) {
    if (inst.parens.empty()) {
        return default_probability;
    }
    if (inst.parens.size() != 1) {
        fail("operation expects at most one probability argument");
    }
    return check_probability(inst.parens[0]);
}

double stim_required_probability(const StimInstruction& inst) {
    if (inst.parens.size() != 1) {
        fail("operation expects one probability argument");
    }
    return check_probability(inst.parens[0]);
}

int target_max_qubit(const std::string& target) {
    if (target == "*" || starts_with(target, "rec[") || starts_with(target, "sweep[")) {
        return -1;
    }
    std::string body = target;
    if (starts_with(body, "!")) {
        body = body.substr(1);
    }
    int max_q = -1;
    std::size_t start = 0;
    while (start <= body.size()) {
        const std::size_t star = body.find('*', start);
        std::string factor = body.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && (factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && all_digits(factor)) {
            max_q = std::max(max_q, std::stoi(factor));
        }
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return max_q;
}

int all_qubits(const std::vector<StimNode>& nodes) {
    int max_q = -1;
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            max_q = std::max(max_q, all_qubits((*block)->body) - 1);
            continue;
        }
        const auto& inst = std::get<StimInstruction>(node);
        if (inst.op == "DETECTOR" || inst.op == "OBSERVABLE_INCLUDE" || inst.op == "TICK" || inst.op == "SHIFT_COORDS") {
            continue;
        }
        for (const auto& target : inst.targets) {
            max_q = std::max(max_q, target_max_qubit(target));
        }
    }
    return max_q + 1;
}

PauliString pauli_on_target(int n, char axis, int q) {
    if (axis == 'X') {
        return pauli_x(n, q);
    }
    if (axis == 'Y') {
        return pauli_y(n, q);
    }
    if (axis == 'Z') {
        return pauli_z(n, q);
    }
    fail("unsupported Pauli axis");
}

std::pair<PauliString, bool> stim_mpp_pauli(int n, const std::string& target) {
    PauliString out = pauli_identity(n);
    bool inverted = false;
    std::size_t start = 0;
    while (start <= target.size()) {
        const std::size_t star = target.find('*', start);
        std::string factor = target.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            inverted = !inverted;
            factor = factor.substr(1);
        }
        if (factor.size() < 2 || !(factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            fail("invalid MPP factor");
        }
        const int q = std::stoi(factor.substr(1));
        out = out * pauli_on_target(n, factor[0], q);
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return {out, inverted};
}

std::vector<std::string> stim_mpp_targets(const StimInstruction& inst) {
    std::vector<std::string> groups;
    std::string current;
    for (const auto& target : inst.targets) {
        if (target == "*") {
            if (current.empty() || ends_with(current, "*")) {
                fail("misplaced MPP combiner");
            }
            current += "*";
        } else if (current.empty()) {
            current = target;
        } else if (ends_with(current, "*")) {
            current += target;
        } else {
            groups.push_back(current);
            current = target;
        }
    }
    if (!current.empty()) {
        if (ends_with(current, "*")) {
            fail("dangling MPP combiner");
        }
        groups.push_back(current);
    }
    return groups;
}

std::pair<int, int> reserve_measurement_record(std::vector<SymbolicBool>& records, SymbolicContext& context) {
    const int condition = context.fresh_condition();
    records.push_back(symbolic_bool(condition));
    return {static_cast<int>(records.size()), condition};
}

SymbolicBool measurement_sign_with_readout_error(SymbolicContext& context, bool inverted, double probability) {
    SymbolicBool sign(inverted);
    if (probability != 0.0) {
        sign = xor_bool(sign, context.fresh_bernoulli_bool(probability));
    }
    return sign;
}

std::vector<double> coords_with_shift(const std::vector<double>& coords, const std::vector<double>& shift) {
    std::vector<double> out(std::max(coords.size(), shift.size()), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (i < coords.size()) {
            out[i] += coords[i];
        }
        if (i < shift.size()) {
            out[i] += shift[i];
        }
    }
    return out;
}

struct StimAccumulator {
    FrameFactoredState state;
    std::vector<SymbolicBool> records;
    std::vector<StimDetector> detectors;
    std::vector<StimObservableInclude> observables;
    std::vector<double> coord_shift;
};

void apply_depolarize1(FrameFactoredState& state, int q, double probability) {
    const std::vector<std::vector<std::uint64_t>> assignments{
        packed_bits({false, false}),
        packed_bits({false, true}),
        packed_bits({true, false}),
        packed_bits({true, true}),
    };
    const auto bits = state.context->fresh_categorical_bools(2, assignments, {1.0 - probability, probability / 3.0, probability / 3.0, probability / 3.0});
    apply_pauli(state, pauli_x(state.n, q), bits[0]);
    apply_pauli(state, pauli_z(state.n, q), bits[1]);
}

void apply_depolarize2(FrameFactoredState& state, int q1, int q2, double probability) {
    std::vector<std::vector<std::uint64_t>> assignments;
    assignments.push_back(packed_bits({false, false, false, false}));
    for (bool x1 : {false, true}) {
        for (bool z1 : {false, true}) {
            for (bool x2 : {false, true}) {
                for (bool z2 : {false, true}) {
                    if (x1 || z1 || x2 || z2) {
                        assignments.push_back(packed_bits({x1, z1, x2, z2}));
                    }
                }
            }
        }
    }
    std::vector<double> probabilities(16, probability / 15.0);
    probabilities[0] = 1.0 - probability;
    const auto bits = state.context->fresh_categorical_bools(4, assignments, probabilities);
    apply_pauli(state, pauli_x(state.n, q1), bits[0]);
    apply_pauli(state, pauli_z(state.n, q1), bits[1]);
    apply_pauli(state, pauli_x(state.n, q2), bits[2]);
    apply_pauli(state, pauli_z(state.n, q2), bits[3]);
}

void apply_record_feedback(FrameFactoredState& state, const std::vector<SymbolicBool>& records, const StimInstruction& inst) {
    if ((inst.targets.size() & 1u) != 0u) {
        fail("record-controlled feedback requires paired targets");
    }
    for (std::size_t i = 0; i < inst.targets.size(); i += 2) {
        const int record = stim_record_index(inst.targets[i], static_cast<int>(records.size()));
        auto [q, inverted] = stim_qubit_target(inst.targets[i + 1], inst.line);
        if (inverted) {
            fail("record-controlled feedback does not accept inverted qubit targets");
        }
        const PauliString pauli = (inst.op == "CX" || inst.op == "CNOT") ? pauli_x(state.n, q) : pauli_z(state.n, q);
        apply_pauli(state, pauli, records[static_cast<std::size_t>(record - 1)]);
    }
}

void apply_reset(FrameFactoredState& state, const PauliString& measurement_pauli, const PauliString& correction_pauli) {
    const int condition = state.context->fresh_condition();
    apply_pauli_measurement(state, measurement_pauli, SymbolicBool(false), std::nullopt, condition);
    apply_pauli(state, correction_pauli, symbolic_bool(condition));
}

void apply_measurement_reset(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    const PauliString& measurement_pauli,
    const PauliString& correction_pauli,
    bool inverted,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, measurement_pauli, sign, record, record_condition);
    apply_pauli(state, correction_pauli, xor_bool(symbolic_bool(record_condition), sign));
}

void apply_instruction(StimAccumulator& acc, const StimInstruction& inst) {
    FrameFactoredState& state = acc.state;
    const std::string& op = inst.op;
    if (op == "QUBIT_COORDS" || op == "TICK") {
        return;
    }
    if (op == "SHIFT_COORDS") {
        if (acc.coord_shift.size() < inst.parens.size()) {
            acc.coord_shift.resize(inst.parens.size(), 0.0);
        }
        for (std::size_t i = 0; i < inst.parens.size(); ++i) {
            acc.coord_shift[i] += inst.parens[i];
        }
        return;
    }
    if (op == "DETECTOR") {
        acc.detectors.push_back({stim_record_indices(inst, acc.records), coords_with_shift(inst.parens, acc.coord_shift), inst.line});
        return;
    }
    if (op == "OBSERVABLE_INCLUDE") {
        if (inst.parens.size() != 1 || inst.parens[0] < 0.0) {
            fail("OBSERVABLE_INCLUDE expects one nonnegative observable index");
        }
        acc.observables.push_back({static_cast<int>(std::llround(inst.parens[0])), stim_record_indices(inst, acc.records), inst.line});
        return;
    }
    if ((op == "CX" || op == "CNOT" || op == "CZ") && !inst.targets.empty() && starts_with(inst.targets[0], "rec[")) {
        apply_record_feedback(state, acc.records, inst);
        return;
    }
    if (op == "CX" || op == "CNOT") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("CX requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_CX(state, qs[i], qs[i + 1]);
        }
    } else if (op == "CZ") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("CZ requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_CZ(state, qs[i], qs[i + 1]);
        }
    } else if (op == "SWAP") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("SWAP requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_SWAP(state, qs[i], qs[i + 1]);
        }
    } else if (op == "H") {
        for (int q : stim_qubit_targets(inst)) {
            left_H(state, q);
        }
    } else if (op == "S" || op == "SQRT_Z") {
        for (int q : stim_qubit_targets(inst)) {
            left_S(state, q);
        }
    } else if (op == "S_DAG" || op == "SQRT_Z_DAG") {
        for (int q : stim_qubit_targets(inst)) {
            left_SDG(state, q);
        }
    } else if (op == "X" || op == "Y" || op == "Z") {
        for (int q : stim_qubit_targets(inst)) {
            if (op == "X") {
                left_X(state, q);
            } else if (op == "Z") {
                left_Z(state, q);
            } else {
                left_X(state, q);
                left_Z(state, q);
            }
        }
    } else if (op == "T" || op == "T_DAG") {
        const double theta = (op == "T") ? kPi / 8.0 : -kPi / 8.0;
        for (int q : stim_qubit_targets(inst)) {
            apply_pauli_rotation(state, pauli_z(state.n, q), theta);
        }
    } else if (op == "M" || op == "MZ" || op == "MX" || op == "MY") {
        const double probability = stim_paren_probability(inst);
        for (auto [q, inverted] : stim_measurement_targets(inst)) {
            const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            const PauliString pauli = op == "MX" ? pauli_x(state.n, q) : (op == "MY" ? pauli_y(state.n, q) : pauli_z(state.n, q));
            apply_pauli_measurement(state, pauli, sign, record, record_condition);
        }
    } else if (op == "MR" || op == "MRZ" || op == "MRX" || op == "MRY") {
        const double probability = stim_paren_probability(inst);
        for (auto [q, inverted] : stim_measurement_targets(inst)) {
            if (op == "MRX") {
                apply_measurement_reset(state, acc.records, pauli_x(state.n, q), pauli_z(state.n, q), inverted, probability);
            } else if (op == "MRY") {
                apply_measurement_reset(state, acc.records, pauli_y(state.n, q), pauli_x(state.n, q), inverted, probability);
            } else {
                apply_measurement_reset(state, acc.records, pauli_z(state.n, q), pauli_x(state.n, q), inverted, probability);
            }
        }
    } else if (op == "R" || op == "RZ" || op == "RX" || op == "RY") {
        for (int q : stim_qubit_targets(inst)) {
            if (op == "RX") {
                apply_reset(state, pauli_x(state.n, q), pauli_z(state.n, q));
            } else if (op == "RY") {
                apply_reset(state, pauli_y(state.n, q), pauli_x(state.n, q));
            } else {
                apply_reset(state, pauli_z(state.n, q), pauli_x(state.n, q));
            }
        }
    } else if (op == "MPP") {
        const double probability = stim_paren_probability(inst);
        for (const auto& target : stim_mpp_targets(inst)) {
            auto [pauli, inverted] = stim_mpp_pauli(state.n, target);
            const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            apply_pauli_measurement(state, pauli, sign, record, record_condition);
        }
    } else if (op == "X_ERROR" || op == "Y_ERROR" || op == "Z_ERROR") {
        const double probability = stim_required_probability(inst);
        for (int q : stim_qubit_targets(inst)) {
            const PauliString pauli = op == "X_ERROR" ? pauli_x(state.n, q) : (op == "Y_ERROR" ? pauli_y(state.n, q) : pauli_z(state.n, q));
            apply_pauli(state, pauli, state.context->fresh_bernoulli_bool(probability));
        }
    } else if (op == "DEPOLARIZE1") {
        const double probability = stim_required_probability(inst);
        for (int q : stim_qubit_targets(inst)) {
            apply_depolarize1(state, q, probability);
        }
    } else if (op == "DEPOLARIZE2") {
        const double probability = stim_required_probability(inst);
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("DEPOLARIZE2 requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            apply_depolarize2(state, qs[i], qs[i + 1], probability);
        }
    } else {
        fail("unsupported Stim operation: " + op);
    }
}

void apply_nodes(StimAccumulator& acc, const std::vector<StimNode>& nodes) {
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            for (int i = 0; i < (*block)->count; ++i) {
                apply_nodes(acc, (*block)->body);
            }
        } else {
            apply_instruction(acc, std::get<StimInstruction>(node));
        }
    }
}

} // namespace

StimParseResult parse_stim_lines(const std::vector<std::string>& lines) {
    std::size_t pos = 0;
    const auto nodes = parse_nodes(lines, pos, false);
    const int n = all_qubits(nodes);
    StimAccumulator acc{FrameFactoredState(n, 0), {}, {}, {}, {}};
    apply_nodes(acc, nodes);
    return StimParseResult{acc.state, acc.records, acc.detectors, acc.observables};
}

StimParseResult parse_stim_text(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return parse_stim_lines(lines);
}

StimParseResult parse_stim_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        fail("failed to open Stim file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return parse_stim_lines(lines);
}

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed) {
    PendingFactoredState pending(parsed.state);
    const FactoredInstructionProgram program = plan_factored_updates(pending);
    const auto records = sample_measurements(program, shots, seed);
    StimSampleSummary summary;
    summary.shots = shots;
    for (const auto& shot_records : records) {
        bool discarded = false;
        for (const auto& detector : parsed.detectors) {
            bool parity = false;
            for (int record : detector.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            discarded = discarded || parity;
        }
        if (discarded) {
            ++summary.discarded;
            continue;
        }
        ++summary.accepted;
        bool logical = false;
        for (const auto& observable : parsed.observables) {
            bool parity = false;
            for (int record : observable.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            logical ^= parity;
        }
        if (logical) {
            ++summary.logical_errors;
        }
    }
    return summary;
}

double discard_rate(const StimSampleSummary& summary) {
    return summary.shots == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : static_cast<double>(summary.discarded) / static_cast<double>(summary.shots);
}

double logical_error_rate(const StimSampleSummary& summary) {
    return summary.accepted == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : static_cast<double>(summary.logical_errors) / static_cast<double>(summary.accepted);
}

} // namespace symft
