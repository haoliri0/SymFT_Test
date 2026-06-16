#include "../core/symft_internal.hpp"

#include "frontend/stim.hpp"
#include "sampler/single_shot.hpp"

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

std::vector<int> stim_record_indices(const StimInstruction& inst, int nrecords) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_record_index(target, nrecords));
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

struct StimCircuitBuilder {
    QuantumCircuit circuit;
    std::vector<double> coord_shift;
};

CircuitInstruction new_circuit_instruction(CircuitInstructionKind kind, int line) {
    CircuitInstruction instruction;
    instruction.kind = kind;
    instruction.line = line;
    return instruction;
}

std::vector<CircuitMeasurementTarget> circuit_measurement_targets(const StimInstruction& inst) {
    std::vector<CircuitMeasurementTarget> out;
    for (auto [q, inverted] : stim_measurement_targets(inst)) {
        out.push_back({q, inverted});
    }
    return out;
}

std::vector<CircuitPauliProduct> circuit_mpp_targets(int n, const StimInstruction& inst) {
    std::vector<CircuitPauliProduct> out;
    for (const auto& target : stim_mpp_targets(inst)) {
        auto [pauli, inverted] = stim_mpp_pauli(n, target);
        out.push_back({std::move(pauli), inverted});
    }
    return out;
}

void append_record_feedback(StimCircuitBuilder& builder, const StimInstruction& inst, CircuitInstructionKind kind) {
    if ((inst.targets.size() & 1u) != 0u) {
        fail("record-controlled feedback requires paired targets");
    }
    CircuitInstruction instruction = new_circuit_instruction(kind, inst.line);
    for (std::size_t i = 0; i < inst.targets.size(); i += 2) {
        const int record = stim_record_index(inst.targets[i], builder.circuit.nrecords);
        auto [q, inverted] = stim_qubit_target(inst.targets[i + 1], inst.line);
        if (inverted) {
            fail("record-controlled feedback does not accept inverted qubit targets");
        }
        instruction.feedback_targets.push_back({record, q});
    }
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_qubit_instruction(StimCircuitBuilder& builder, const StimInstruction& inst, CircuitInstructionKind kind) {
    CircuitInstruction instruction = new_circuit_instruction(kind, inst.line);
    instruction.qubits = stim_qubit_targets(inst);
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_measurement_instruction(StimCircuitBuilder& builder, const StimInstruction& inst, CircuitInstructionKind kind) {
    CircuitInstruction instruction = new_circuit_instruction(kind, inst.line);
    instruction.probability = stim_paren_probability(inst);
    instruction.measurement_targets = circuit_measurement_targets(inst);
    builder.circuit.nrecords += static_cast<int>(instruction.measurement_targets.size());
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_instruction(StimCircuitBuilder& builder, const StimInstruction& inst) {
    const std::string& op = inst.op;
    if (op == "QUBIT_COORDS") {
        return;
    }
    if (op == "TICK") {
        builder.circuit.instructions.push_back(new_circuit_instruction(CircuitInstructionKind::Tick, inst.line));
        return;
    }
    if (op == "SHIFT_COORDS") {
        if (builder.coord_shift.size() < inst.parens.size()) {
            builder.coord_shift.resize(inst.parens.size(), 0.0);
        }
        for (std::size_t i = 0; i < inst.parens.size(); ++i) {
            builder.coord_shift[i] += inst.parens[i];
        }
        return;
    }
    if (op == "DETECTOR") {
        builder.circuit.detectors.push_back(
            {stim_record_indices(inst, builder.circuit.nrecords),
             coords_with_shift(inst.parens, builder.coord_shift),
             inst.line});
        return;
    }
    if (op == "OBSERVABLE_INCLUDE") {
        if (inst.parens.size() != 1 || inst.parens[0] < 0.0) {
            fail("OBSERVABLE_INCLUDE expects one nonnegative observable index");
        }
        builder.circuit.observables.push_back(
            {static_cast<int>(std::llround(inst.parens[0])),
             stim_record_indices(inst, builder.circuit.nrecords),
             inst.line});
        return;
    }
    if ((op == "CX" || op == "CNOT" || op == "CY" || op == "CZ") && !inst.targets.empty() && starts_with(inst.targets[0], "rec[")) {
        const CircuitInstructionKind kind = (op == "CX" || op == "CNOT")
                                                ? CircuitInstructionKind::FeedbackX
                                                : (op == "CY" ? CircuitInstructionKind::FeedbackY
                                                              : CircuitInstructionKind::FeedbackZ);
        append_record_feedback(builder, inst, kind);
        return;
    }
    if (op == "CX" || op == "CNOT") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::CX);
    } else if (op == "CY") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::CY);
    } else if (op == "CZ") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::CZ);
    } else if (op == "SWAP") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::SWAP);
    } else if (op == "H") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::H);
    } else if (op == "S" || op == "SQRT_Z") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::S);
    } else if (op == "S_DAG" || op == "SQRT_Z_DAG") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::SDG);
    } else if (op == "X") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::X);
    } else if (op == "Y") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::Y);
    } else if (op == "Z") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::Z);
    } else if (op == "T") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::T);
    } else if (op == "T_DAG") {
        append_qubit_instruction(builder, inst, CircuitInstructionKind::TDag);
    } else if (op == "M" || op == "MZ" || op == "MX" || op == "MY") {
        append_measurement_instruction(
            builder,
            inst,
            op == "MX" ? CircuitInstructionKind::MX : (op == "MY" ? CircuitInstructionKind::MY : CircuitInstructionKind::MZ));
    } else if (op == "MR" || op == "MRZ" || op == "MRX" || op == "MRY") {
        append_measurement_instruction(
            builder,
            inst,
            op == "MRX" ? CircuitInstructionKind::MRX : (op == "MRY" ? CircuitInstructionKind::MRY : CircuitInstructionKind::MRZ));
    } else if (op == "R" || op == "RZ" || op == "RX" || op == "RY") {
        append_qubit_instruction(
            builder,
            inst,
            op == "RX" ? CircuitInstructionKind::RX : (op == "RY" ? CircuitInstructionKind::RY : CircuitInstructionKind::RZ));
    } else if (op == "MPP") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::MPP, inst.line);
        instruction.probability = stim_paren_probability(inst);
        instruction.pauli_products = circuit_mpp_targets(builder.circuit.nqubits, inst);
        builder.circuit.nrecords += static_cast<int>(instruction.pauli_products.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "X_ERROR" || op == "Y_ERROR" || op == "Z_ERROR") {
        CircuitInstruction instruction = new_circuit_instruction(
            op == "X_ERROR" ? CircuitInstructionKind::XError
                             : (op == "Y_ERROR" ? CircuitInstructionKind::YError : CircuitInstructionKind::ZError),
            inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "DEPOLARIZE1") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::Depolarize1, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "DEPOLARIZE2") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::Depolarize2, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else {
        fail("unsupported Stim operation: " + op);
    }
}

void append_nodes(StimCircuitBuilder& builder, const std::vector<StimNode>& nodes) {
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            for (int i = 0; i < (*block)->count; ++i) {
                append_nodes(builder, (*block)->body);
            }
        } else {
            append_instruction(builder, std::get<StimInstruction>(node));
        }
    }
}

} // namespace

QuantumCircuit parse_stim_circuit_lines(const std::vector<std::string>& lines) {
    std::size_t pos = 0;
    const auto nodes = parse_nodes(lines, pos, false);
    StimCircuitBuilder builder;
    builder.circuit.nqubits = all_qubits(nodes);
    append_nodes(builder, nodes);
    return builder.circuit;
}

QuantumCircuit parse_stim_circuit_text(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return parse_stim_circuit_lines(lines);
}

QuantumCircuit parse_stim_circuit_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        fail("failed to open Stim file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return parse_stim_circuit_lines(lines);
}

StimParseResult parse_stim_lines(const std::vector<std::string>& lines) {
    const QuantumCircuit circuit = parse_stim_circuit_lines(lines);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        circuit.detectors,
        circuit.observables};
}

StimParseResult parse_stim_text(const std::string& text) {
    const QuantumCircuit circuit = parse_stim_circuit_text(text);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        circuit.detectors,
        circuit.observables};
}

StimParseResult parse_stim_file(const std::string& path) {
    const QuantumCircuit circuit = parse_stim_circuit_file(path);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        circuit.detectors,
        circuit.observables};
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
