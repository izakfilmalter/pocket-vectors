import CoreML
import Foundation

/// Where Core ML will run a model's ops, from `MLComputePlan`: per op, the device it dispatches to
/// (preferred) and the devices that could run it (supported), weighted by Core ML's cost estimate.
/// Use it to check that an encoder really runs on the Neural Engine, not just that you asked for it.
/// Confirm at run time with an Instruments trace (the "Neural Engine" instrument shows prediction
/// intervals) and by timing `.cpuAndNeuralEngine` against `.cpuOnly`.
public struct ComputePlanReport: Sendable, CustomStringConvertible {
    public let computeUnits: MLComputeUnits
    /// Ops that have a device (constants and the like have none).
    public let operations: Int
    public let preferredNeuralEngine: Int
    public let preferredCPU: Int
    public let preferredGPU: Int
    public let neuralEngineSupported: Int
    /// Core ML's estimated cost of the ops preferred on the Neural Engine, over all ops (0…1).
    public let neuralEngineCostShare: Double
    /// Operator name → count, for ops not planned on the Neural Engine.
    public let offNeuralEngine: [String: Int]

    public var description: String {
        let off = offNeuralEngine.sorted { $0.value > $1.value }.map { "\($0.key) ×\($0.value)" }.joined(separator: ", ")
        return "\(operations) ops: \(preferredNeuralEngine) Neural Engine, \(preferredCPU) CPU, \(preferredGPU) GPU; "
            + String(format: "%.1f %% of estimated cost on the Neural Engine", neuralEngineCostShare * 100)
            + (off.isEmpty ? "" : "; not on the Neural Engine: \(off)")
    }

    /// The plan for a compiled model (`.mlmodelc`) under `computeUnits`.
    public static func load(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> ComputePlanReport {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: cfg)
        guard case let .program(program) = plan.modelStructure else { throw EncoderError.notAnMLProgram }
        var n = 0, ane = 0, cpu = 0, gpu = 0, sup = 0
        var cost = 0.0, costANE = 0.0
        var off: [String: Int] = [:]
        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                for b in op.blocks { walk(b) }
                guard let u = plan.deviceUsage(for: op) else { continue }
                n += 1
                let w = plan.estimatedCost(of: op)?.weight ?? 0
                cost += w
                switch u.preferred {
                case .neuralEngine: ane += 1; costANE += w
                case .cpu: cpu += 1; off[op.operatorName, default: 0] += 1
                case .gpu: gpu += 1; off[op.operatorName, default: 0] += 1
                @unknown default: off[op.operatorName, default: 0] += 1
                }
                if u.supported.contains(where: { if case .neuralEngine = $0 { true } else { false } }) { sup += 1 }
            }
        }
        for f in program.functions.values { walk(f.block) }
        return ComputePlanReport(computeUnits: computeUnits, operations: n, preferredNeuralEngine: ane, preferredCPU: cpu,
                                 preferredGPU: gpu, neuralEngineSupported: sup,
                                 neuralEngineCostShare: cost > 0 ? costANE / cost : 0, offNeuralEngine: off)
    }
}
