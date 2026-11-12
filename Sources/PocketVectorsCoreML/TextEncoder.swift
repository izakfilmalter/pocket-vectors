import CoreML
import Foundation

/// A sentence-embedding model (BERT family: bge, MiniLM, e5, gte …) converted to Core ML, run on the
/// Neural Engine by default. Tokenises in Swift, makes one `predict` call, returns a unit-length vector.
///
/// The model must take token ids (`Int32`, shape `[1, L]`) for one or more fixed lengths `L`, and
/// output the pooled embedding. It may also take an attention mask (set `attentionMaskName`). Pooling
/// and normalisation are usually folded into the model at conversion; the vector is re-normalised here.
///
/// Thread-safe: calls are serialised (one model, reused input buffers).
public final class CoreMLTextEncoder: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var inputName = "input_ids"
        public var attentionMaskName: String? = nil
        public var outputName = "embedding"
        /// Fixed input lengths the model was converted for, shortest first; a query uses the
        /// shortest that fits and is truncated to the longest.
        public var lengths = [32, 64, 128]
        /// Prepended to every text, e.g. bge's "Represent this sentence for searching relevant passages: ".
        public var prefix = ""
        /// `.cpuAndNeuralEngine` by default. Not `.all`: on a Mac, Core ML may then plan the whole model
        /// for the GPU. Check where it runs with `ComputePlanReport`.
        public var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
        public init() {}
    }

    public let configuration: Configuration
    public let tokenizer: WordPieceTokenizer
    private let model: MLModel
    private var ids: [Int: MLMultiArray] = [:]
    private var masks: [Int: MLMultiArray] = [:]
    private let lock = NSLock()

    /// `modelURL`: a compiled `.mlmodelc` (what Xcode puts in an app bundle), or an `.mlpackage`
    /// (compiled here once, slowly: tools only).
    public init(modelURL: URL, tokenizer: WordPieceTokenizer, configuration: Configuration = Configuration()) throws {
        precondition(!configuration.lengths.isEmpty, "at least one input length")
        let cfg = MLModelConfiguration()
        cfg.computeUnits = configuration.computeUnits
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
        model = try MLModel(contentsOf: compiled, configuration: cfg)
        self.tokenizer = tokenizer
        self.configuration = configuration
        for l in configuration.lengths {
            ids[l] = try MLMultiArray(shape: [1, NSNumber(value: l)], dataType: .int32)
            if configuration.attentionMaskName != nil {
                masks[l] = try MLMultiArray(shape: [1, NSNumber(value: l)], dataType: .int32)
            }
        }
    }

    /// The token ids a text becomes ([CLS] … [SEP], truncated to the longest input length).
    public func tokens(_ text: String) -> [Int32] {
        tokenizer.encode(configuration.prefix + text, maxLength: configuration.lengths.last!)
    }

    /// A unit-length embedding of `text`.
    public func embed(_ text: String) throws -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let t = tokens(text)
        let len = configuration.lengths.first { $0 >= t.count } ?? configuration.lengths.last!
        let arr = ids[len]!
        let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: len)
        for i in 0..<len { p[i] = i < t.count ? t[i] : tokenizer.pad }
        var inputs: [String: Any] = [configuration.inputName: arr]
        if let name = configuration.attentionMaskName, let m = masks[len] {
            let mp = m.dataPointer.bindMemory(to: Int32.self, capacity: len)
            for i in 0..<len { mp[i] = i < t.count ? 1 : 0 }
            inputs[name] = m
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        guard let e = out.featureValue(for: configuration.outputName)?.multiArrayValue else {
            throw EncoderError.missingOutput(configuration.outputName)
        }
        let n = e.count
        var v = [Float](repeating: 0, count: n)
        switch e.dataType {
        case .float16:
            let src = e.dataPointer.bindMemory(to: Float16.self, capacity: n)
            for i in 0..<n { v[i] = Float(src[i]) }
        case .float32:
            let src = e.dataPointer.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { v[i] = src[i] }
        default:
            for i in 0..<n { v[i] = e[i].floatValue }
        }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return norm > 0 ? v.map { $0 / norm } : v
    }
}

public enum EncoderError: Error {
    case missingOutput(String)
    case notAnMLProgram
}
