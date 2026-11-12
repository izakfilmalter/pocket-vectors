import Accelerate
import CPocketVectors
import Foundation

/// One search result: the row (the vector's position in the index) and its similarity to the query.
public struct SearchResult: Sendable, Hashable {
    /// Position of the vector in the index (the order it was written in).
    public let row: Int
    /// Dot product with the query; cosine similarity when both are unit length (approximate for
    /// fp16 and int8 storage).
    public let score: Float
    public init(row: Int, score: Float) { self.row = row; self.score = score }
}

public enum FlatIndexError: Error, Equatable, CustomStringConvertible {
    case invalidDimension(Int)
    case wrongVectorLength(row: Int, expected: Int, got: Int)
    case notAnIndexFile
    case unsupportedVersion(UInt32)
    case corrupt(String)
    case empty

    public var description: String {
        switch self {
        case .invalidDimension(let d): return "dimension \(d) is not a positive multiple of 16"
        case .wrongVectorLength(let r, let e, let g): return "vector \(r) has \(g) values, expected \(e)"
        case .notAnIndexFile: return "not a pocket-vectors index file"
        case .unsupportedVersion(let v): return "unsupported index file version \(v)"
        case .corrupt(let m): return "corrupt index file: \(m)"
        case .empty: return "no vectors"
        }
    }
}

/// Exact brute-force nearest-neighbour search over a contiguous matrix of vectors, by dot product.
///
/// - `.int8` (the default) stores each row as int8 with its own scale (max-abs quantisation): 4×
///   smaller than fp32, scored with the ARMv8.2 `sdot` instruction. On 384-d text embeddings it keeps
///   recall@10 ≈ 0.98 against exact fp32 search.
/// - `.float16` stores rows as IEEE half (2× smaller, recall ≈ 1.0); `.float32` is exact (Accelerate).
///
/// Store unit-length vectors to search by cosine similarity. An index is immutable and `Sendable`:
/// any number of threads may search it at once (each search has its own score buffer). Opened from a
/// file it is memory-mapped, so opening is O(1) and the vectors' clean pages don't count against an
/// iOS app's memory limit; call `prefault()` off the main thread to page them in before the first query.
public final class FlatIndex: Sendable {
    public enum Format: UInt32, Sendable, CaseIterable {
        case float32 = 0, float16 = 1, int8 = 2
        var width: Int { switch self { case .float32: 4; case .float16: 2; case .int8: 1 } }
    }

    public let format: Format
    public let dimension: Int
    public let count: Int
    /// The bytes holding vectors (and int8 scales): what the index costs on disk, or in memory when
    /// it was built in memory or opened with `mapped: false`.
    public var byteCount: Int { data.count + (scales?.count ?? 0) }

    // Opened from a file, `data` and `scales` are slices of its (mapped) bytes.
    private let data: Data
    private let scales: Data?

    // MARK: building

    /// Builds an index in memory from `vectors` (each `dimension` long; `dimension % 16 == 0`).
    public convenience init(vectors: [[Float]], format: Format = .int8) throws {
        guard let first = vectors.first else { throw FlatIndexError.empty }
        let (data, scales) = try Self.encode(vectors, dimension: first.count, format: format)
        try self.init(format: format, dimension: first.count, count: vectors.count, data: data, scales: scales)
    }

    /// Opens an index file written by `write(_:format:to:)`. `mapped` (default) memory-maps it;
    /// `false` reads it into memory.
    public convenience init(contentsOf url: URL, mapped: Bool = true) throws {
        let file = try Data(contentsOf: url, options: mapped ? [.alwaysMapped] : [])
        let h = try IndexFileHeader(file)
        // slices share the mapping (no copy); offsets are 64-byte aligned, so rows load aligned
        let s = file.startIndex
        let data = file[(s + h.dataOffset)..<(s + h.dataOffset + h.count * h.dimension * h.format.width)]
        let scales = h.format == .int8 ? file[(s + h.scalesOffset)..<(s + h.scalesOffset + h.count * 4)] : nil
        try self.init(format: h.format, dimension: h.dimension, count: h.count, data: data, scales: scales)
    }

    init(format: Format, dimension: Int, count: Int, data: Data, scales: Data?) throws {
        guard dimension > 0, dimension % 16 == 0 else { throw FlatIndexError.invalidDimension(dimension) }
        guard count > 0 else { throw FlatIndexError.empty }
        guard data.count == count * dimension * format.width else {
            throw FlatIndexError.corrupt("\(data.count) bytes of vectors for \(count) × \(dimension) × \(format.width)")
        }
        if format == .int8 { guard scales?.count == count * 4 else { throw FlatIndexError.corrupt("int8 scales missing") } }
        self.format = format
        self.dimension = dimension
        self.count = count
        self.data = data
        self.scales = scales
    }

    /// Writes `vectors` as an index file (one file: header, int8 scales, rows; see IndexFile.swift).
    public static func write(_ vectors: [[Float]], format: Format = .int8, to url: URL) throws {
        guard let first = vectors.first else { throw FlatIndexError.empty }
        let (data, scales) = try encode(vectors, dimension: first.count, format: format)
        let file = IndexFileHeader.file(format: format, dimension: first.count, count: vectors.count, data: data, scales: scales)
        try file.write(to: url, options: .atomic)
    }

    static func encode(_ vectors: [[Float]], dimension: Int, format: Format) throws -> (Data, Data?) {
        guard dimension > 0, dimension % 16 == 0 else { throw FlatIndexError.invalidDimension(dimension) }
        for (i, v) in vectors.enumerated() where v.count != dimension {
            throw FlatIndexError.wrongVectorLength(row: i, expected: dimension, got: v.count)
        }
        switch format {
        case .float32:
            return (vectors.flatMap { $0 }.withUnsafeBufferPointer { Data(buffer: $0) }, nil)
        case .float16:
            return (vectors.flatMap { $0.map { Float16($0) } }.withUnsafeBufferPointer { Data(buffer: $0) }, nil)
        case .int8:
            var rows = [Int8](), sc = [Float]()
            rows.reserveCapacity(vectors.count * dimension)
            sc.reserveCapacity(vectors.count)
            for v in vectors {
                let (q, s) = Quantizer.int8(v)
                rows += q
                sc.append(s)
            }
            return (rows.withUnsafeBufferPointer { Data(buffer: $0) }, sc.withUnsafeBufferPointer { Data(buffer: $0) })
        }
    }

    // MARK: reading

    /// Touches every page of a memory-mapped index so the first query doesn't pay the page-ins.
    /// Call it off the main thread at launch. Returns a checksum so the reads aren't optimised away.
    @discardableResult
    public func prefault() -> Int {
        var sum = 0
        for d in [data, scales ?? Data()] {
            d.withUnsafeBytes { p in
                var i = 0
                while i < p.count { sum &+= Int(p[i]); i += 4096 }
            }
        }
        return sum
    }

    /// Row `row` as floats (int8 rows dequantised with their scale).
    public func vector(row: Int) -> [Float] {
        precondition(row >= 0 && row < count, "row \(row) out of 0..<\(count)")
        return data.withUnsafeBytes { m in
            switch format {
            case .float32: return Array(m.bindMemory(to: Float.self)[(row * dimension)..<((row + 1) * dimension)])
            case .float16: return m.bindMemory(to: Float16.self)[(row * dimension)..<((row + 1) * dimension)].map { Float($0) }
            case .int8:
                let s = scales!.withUnsafeBytes { $0.load(fromByteOffset: row * 4, as: Float.self) }
                return m.bindMemory(to: Int8.self)[(row * dimension)..<((row + 1) * dimension)].map { Float($0) * s }
            }
        }
    }

    // MARK: search

    /// The `k` rows (at most 1,024) with the highest dot product with `query`, best first. With a
    /// `filter`, only rows it allows: a row range scans just that slice; a mask or predicate scores
    /// every row and drops the rest (exact, unlike filtered graph search, at the same cost as no filter).
    public func search(_ query: [Float], k: Int, filter: RowFilter? = nil) -> [SearchResult] {
        precondition(query.count == dimension, "query has \(query.count) values, index dimension is \(dimension)")
        precondition(k >= 0, "k must not be negative")
        let range: Range<Int>
        if case .range(let r)? = filter?.kind { range = r.clamped(to: 0..<count) } else { range = 0..<count }
        guard k > 0, !range.isEmpty else { return [] }
        var scores = [Float](repeating: 0, count: range.count)
        scores.withUnsafeMutableBufferPointer { out in score(query, rows: range, into: out.baseAddress!) }
        switch filter?.kind {
        case .mask(let bits)?:
            for i in 0..<range.count where !bits.contains(range.lowerBound + i) { scores[i] = -.infinity }
        case .predicate(let keep)?:
            for i in 0..<range.count where !keep(range.lowerBound + i) { scores[i] = -.infinity }
        default: break
        }
        let kk = min(k, 1024, range.count)
        var idx = [UInt32](repeating: 0, count: kk)
        var sc = [Float](repeating: 0, count: kk)
        let n = pv_topk(scores, scores.count, kk, &idx, &sc)
        return (0..<n).compactMap { i in
            sc[i] == -.infinity ? nil : SearchResult(row: range.lowerBound + Int(idx[i]), score: sc[i])
        }
    }

    /// Scores rows `rows` against `query` into `out` (rows.count floats).
    func score(_ query: [Float], rows: Range<Int>, into out: UnsafeMutablePointer<Float>) {
        let n = rows.count
        switch format {
        case .float32:
            data.withUnsafeBytes { m in
                let base = m.bindMemory(to: Float.self).baseAddress! + rows.lowerBound * dimension
                // out (n × 1) = M (n × dim, row-major) · q (dim × 1)
                vDSP_mmul(base, 1, query, 1, out, 1, vDSP_Length(n), 1, vDSP_Length(dimension))
            }
        case .float16:
            data.withUnsafeBytes { m in
                pv_dot_f16(query, m.bindMemory(to: UInt16.self).baseAddress! + rows.lowerBound * dimension, n, dimension, out)
            }
        case .int8:
            let (q8, qs) = Quantizer.int8(query)
            data.withUnsafeBytes { m in
                scales!.withUnsafeBytes { s in
                    pv_dot_i8(q8, qs, m.bindMemory(to: Int8.self).baseAddress! + rows.lowerBound * dimension,
                              s.bindMemory(to: Float.self).baseAddress! + rows.lowerBound, n, dimension, out)
                }
            }
        }
    }
}

/// Which rows a search may return.
public struct RowFilter: Sendable {
    enum Kind: Sendable {
        case range(Range<Int>)
        case mask(RowSet)
        case predicate(@Sendable (Int) -> Bool)
    }
    let kind: Kind

    /// Rows in `range` only. The fastest filter: only that slice is scanned. Write rows grouped by
    /// the field you filter on most (a collection, a language, a date bucket) to use it.
    public static func rows(_ range: Range<Int>) -> RowFilter { RowFilter(kind: .range(range)) }
    /// Rows in a precomputed set (build it once per filter value and reuse it).
    public static func rows(in set: RowSet) -> RowFilter { RowFilter(kind: .mask(set)) }
    /// Rows for which `keep(row)` is true (called once per row per search).
    public static func `where`(_ keep: @escaping @Sendable (Int) -> Bool) -> RowFilter { RowFilter(kind: .predicate(keep)) }
}

/// A set of rows as a bitmap: `count / 8` bytes, O(1) membership.
public struct RowSet: Sendable, Hashable {
    private var words: [UInt64]
    public let capacity: Int

    public init(capacity: Int) {
        self.capacity = capacity
        words = [UInt64](repeating: 0, count: (capacity + 63) / 64)
    }

    public init<S: Sequence>(capacity: Int, rows: S) where S.Element == Int {
        self.init(capacity: capacity)
        for r in rows { insert(r) }
    }

    public mutating func insert(_ row: Int) {
        precondition(row >= 0 && row < capacity, "row \(row) out of 0..<\(capacity)")
        words[row >> 6] |= 1 << UInt64(row & 63)
    }

    public func contains(_ row: Int) -> Bool {
        row >= 0 && row < capacity && words[row >> 6] & (1 << UInt64(row & 63)) != 0
    }

    public var count: Int { words.reduce(0) { $0 + $1.nonzeroBitCount } }
}

/// int8 quantisation as the index stores it: per vector, scale so the largest magnitude maps to 127.
public enum Quantizer {
    /// (values, scale) with value ≈ int8 × scale.
    public static func int8(_ v: [Float]) -> ([Int8], Float) {
        var m: Float = 0
        vDSP_maxmgv(v, 1, &m, vDSP_Length(v.count))
        guard m > 0, m.isFinite else { return ([Int8](repeating: 0, count: v.count), 1) }
        let s = 127 / m
        return (v.map { Int8(max(-127, min(127, ($0 * s).rounded()))) }, 1 / s)
    }
}
