import CPocketVectors
import Foundation
import Testing
@testable import PocketVectors

/// Deterministic synthetic embeddings: unit vectors around `clusters` random centres (text embeddings
/// are clustered too; uniform random vectors are the worst case for any quantiser).
struct Synthetic {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func gaussian() -> Float {
        let u1 = max(Double(next() >> 11) / Double(1 << 53), 1e-12), u2 = Double(next() >> 11) / Double(1 << 53)
        return Float(sqrt(-2 * log(u1)) * cos(2 * .pi * u2))
    }
    static func unit(_ v: [Float]) -> [Float] {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return v.map { $0 / n }
    }
    mutating func vectors(_ n: Int, dim: Int, clusters: Int = 64, spread: Float = 0.6) -> [[Float]] {
        let centres = (0..<clusters).map { _ in Self.unit((0..<dim).map { _ in gaussian() }) }
        return (0..<n).map { _ in
            let c = centres[Int(next() % UInt64(clusters))]
            return Self.unit(c.enumerated().map { $0.element + spread * gaussian() / Float(dim).squareRoot() })
        }
    }
}

func exactTop(_ vectors: [[Float]], _ q: [Float], k: Int, allow: (Int) -> Bool = { _ in true }) -> [Int] {
    vectors.indices.filter(allow)
        .map { ($0, zip(vectors[$0], q).reduce(Float(0)) { $0 + $1.0 * $1.1 }) }
        .sorted { $0.1 > $1.1 }.prefix(k).map(\.0)
}

@Suite struct KernelTests {
    @Test func int8KernelsMatchScalarReference() {
        var g = Synthetic(seed: 1)
        let dim = 384, n = 257
        let rows = (0..<(n * dim)).map { _ in Int8(truncatingIfNeeded: Int(g.next() % 255) - 127) }
        let q = (0..<dim).map { _ in Int8(truncatingIfNeeded: Int(g.next() % 255) - 127) }
        let scales = (0..<n).map { _ in Float(g.next() % 1000 + 1) / 1e5 }
        let ref: [Float] = (0..<n).map { i in
            Float((0..<dim).reduce(Int32(0)) { $0 + Int32(rows[i * dim + $1]) * Int32(q[$1]) }) * 0.01 * scales[i]
        }
        for sdot in [true, false] {
            pv_force_dotprod(sdot ? 1 : 0)
            var out = [Float](repeating: 0, count: n)
            pv_dot_i8(q, 0.01, rows, scales, n, dim, &out)
            for i in 0..<n { #expect(abs(out[i] - ref[i]) <= abs(ref[i]) * 1e-5 + 1e-6, "sdot \(sdot) row \(i)") }
        }
        pv_force_dotprod(1)
        #expect(pv_has_dotprod() == 1)
    }

    @Test func float16KernelMatchesScalarReference() {
        var g = Synthetic(seed: 2)
        let dim = 64, n = 33
        let v = g.vectors(n, dim: dim)
        let q = g.vectors(1, dim: dim)[0]
        let rows = v.flatMap { $0.map { Float16($0).bitPattern } }
        var out = [Float](repeating: 0, count: n)
        pv_dot_f16(q, rows, n, dim, &out)
        for i in 0..<n {
            let ref = zip(v[i], q).reduce(Float(0)) { $0 + Float(Float16($1.0)) * $1.1 }
            #expect(abs(out[i] - ref) < 1e-4)
        }
    }

    @Test func topKIsSortedAndExact() {
        var g = Synthetic(seed: 3)
        let scores = (0..<10_000).map { _ in g.gaussian() }
        var idx = [UInt32](repeating: 0, count: 25), sc = [Float](repeating: 0, count: 25)
        #expect(pv_topk(scores, scores.count, 25, &idx, &sc) == 25)
        let ref = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(25)
        #expect(idx.map(Int.init) == Array(ref))
        #expect(pv_topk(scores, 3, 25, &idx, &sc) == 3)
    }
}

@Suite struct IndexTests {
    static let dim = 128
    static let data: ([[Float]], [[Float]]) = {
        var g = Synthetic(seed: 42)
        let v = g.vectors(5_000, dim: dim)
        let q = g.vectors(50, dim: dim)
        return (v, q)
    }()

    func recall(_ index: FlatIndex, filter: RowFilter? = nil, allow: (Int) -> Bool = { _ in true }) -> Double {
        let (v, qs) = Self.data
        var hit = 0
        for q in qs {
            let truth = Set(exactTop(v, q, k: 10, allow: allow))
            hit += index.search(q, k: 10, filter: filter).filter { truth.contains($0.row) }.count
        }
        return Double(hit) / Double(qs.count * 10)
    }

    @Test(arguments: FlatIndex.Format.allCases) func recallAgainstExact(format: FlatIndex.Format) throws {
        let index = try FlatIndex(vectors: Self.data.0, format: format)
        let r = recall(index)
        switch format {
        case .float32: #expect(r == 1)
        case .float16: #expect(r >= 0.99)
        case .int8: #expect(r >= 0.95)
        }
        #expect(index.byteCount == 5_000 * Self.dim * (format == .float32 ? 4 : format == .float16 ? 2 : 1) + (format == .int8 ? 20_000 : 0))
    }

    @Test func scoresAreSimilarities() throws {
        let (v, _) = Self.data
        let index = try FlatIndex(vectors: v, format: .int8)
        let top = index.search(v[17], k: 1)
        #expect(top.first?.row == 17)
        #expect(abs((top.first?.score ?? 0) - 1) < 0.01)
        #expect(zip(index.vector(row: 17), v[17]).allSatisfy { abs($0 - $1) < 0.01 })
    }

    @Test func filtersReturnOnlyAllowedRowsAndMatchExact() throws {
        let (v, _) = Self.data
        let index = try FlatIndex(vectors: v, format: .float32)
        let range = 1_000..<1_400
        #expect(recall(index, filter: .rows(range), allow: { range.contains($0) }) == 1)
        let set = RowSet(capacity: v.count, rows: stride(from: 0, to: v.count, by: 7))
        #expect(set.count == (v.count + 6) / 7)
        #expect(recall(index, filter: .rows(in: set), allow: { $0 % 7 == 0 }) == 1)
        #expect(recall(index, filter: .where { $0 % 3 == 1 }, allow: { $0 % 3 == 1 }) == 1)
        for q in Self.data.1.prefix(5) {
            #expect(index.search(q, k: 10, filter: .rows(range)).allSatisfy { range.contains($0.row) })
            #expect(index.search(q, k: 10, filter: .rows(in: set)).allSatisfy { $0.row % 7 == 0 })
        }
        // fewer allowed rows than k: only those come back
        let three = RowSet(capacity: v.count, rows: [5, 9, 4_999])
        #expect(Set(index.search(Self.data.1[0], k: 10, filter: .rows(in: three)).map(\.row)) == [5, 9, 4_999])
        #expect(index.search(Self.data.1[0], k: 10, filter: .rows(10..<10)).isEmpty)
        #expect(index.search(Self.data.1[0], k: 10, filter: .rows(4_990..<9_000)).count == 10)
    }

    @Test(arguments: FlatIndex.Format.allCases) func fileRoundTrip(format: FlatIndex.Format) throws {
        let (v, qs) = Self.data
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fv-\(UUID().uuidString).fvx")
        defer { try? FileManager.default.removeItem(at: url) }
        try FlatIndex.write(v, format: format, to: url)
        let mem = try FlatIndex(vectors: v, format: format)
        for mapped in [true, false] {
            let file = try FlatIndex(contentsOf: url, mapped: mapped)
            #expect(file.count == v.count && file.dimension == Self.dim && file.format == format)
            file.prefault()
            for q in qs.prefix(10) {
                // same rows; scores within float rounding (Accelerate's fp32 path may vary in the last
                // bits with the buffer's alignment: heap vs mapped file)
                let a = file.search(q, k: 10), b = mem.search(q, k: 10)
                #expect(a.map(\.row) == b.map(\.row))
                #expect(zip(a, b).allSatisfy { abs($0.score - $1.score) <= 1e-5 })
            }
        }
    }

    @Test func rejectsBadInput() throws {
        #expect(throws: FlatIndexError.invalidDimension(10)) { try FlatIndex(vectors: [[Float](repeating: 0, count: 10)]) }
        #expect(throws: FlatIndexError.empty) { try FlatIndex(vectors: []) }
        #expect(throws: FlatIndexError.wrongVectorLength(row: 1, expected: 16, got: 32)) {
            try FlatIndex(vectors: [[Float](repeating: 1, count: 16), [Float](repeating: 1, count: 32)])
        }
        let dir = FileManager.default.temporaryDirectory
        let junk = dir.appendingPathComponent("fv-junk-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: junk) }
        try Data(repeating: 7, count: 100).write(to: junk)
        #expect(throws: FlatIndexError.notAnIndexFile) { try FlatIndex(contentsOf: junk) }
        // a truncated file is rejected, not read past its end
        let good = dir.appendingPathComponent("fv-trunc-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: good) }
        try FlatIndex.write(Self.data.0, format: .int8, to: good)
        let bytes = try Data(contentsOf: good)
        try bytes.prefix(bytes.count - 1).write(to: good)
        #expect(throws: FlatIndexError.self) { try FlatIndex(contentsOf: good) }
    }

    @Test func concurrentSearchesMatchSerial() async throws {
        let index = try FlatIndex(vectors: Self.data.0, format: .int8)
        let qs = Self.data.1
        let serial = qs.map { index.search($0, k: 10) }
        let parallel = await withTaskGroup(of: (Int, [SearchResult]).self) { group in
            for (i, q) in qs.enumerated() { group.addTask { (i, index.search(q, k: 10)) } }
            var out = [[SearchResult]](repeating: [], count: qs.count)
            for await (i, r) in group { out[i] = r }
            return out
        }
        #expect(parallel == serial)
    }

    @Test func quantizerHandlesZeroAndExtremes() {
        let (z, s) = Quantizer.int8([Float](repeating: 0, count: 16))
        #expect(z.allSatisfy { $0 == 0 } && s == 1)
        let (q, sc) = Quantizer.int8([1, -1, 0.5] + [Float](repeating: 0, count: 13))
        #expect(q[0] == 127 && q[1] == -127 && q[2] == 64 && abs(sc - 1 / 127) < 1e-9)
    }
}
