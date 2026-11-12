import Darwin
import PocketVectors
import Foundation

// pocket-vectors-bench [--rows 40000] [--dim 384] [--queries 500] [--repeats 3]
// Synthetic, seeded data only: unit vectors around 256 random centres (clustered, like text
// embeddings). Prints recall@10 against exact fp32, latency, size and memory for each format, and
// the filters. Run on a quiet machine: `swift run -c release pocket-vectors-bench`.
let args = CommandLine.arguments
func opt(_ n: String, _ d: Int) -> Int { args.firstIndex(of: n).flatMap { Int(args[$0 + 1]) } ?? d }
let rows = opt("--rows", 40_000), dim = opt("--dim", 384), nq = opt("--queries", 500), repeats = opt("--repeats", 3)

struct RNG {
    var s: UInt64
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func gaussian() -> Float {
        let u1 = max(Double(next() >> 11) / Double(1 << 53), 1e-12), u2 = Double(next() >> 11) / Double(1 << 53)
        return Float(sqrt(-2 * log(u1)) * cos(2 * .pi * u2))
    }
}
func unit(_ v: [Float]) -> [Float] { let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n } }
var g = RNG(s: 7)
let centres = (0..<256).map { _ in unit((0..<dim).map { _ in g.gaussian() }) }
@MainActor func sample() -> [Float] {
    let c = centres[Int(g.next() % 256)]
    return unit(c.map { $0 + 0.8 * g.gaussian() / Float(dim).squareRoot() })
}
let vectors = (0..<rows).map { _ in sample() }
let queries = (0..<nq).map { _ in sample() }

func footprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}
func ms<T>(_ f: () throws -> T) rethrows -> (T, Double) {
    let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let r = try f()
    return (r, Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6)
}
func pct(_ xs: [Double], _ p: Double) -> Double { let s = xs.sorted(); return s[min(s.count - 1, Int(Double(s.count) * p / 100))] }

var load = [Double](repeating: 0, count: 3)
getloadavg(&load, 3)
let exact = try FlatIndex(vectors: vectors, format: .float32)
let truth = queries.map { Set(exact.search($0, k: 10).map(\.row)) }
print("pocket-vectors-bench: \(rows) × \(dim) synthetic clustered unit vectors, \(nq) queries ×\(repeats), top 10; load \(String(format: "%.1f", load[0])) on \(ProcessInfo.processInfo.activeProcessorCount) CPUs\n")
print("| format | file MB | open ms (mmap) | p50 ms | p95 ms | recall@10 | footprint +MB (mmap, after queries) |")
print("|---|---:|---:|---:|---:|---:|---:|")
let dir = FileManager.default.temporaryDirectory.appendingPathComponent("pocket-vectors-bench-\(getpid())")
try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: dir) }
for format in FlatIndex.Format.allCases {
    let url = dir.appendingPathComponent("\(format).fvx")
    try FlatIndex.write(vectors, format: format, to: url)
    let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    let fp0 = footprint()
    let (index, openMs) = try ms { try FlatIndex(contentsOf: url) }
    index.prefault()
    var t: [Double] = [], hit = 0
    for r in 0..<repeats {
        for (i, q) in queries.enumerated() {
            let (res, m) = ms { index.search(q, k: 10) }
            t.append(m)
            if r == 0 { hit += res.filter { truth[i].contains($0.row) }.count }
        }
    }
    print(String(format: "| %@ | %.1f | %.3f | %.3f | %.3f | %.4f | %.1f |", "\(format)", Double(size) / 1e6, openMs, pct(t, 50), pct(t, 95),
                 Double(hit) / Double(nq * 10), Double(footprint() - fp0) / 1e6))
}
let i8 = try FlatIndex(vectors: vectors, format: .int8)
let tenth = RowSet(capacity: rows, rows: stride(from: 0, to: rows, by: 10))
print("\nFilters on int8 (p50 / p95 ms): ", terminator: "")
for (name, f) in [("range of 10 % of rows", RowFilter.rows(0..<(rows / 10))), ("RowSet of every 10th row", .rows(in: tenth)),
                  ("predicate, every 10th row", .where { $0 % 10 == 0 })] {
    var t: [Double] = []
    for q in queries { t.append(ms { i8.search(q, k: 10, filter: f) }.1) }
    print(String(format: "%@ %.3f / %.3f; ", name, pct(t, 50), pct(t, 95)), terminator: "")
}
print("")
