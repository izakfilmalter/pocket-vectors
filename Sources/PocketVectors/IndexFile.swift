import Foundation

/// The index file: one little-endian file, memory-mappable.
///
///     offset  size  field
///     0       8     magic "FLATVEC\0"
///     8       4     version (1)
///     12      4     format (0 float32, 1 float16, 2 int8)
///     16      4     dimension
///     20      4     reserved (0)
///     24      8     count
///     32      8     scales offset (int8: count float32 scales; else 0)
///     40      8     data offset (count × dimension values, row-major)
///     48      16    reserved (0)
///     64            scales, then rows; each section starts on a 64-byte boundary
struct IndexFileHeader {
    static let magic: [UInt8] = Array("FLATVEC".utf8) + [0]
    static let size = 64
    let format: FlatIndex.Format
    let dimension: Int
    let count: Int
    let scalesOffset: Int
    let dataOffset: Int

    init(_ file: Data) throws {
        guard file.count >= Self.size, Array(file.prefix(8)) == Self.magic else { throw FlatIndexError.notAnIndexFile }
        func u32(_ o: Int) -> UInt32 { file.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) } }
        func u64(_ o: Int) -> UInt64 { file.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt64.self)) } }
        guard u32(8) == 1 else { throw FlatIndexError.unsupportedVersion(u32(8)) }
        guard let f = FlatIndex.Format(rawValue: u32(12)) else { throw FlatIndexError.corrupt("format \(u32(12))") }
        format = f
        dimension = Int(u32(16))
        guard dimension > 0, dimension % 16 == 0 else { throw FlatIndexError.invalidDimension(dimension) }
        guard u64(24) > 0, u64(24) < UInt64(Int32.max) else { throw FlatIndexError.corrupt("count \(u64(24))") }
        count = Int(u64(24))
        scalesOffset = Int(clamping: u64(32))
        dataOffset = Int(clamping: u64(40))
        let dataEnd = dataOffset.addingReportingOverflow(count * dimension * f.width)
        guard dataOffset >= Self.size, dataOffset % 64 == 0, !dataEnd.overflow, dataEnd.partialValue <= file.count else {
            throw FlatIndexError.corrupt("rows run past the end of the file")
        }
        if f == .int8 {
            guard scalesOffset >= Self.size, scalesOffset % 64 == 0, scalesOffset + count * 4 <= dataOffset else {
                throw FlatIndexError.corrupt("int8 scales out of place")
            }
        }
    }

    static func file(format: FlatIndex.Format, dimension: Int, count: Int, data: Data, scales: Data?) -> Data {
        func align(_ x: Int) -> Int { (x + 63) / 64 * 64 }
        let scalesOffset = scales == nil ? 0 : size
        let dataOffset = align(size + (scales?.count ?? 0))
        var out = Data(count: dataOffset)
        out.replaceSubrange(0..<8, with: magic)
        func put32(_ o: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.replaceSubrange(o..<(o + 4), with: $0) } }
        func put64(_ o: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { out.replaceSubrange(o..<(o + 8), with: $0) } }
        put32(8, 1); put32(12, format.rawValue); put32(16, UInt32(dimension))
        put64(24, UInt64(count)); put64(32, UInt64(scalesOffset)); put64(40, UInt64(dataOffset))
        if let scales { out.replaceSubrange(size..<(size + scales.count), with: scales) }
        out.append(data)
        return out
    }
}
