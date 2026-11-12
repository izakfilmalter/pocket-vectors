import Foundation

/// BERT uncased WordPiece tokenizer (what bge-small-en-v1.5, all-MiniLM-L6-v2 and other BERT-family
/// embedders use), matching Hugging Face's BertTokenizer: clean, split CJK, lowercase + strip accents,
/// split on whitespace and punctuation, greedy longest-match WordPiece with "##" continuations.
public struct WordPieceTokenizer: Sendable {
    public let vocab: [String: Int32]
    public let cls: Int32, sep: Int32, unk: Int32, pad: Int32
    private let maxCharsPerWord = 100

    /// `vocab.txt` from the model's Hugging Face repository: one token per line, id = line number.
    public init(vocabURL: URL) throws {
        try self.init(vocabText: String(contentsOf: vocabURL, encoding: .utf8))
    }

    public init(vocabText: String) throws {
        let lines = vocabText.split(separator: "\n", omittingEmptySubsequences: false)
        var v: [String: Int32] = [:]
        v.reserveCapacity(lines.count)
        for (i, l) in lines.enumerated() where !l.isEmpty { v[String(l)] = Int32(i) }
        guard let c = v["[CLS]"], let s = v["[SEP]"], let u = v["[UNK]"], let p = v["[PAD]"] else {
            throw TokenizerError.missingSpecialTokens
        }
        vocab = v
        cls = c; sep = s; unk = u; pad = p
    }

    /// [CLS] tokens [SEP], truncated to maxLength.
    public func encode(_ text: String, maxLength: Int = 128) -> [Int32] {
        var ids: [Int32] = [cls]
        for word in basicTokens(text) {
            for id in wordPiece(word) {
                if ids.count >= maxLength - 1 { break }
                ids.append(id)
            }
        }
        ids.append(sep)
        return ids
    }

    public func basicTokens(_ text: String) -> [String] {
        // NFD + drop combining marks = BERT's strip_accents; lowercase first, as HF does
        let lowered = text.lowercased().decomposedStringWithCanonicalMapping
        var tokens: [String] = []
        var cur = String.UnicodeScalarView()
        func flush() {
            if !cur.isEmpty { tokens.append(String(cur)); cur = String.UnicodeScalarView() }
        }
        for u in lowered.unicodeScalars {
            let v = u.value
            if v == 0 || v == 0xFFFD || (isControl(u)) { continue }
            if u.properties.generalCategory == .nonspacingMark { continue }
            if isWhitespace(u) { flush(); continue }
            if isPunctuation(u) || isCJK(v) {
                flush()
                tokens.append(String(Character(u)))
                continue
            }
            cur.append(u)
        }
        flush()
        return tokens
    }

    private func wordPiece(_ word: String) -> [Int32] {
        let chars = Array(word.unicodeScalars)
        if chars.count > maxCharsPerWord { return [unk] }
        var out: [Int32] = []
        var start = 0
        while start < chars.count {
            var end = chars.count
            var found: Int32?
            while start < end {
                var s = String(String.UnicodeScalarView(chars[start..<end]))
                if start > 0 { s = "##" + s }
                if let id = vocab[s] { found = id; break }
                end -= 1
            }
            guard let id = found else { return [unk] }
            out.append(id)
            start = end
        }
        return out
    }

    private func isWhitespace(_ u: Unicode.Scalar) -> Bool {
        u == " " || u == "\t" || u == "\n" || u == "\r" || u.properties.generalCategory == .spaceSeparator
    }

    private func isControl(_ u: Unicode.Scalar) -> Bool {
        if u == "\t" || u == "\n" || u == "\r" { return false }
        switch u.properties.generalCategory {
        case .control, .format: return true
        default: return false
        }
    }

    private func isPunctuation(_ u: Unicode.Scalar) -> Bool {
        let v = u.value
        if (33...47).contains(v) || (58...64).contains(v) || (91...96).contains(v) || (123...126).contains(v) { return true }
        switch u.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation: return true
        default: return false
        }
    }

    private func isCJK(_ v: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) || (0x20000...0x2A6DF).contains(v)
            || (0x2A700...0x2B73F).contains(v) || (0x2B740...0x2B81F).contains(v) || (0x2B820...0x2CEAF).contains(v)
            || (0xF900...0xFAFF).contains(v) || (0x2F800...0x2FA1F).contains(v)
    }
}

public enum TokenizerError: Error {
    /// The vocabulary lacks one of [CLS], [SEP], [UNK], [PAD].
    case missingSpecialTokens
}
