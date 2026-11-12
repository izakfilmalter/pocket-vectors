import Foundation
import Testing
@testable import PocketVectorsCoreML

@Suite struct TokenizerTests {
    // A tiny BERT-style vocabulary (ids are line numbers).
    static let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "un", "##believ", "##able", ",", "!", "cafe", "中", "the"]
    let tok = try! WordPieceTokenizer(vocabText: vocab.joined(separator: "\n"))
    func id(_ s: String) -> Int32 { Int32(Self.vocab.firstIndex(of: s)!) }

    @Test func basicTokensLowercaseStripAccentsAndSplit() {
        #expect(tok.basicTokens("Hello, Wörld!") == ["hello", ",", "world", "!"])
        #expect(tok.basicTokens("Café\tthe\u{00A0}中x") == ["cafe", "the", "中", "x"])
        #expect(tok.basicTokens("a\u{200B}b") == ["ab"])   // format characters are dropped
    }

    @Test func wordPieceGreedyLongestMatch() {
        #expect(tok.encode("Hello unbelievable world") == [id("[CLS]"), id("hello"), id("un"), id("##believ"), id("##able"), id("world"), id("[SEP]")])
        #expect(tok.encode("xyz") == [id("[CLS]"), id("[UNK]"), id("[SEP]")])
    }

    @Test func truncatesToMaxLength() {
        let ids = tok.encode(String(repeating: "hello ", count: 50), maxLength: 8)
        #expect(ids.count == 8 && ids.first == id("[CLS]") && ids.last == id("[SEP]"))
    }

    @Test func rejectsVocabularyWithoutSpecialTokens() {
        #expect(throws: TokenizerError.self) { try WordPieceTokenizer(vocabText: "hello\nworld") }
    }
}
