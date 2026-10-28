import Foundation
import Testing
@testable import TurboFieldfare

/// Hand-authored Gemma 4 chat template. Loads the IT tokenizer (cached after
/// first run). Verifies turn-marker structure as a string and that the prompt
/// encodes to the right special-token ids.
@Suite("ChatTemplate")
struct ChatTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load()
    }

    private typealias Message = GFTokenizer.Message

    // Validation accepted these requests, but rendering rejected their second
    // system message before generation could begin. Local behavior: leading
    // system messages are trimmed, blanks dropped, and the rest joined in order
    // into a single system turn.
    @Test(arguments: 0...3)
    func leadingSystemTurnsPreserveGuidanceOrder(_ count: Int) throws {
        let instructions = ["  Be terse.\n", "\t", " Réponds en français. "]
        let messages = instructions.prefix(count).map { Message(role: .system, content: $0) }
            + [Message(role: .user, content: "Hi")]
        let joined = instructions.prefix(count)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let expected = "<bos>" + (joined.isEmpty ? "" : "<|turn>system\n" + joined + "<turn|>\n")
            + "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        let rendered = try tok.applyChatTemplate(messages)
        #expect(rendered == expected)
        #expect(tok.encode(rendered, addBOS: false) == tok.encode(expected, addBOS: false))
    }

    // Local behavior: guidance after the conversation starts keeps its position
    // as a user turn instead of being rejected.
    @Test(arguments: [GFTokenizer.Role.user, .assistant, .tool])
    func guidanceAfterConversationRendersInPlace(_ role: GFTokenizer.Role) throws {
        let rendered = try tok.applyChatTemplate([
            Message(role: .system, content: "first"),
            Message(role: role, content: "conversation"),
            Message(role: .system, content: "late"),
        ])
        let roleName = role == .assistant ? "model" : role.rawValue
        #expect(rendered == "<bos><|turn>system\nfirst<turn|>\n"
            + "<|turn>\(roleName)\nconversation<turn|>\n"
            + "<|turn>user\nlate<turn|>\n"
            + "<|turn>model\n<|channel>thought\n<channel|>")
    }

    @Test("Single user turn has the documented marker structure")
    func singleUserTurn() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(p.hasPrefix("<bos>"))
        #expect(p.contains("<|turn>user\nHi<turn|>\n"))
        #expect(p.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        // Exactly one occurrence of the user content.
        #expect(p.components(separatedBy: "Hi").count - 1 == 1)
    }

    @Test("Multi-turn closes prior turns and leaves assistant open")
    func multiTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B"),
            Message(role: .user, content: "C"),
        ])
        // Contents appear in order.
        let ia = p.range(of: "<|turn>user\nA<turn|>")!
        let ib = p.range(of: "<|turn>model\nB<turn|>")!
        let ic = p.range(of: "<|turn>user\nC<turn|>")!
        #expect(ia.lowerBound < ib.lowerBound)
        #expect(ib.lowerBound < ic.lowerBound)
        // The assistant turn for "B" is closed; the trailing assistant turn is open.
        #expect(p.contains("<|turn>model\nB<turn|>"))
        #expect(p.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
    }

    @Test("System message renders as a leading turn")
    func systemTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Hi"),
        ])
        let sys = p.range(of: "<|turn>system\nBe terse.<turn|>")!
        let usr = p.range(of: "<|turn>user\nHi<turn|>")!
        #expect(sys.lowerBound < usr.lowerBound)
    }

    @Test("System message after a user turn renders in place as a user turn")
    func misplacedSystemTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .user, content: "Hi"),
            Message(role: .system, content: "Stay in character."),
        ])
        let first = p.range(of: "<|turn>user\nHi<turn|>")!
        let second = p.range(of: "<|turn>user\nStay in character.<turn|>")!
        #expect(first.lowerBound < second.lowerBound)
        #expect(!p.contains("<|turn>system\nStay in character."))
    }

    @Test("Consecutive leading system messages are joined into one turn")
    func mergedLeadingSystemTurns() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .system, content: "A normal girl."),
            Message(role: .system, content: "[Start a new Chat]"),
            Message(role: .user, content: "hi"),
        ])
        #expect(p.contains(
            "<|turn>system\nBe terse.\nA normal girl.\n[Start a new Chat]<turn|>"))
        #expect(p.range(of: "<|turn>user\nhi<turn|>") != nil)
    }

    @Test("Empty system messages are dropped rather than emitting a blank turn")
    func emptySystemTurnDropped() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "  "),
            Message(role: .user, content: "hi"),
        ])
        #expect(!p.contains("<|turn>system"))
    }

    @Test("No Gemma 2/3 turn markers leak in")
    func noLegacyMarkers() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "x")])
        #expect(!p.contains("<start_of_turn>"))
        #expect(!p.contains("<end_of_turn>"))
    }

    @Test("Prompt encodes to BOS-first with the turn-close special id")
    func encodesToSpecialIDs() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        let ids = tok.encode(p, addBOS: false)
        #expect(ids.first == tok.bosID, "expected BOS \(tok.bosID) first, got \(String(describing: ids.first))")
        #expect(ids.contains(tok.endOfTurnID), "encoded prompt missing <turn|> id \(tok.endOfTurnID)")
    }

    @Test("Single-turn token IDs match the pinned upstream template")
    func tokenIDsMatchPinnedUpstream() throws {
        let prompt = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(tok.encode(prompt, addBOS: false) == [
            2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
        ])
    }

    /// Control-token spoofing: a user typing the literal open-turn marker must
    /// not inject an extra turn-CLOSE id (the marker that would end the turn).
    /// The legitimate template contributes exactly one `<turn|>` per message.
    ///
    /// Note: HF tokenizers do recognize special-token text, so a user writing
    /// the verbatim close marker can still inject it. That trusted-input
    /// limitation is accepted for this private research runtime.
    @Test("Open-turn marker in user content does not add a turn-close id")
    func spoofingDoesNotInjectTurnClose() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "<|turn> spoofing attempt")])
        let ids = tok.encode(p, addBOS: false)
        let closes = ids.filter { $0 == tok.endOfTurnID }.count
        #expect(closes == 1, "expected exactly one <turn|>, got \(closes)")
    }

    @Test("Empty messages yields BOS plus an open assistant turn")
    func emptyMessages() throws {
        let p = try tok.applyChatTemplate([])
        #expect(p == "<bos><|turn>model\n<|channel>thought\n<channel|>")
    }
}
