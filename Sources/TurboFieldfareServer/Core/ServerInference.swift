import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
    }
}

enum StructuredOutputFailureKind: String, Equatable, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

enum StructuredOutputFailureCause: String, Equatable, Sendable {
    case malformed
    case unknownTool = "unknown_tool"
    case oversized
    case unexpected
    case none

    static func classify(_ error: Error) -> Self {
        guard let parserError = error as? GemmaToolCallParserError else {
            return .unexpected
        }
        switch parserError {
        case .malformed: return .malformed
        case .unknownTool: return .unknownTool
        case .oversized: return .oversized
        }
    }
}

struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    let renderedPromptTokens: Int
    let effectivePromptTokens: Int
    let resultPromptTokens: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let completionTokens: Int
    let maxCompletionTokens: Int
    let rawStop: String
    let kvPosition: Int
    let kvBackedTokens: Int
    let boundaryTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let stopStringMatched: Bool
    let toolStartCount: Int
    let toolEndCount: Int
    let toolResponseCount: Int
    let toolResponseEndCount: Int
    let lastToolStartOffset: Int
    let lastToolEndOffset: Int
    let lastToolResponseOffset: Int
    let lastToolResponseEndOffset: Int
    let effectiveCountMatchesResult: Bool
    let effectivePrefixMatchesKV: Bool
    let kvPositionMatchesHistory: Bool
    let completionCountMatchesHistory: Bool
    let prefillAccountingMatches: Bool
    let renderedPromptHash: String
    let effectivePromptHash: String
    let generatedHash: String

    init(
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        result: RawDecodeResult,
        maxCompletionTokens: Int,
        decodedCalls: Int,
        visibleBytes: Int,
        stopStringMatched: Bool,
        toolStartID: Int32,
        toolEndID: Int32,
        toolResponseID: Int32,
        toolResponseEndID: Int32
    ) {
        let safePrefillCount = min(
            max(result.prefillTokens, 0),
            result.kvBackedTokenIDs.count)
        let committedGenerated = result.kvBackedTokenIDs.dropFirst(safePrefillCount)
        let boundary = result.uncommittedBoundaryTokenIDs[...]
        let generatedSegments = [committedGenerated, boundary]

        var toolStartCount = 0
        var toolEndCount = 0
        var toolResponseCount = 0
        var toolResponseEndCount = 0
        var lastToolStartOffset = -1
        var lastToolEndOffset = -1
        var lastToolResponseOffset = -1
        var lastToolResponseEndOffset = -1
        var offset = 0
        for segment in generatedSegments {
            for tokenID in segment {
                if tokenID == toolStartID {
                    toolStartCount += 1
                    lastToolStartOffset = offset
                }
                if tokenID == toolEndID {
                    toolEndCount += 1
                    lastToolEndOffset = offset
                }
                if tokenID == toolResponseID {
                    toolResponseCount += 1
                    lastToolResponseOffset = offset
                }
                if tokenID == toolResponseEndID {
                    toolResponseEndCount += 1
                    lastToolResponseEndOffset = offset
                }
                offset += 1
            }
        }

        let (prefillAccounted, prefillOverflow) = result.cachedPromptTokens
            .addingReportingOverflow(result.computedPrefillTokens)

        self.renderedPromptTokens = renderedPromptIDs.count
        self.effectivePromptTokens = effectivePromptIDs.count
        self.resultPromptTokens = result.prefillTokens
        self.cachedPromptTokens = result.cachedPromptTokens
        self.computedPrefillTokens = result.computedPrefillTokens
        self.completionTokens = result.newTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.rawStop = Self.rawStop(result.reason)
        self.kvPosition = result.kvPosition
        self.kvBackedTokens = result.kvBackedTokenIDs.count
        self.boundaryTokens = result.uncommittedBoundaryTokenIDs.count
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.stopStringMatched = stopStringMatched
        self.toolStartCount = toolStartCount
        self.toolEndCount = toolEndCount
        self.toolResponseCount = toolResponseCount
        self.toolResponseEndCount = toolResponseEndCount
        self.lastToolStartOffset = lastToolStartOffset
        self.lastToolEndOffset = lastToolEndOffset
        self.lastToolResponseOffset = lastToolResponseOffset
        self.lastToolResponseEndOffset = lastToolResponseEndOffset
        self.effectiveCountMatchesResult = effectivePromptIDs.count == result.prefillTokens
        self.effectivePrefixMatchesKV = result.kvBackedTokenIDs.count >= effectivePromptIDs.count
            && result.kvBackedTokenIDs.prefix(effectivePromptIDs.count)
                .elementsEqual(effectivePromptIDs)
        self.kvPositionMatchesHistory = result.kvPosition == result.kvBackedTokenIDs.count
        self.completionCountMatchesHistory = offset == result.newTokens
        self.prefillAccountingMatches = !prefillOverflow
            && prefillAccounted == result.prefillTokens
        self.renderedPromptHash = Self.i32leSHA256([renderedPromptIDs[...]])
        self.effectivePromptHash = Self.i32leSHA256([effectivePromptIDs[...]])
        self.generatedHash = Self.i32leSHA256(generatedSegments)
    }

    static func i32leSHA256(_ segments: [ArraySlice<Int32>]) -> String {
        var hasher = SHA256()
        var bytes = Data()
        bytes.reserveCapacity(4_096)
        for segment in segments {
            for tokenID in segment {
                var littleEndian = UInt32(bitPattern: tokenID).littleEndian
                withUnsafeBytes(of: &littleEndian) {
                    bytes.append(contentsOf: $0)
                }
                if bytes.count == 4_096 {
                    hasher.update(data: bytes)
                    bytes.removeAll(keepingCapacity: true)
                }
            }
        }
        if !bytes.isEmpty { hasher.update(data: bytes) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    var logDescription: String {
        [
            "rendered_prompt_tokens=\(renderedPromptTokens)",
            "effective_prompt_tokens=\(effectivePromptTokens)",
            "result_prompt_tokens=\(resultPromptTokens)",
            "cached_prompt_tokens=\(cachedPromptTokens)",
            "computed_prefill_tokens=\(computedPrefillTokens)",
            "completion_tokens=\(completionTokens)",
            "max_completion_tokens=\(maxCompletionTokens)",
            "raw_stop=\(rawStop)",
            "kv_position=\(kvPosition)",
            "kv_backed_tokens=\(kvBackedTokens)",
            "boundary_tokens=\(boundaryTokens)",
            "decoded_calls=\(decodedCalls)",
            "visible_bytes=\(visibleBytes)",
            "stop_string_matched=\(stopStringMatched)",
            "tool_start_count=\(toolStartCount)",
            "tool_end_count=\(toolEndCount)",
            "tool_response_count=\(toolResponseCount)",
            "tool_response_end_count=\(toolResponseEndCount)",
            "last_tool_start_offset=\(lastToolStartOffset)",
            "last_tool_end_offset=\(lastToolEndOffset)",
            "last_tool_response_offset=\(lastToolResponseOffset)",
            "last_tool_response_end_offset=\(lastToolResponseEndOffset)",
            "effective_count_matches_result=\(effectiveCountMatchesResult)",
            "effective_prefix_matches_kv=\(effectivePrefixMatchesKV)",
            "kv_position_matches_history=\(kvPositionMatchesHistory)",
            "completion_count_matches_history=\(completionCountMatchesHistory)",
            "prefill_accounting_matches=\(prefillAccountingMatches)",
            "rendered_prompt_i32le_sha256=\(renderedPromptHash)",
            "effective_prompt_i32le_sha256=\(effectivePromptHash)",
            "generated_i32le_sha256=\(generatedHash)",
        ].joined(separator: " ")
    }

    private static func rawStop(_ reason: StopReason) -> String {
        switch reason {
        case .eos: "eos"
        case .endOfTurn: "end_of_turn"
        case .maxTokens: "max_tokens"
        case .stopString: "stop_string"
        case .toolCalls: "tool_calls"
        }
    }
}

struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue) \(diagnostics.logDescription)"
    }
}

public struct ServerPreparedRequest: Sendable {
    public let request: ValidatedChatRequest
    fileprivate let promptIDs: [Int32]?

    public var promptTokenCount: Int? { promptIDs?.count }

    init(request: ValidatedChatRequest, promptIDs: [Int32]? = nil) {
        self.request = request
        self.promptIDs = promptIDs
    }
}

public protocol ServerInferenceBackend: Sendable {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
    func generate(_ prepared: ServerPreparedRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public extension ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request)
    }

    func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(prepared.request, onEvent: onEvent)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        guard admittedCount <= queueLimit else { throw ServerRequestError.queueFull }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerInferenceBackend {
    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache = ServerPromptCache()

    /// Local addition: KV bytes as they stood the moment a prompt finished
    /// prefilling, keyed by that prompt. Two things defeat the token cache in
    /// normal chat use — stopping a reply mid-stream, and swiping for a reroll —
    /// and both leave the KV holding tokens the next request does not want.
    /// The cursor cannot simply be wound back (SWA layers recycle ring slots
    /// after 128 tokens), but the bytes can be put back, which costs a memcpy
    /// instead of a full re-prefill.
    /// Both KV states of the last completed turn, keyed by the messages that
    /// produced them. Keyed by messages rather than tokens on purpose: the KV
    /// is built from the server's own generated token IDs, which a client
    /// re-rendering the same conversation from text does not reproduce, so
    /// token-prefix comparison diverges almost immediately.
    private struct TurnSnapshot {
        let messages: [GFTokenizer.Message]
        let tools: [GFTokenizer.FunctionDefinition]
        /// Tokens and KV as of the end of prefill, before any reply token.
        /// Replaying these regenerates the same turn — a swipe.
        let promptTokens: [Int32]
        let beforeGeneration: KVCacheManager.Snapshot
        /// KV once the reply was committed; the state `promptCache` describes.
        let afterGeneration: KVCacheManager.Snapshot
    }
    private var lastTurn: TurnSnapshot?

    private struct PrefillSnapshot {
        let tokens: [Int32]
        let kv: KVCacheManager.Snapshot
    }

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix,
                            runtime runtimeOverride: RuntimeConfiguration? = nil) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let context = try MetalContext()
        let runtime = runtimeOverride ?? RuntimeConfiguration(forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: runner,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 runner: RealForwardRunner,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
    }

    /// True when `request` is a reroll of the last completed turn: identical
    /// messages and tools, with no new turn appended. That is what a swipe
    /// looks like, and the token cache can never serve it — there is nothing to
    /// extend, only something to replay.
    private func isSwipe(of request: ValidatedChatRequest) -> Bool {
        guard let last = lastTurn else { return false }
        return request.tools == last.tools
            && request.messages.count == last.messages.count
            && request.messages.elementsEqual(last.messages)
    }

    /// Rewind one token off the pre-generation snapshot so the resume has a
    /// token left to prefill, and hand back the KV-space prompt to replay.
    private func restoreForSwipe() -> (tokens: [Int32], resumeAt: Int)? {
        guard let last = lastTurn, last.promptTokens.count > 1 else { return nil }
        let resumeAt = last.promptTokens.count - 1
        do {
            try runner.restoreKVCache(last.beforeGeneration, rewoundBy: 1)
            return (last.promptTokens, resumeAt)
        } catch {
            print("[prompt-cache] swipe restore failed: \(error)")
            lastTurn = nil
            runner.reset()
            return nil
        }
    }

    /// Local addition: explain why the single-prefix cache failed to match.
    /// Read-only; used purely for the diagnostic log line.
    private func promptCacheMissReason(_ request: ValidatedChatRequest,
                                       _ renderedPromptIDs: [Int32]) -> String {
        guard let entry = promptCache.entry else {
            return "cold (no cached turn yet)"
        }
        guard entry.domain == promptCacheDomain else {
            return "runtime/model domain changed"
        }
        guard entry.tools == request.tools else {
            return "tool definitions changed"
        }

        let shared = zip(renderedPromptIDs, entry.kvBackedTokenIDs)
            .prefix { $0.0 == $0.1 }.count
        if shared < entry.kvBackedTokenIDs.count {
            return "token \(shared) of \(entry.kvBackedTokenIDs.count) diverges"
                + " (prompt prefix rewritten upstream)"
        }

        let cachedCount = entry.inputMessages.count
        guard request.messages.count > cachedCount + 1 else {
            return "history shrank: \(request.messages.count) messages vs"
                + " cached \(cachedCount)+assistant (context truncation?)"
        }
        for i in 0..<cachedCount where request.messages[i] != entry.inputMessages[i] {
            return "message[\(i)] role=\(request.messages[i].role.rawValue)"
                + " edited since it was cached"
        }
        let incoming = request.messages[cachedCount]
        guard incoming.role == .assistant else {
            return "message[\(cachedCount)] is role=\(incoming.role.rawValue),"
                + " expected the cached assistant turn"
        }
        if incoming.content != entry.assistantTurn.message.content {
            return "assistant reply was modified before resend"
                + " (regex script / markdown fixer / trim?)"
        }
        let continuation = request.messages.count - (cachedCount + 1)
        if continuation != 1 {
            return "\(continuation) new messages after the assistant turn;"
                + " cache resumes only on exactly 1 new user turn"
        }
        return "structural match failed (stop reason \(entry.assistantTurn.rawStopReason))"
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prepared = try await prepare(request)
        return try await generate(prepared, onEvent: onEvent)
    }

    public func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request, promptIDs: try renderPrompt(request))
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        var completed = false
        // Captured once this request's prompt is fully prefilled, so an abort
        // can put the KV back where it was instead of discarding everything.
        var pendingSnapshot: PrefillSnapshot?
        defer {
            if !completed {
                // Put the KV back to the end of the last completed turn. That
                // is exactly the state `promptCache`'s entry describes, so the
                // entry stays valid and the next turn resumes normally instead
                // of re-prefilling the whole conversation.
                if let last = lastTurn {
                    do {
                        try runner.restoreKVCache(last.afterGeneration)
                        print("[prompt-cache] ABORT rolled back to last turn "
                            + "(\(last.afterGeneration.tokenPosition) tokens, "
                            + "\(last.afterGeneration.byteCount / 1_048_576) MB)")
                    } catch {
                        print("[prompt-cache] ABORT rollback failed: \(error)")
                        promptCache.invalidate()
                        lastTurn = nil
                        runner.reset()
                    }
                } else {
                    promptCache.invalidate()
                    runner.reset()
                }
            }
        }
        let needsToolTemplate = usesToolTemplate(request)
        let promptIDs = try prepared.promptIDs ?? renderPrompt(request)

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                // Local change: a miss silently re-prefills the whole
                // conversation, which is the dominant cost in long chats.
                // Before paying that, see whether this prompt matches a
                // snapshot from an earlier prefill — a swipe, or a retry after
                // a stopped stream — and restore the bytes instead. Otherwise
                // say why we missed, so a client that rewrites the prompt each
                // turn can be identified.
                let reason = promptCacheMissReason(request, promptIDs)
                promptCache.invalidate()
                if isSwipe(of: request), let swipe = restoreForSwipe() {
                    print("[prompt-cache] SWIPE reused=\(swipe.resumeAt) "
                        + "new=\(swipe.tokens.count - swipe.resumeAt) "
                        + "total=\(swipe.tokens.count)")
                    effectivePromptIDs = swipe.tokens
                    completionStart = .resume(cachedPromptTokens: swipe.resumeAt)
                } else {
                    print("[prompt-cache] MISS prompt_tokens=\(promptIDs.count) "
                        + "reason=\(reason)")
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                }
            case .hit(let effective, let cached):
                print("[prompt-cache] HIT reused=\(cached) "
                    + "new=\(effective.count - cached) total=\(effective.count)")
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let decoder = needsToolTemplate
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false
        // Bound locally so the progress closure does not capture `self` and
        // become actor-isolated; `runner` already crosses this boundary as
        // `producer:` below.
        let snapshotSource = runner

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill(let done, let total):
                        // Snapshot once the prompt is fully resident, before
                        // any generated token has moved the cursor.
                        if done == total, pendingSnapshot == nil,
                           let kv = snapshotSource.snapshotKVCache() {
                            pendingSnapshot = PrefillSnapshot(
                                tokens: effectivePromptIDs, kv: kv)
                        }
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        let visible = stopMatcher.push(text)
                        if !visible.isEmpty {
                            content += visible
                            onEvent(.content(visible))
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: promptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: calls.count,
                    visibleBytes: content.utf8.count,
                    stopStringMatched: stopMatcher.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        do {
            try decoder?.finish()
        } catch {
            throw structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error))
        }
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if promptCacheMode == .singlePrefix {
            promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
            // Retain both KV states of this turn: the pre-generation one lets a
            // swipe replay it, the post-generation one lets an aborted next
            // turn roll back to it.
            if let pending = pendingSnapshot,
               let after = runner.snapshotKVCache() {
                lastTurn = TurnSnapshot(
                    messages: request.messages,
                    tools: request.tools,
                    promptTokens: pending.tokens,
                    beforeGeneration: pending.kv,
                    afterGeneration: after)
            } else {
                lastTurn = nil
            }
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens))
    }

    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        let promptIDs: [Int32]
        if usesToolTemplate(request) {
            promptIDs = try tokenizer.encodeToolChat(
                messages: request.messages,
                tools: request.tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(request.messages)
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return promptIDs
    }

    private func usesToolTemplate(_ request: ValidatedChatRequest) -> Bool {
        !request.tools.isEmpty || request.messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }
}
