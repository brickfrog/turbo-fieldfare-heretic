import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Gemma 4 interleaves 25 sliding-window
/// layers with 5 full-attention layers (the latter carry the K=V shared-tensor
/// quirk). Sourced from `ArchConfig.fullAttentionLayerMask`.
public enum LayerKind: Sendable { case swa, full }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token (numKVHeads * headDim * sizeof(FP16)).
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    /// Ring start slot. 0 under linear storage; the hook for the wrap-aware path.
    public let startSlot: Int
}

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; FP16 ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// Gemma 4 full-attention layers carry the `attention_k_eq_v` quirk: K and V
/// share the `k_proj` weight, so a single 4-bit dequant + GEMV produces the
/// raw projection. But after that the K-slot runs `k_norm` (per-head, with
/// scale) + RoPE while the V-slot runs `v_norm` (per-head, no scale) and
/// skips RoPE — they diverge before entering attention. So the cache buffers
/// must be separate; aliasing them would smash the V values with K's normed,
/// rotated bytes (gemma4-block.md §2.2). We always allocate K and V slots,
/// independent of `attentionKEqV`.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool

    private let kBuffers: [MTLBuffer]
    private let vBuffers: [MTLBuffer]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]
    private let slidingWindowTokens: Int

    public private(set) var position: Int = 0
    public private(set) var highWaterPosition: Int = 0

    private static let fp16Size = 2

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        self.config = config
        self.maxContext = maxContext
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))
        self.slidingWindowTokens = slidingWindow ?? config.slidingWindow

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let stride = isFull ? fullStride : swaStride
            let capacity = ringEnabled && !isFull ? swaCapacity : maxContext
            let length = capacity * stride

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`.
    public func bufferLength(layer: Int) -> Int {
        return capacityTokens[layer] * strides[layer]
    }

    /// Write target for this layer's K projection at `position`.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `kSlot` — full layers no longer alias K and V (Gemma 4
    /// applies different per-head norms + RoPE to K vs V; gemma4-block.md §2.2).
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: kBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: vBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
        highWaterPosition = max(highWaterPosition, position)
    }

    /// Maximum safe rewind before a reused SWA ring slot could become visible.
    public var maxRewindTokens: Int {
        guard fp16RingEnabled else { return maxContext }
        var slack = maxContext
        for layer in 0..<config.numLayers where kinds[layer] == .swa {
            let capacity = capacityTokens[layer]
            guard capacity < maxContext else { continue }
            slack = min(slack, max(0, capacity - slidingWindowTokens))
        }
        return slack
    }

    /// Abandon the newest rows while keeping the remaining KV lineage intact.
    public func rewind(to newPosition: Int) {
        precondition(newPosition >= 0, "rewind target must be non-negative")
        precondition(newPosition <= position,
                     "rewind target \(newPosition) is beyond position \(position)")
        precondition(highWaterPosition - newPosition <= maxRewindTokens,
                     "rewind to \(newPosition) exceeds ring slack \(maxRewindTokens)")
        position = newPosition
    }

    /// A copy of the live KV bytes plus the cursor they belong to.
    ///
    /// Local addition. The cursor alone cannot be rewound: SWA layers store at
    /// `position % capacity`, so generating N tokens past P overwrites the
    /// slots holding `[P - capacity, P - capacity + N)`. Attention at a rewound
    /// P needs `[P - slidingWindow, P]`, which survives only while
    /// `N <= capacity - slidingWindow` (128 tokens here). Restoring the bytes
    /// sidesteps that entirely, and a ~570 MB copy at full context beats
    /// re-prefilling the conversation by roughly three orders of magnitude.
    public struct Snapshot: @unchecked Sendable {
        fileprivate let k: [Data]
        fileprivate let v: [Data]
        fileprivate let position: Int
        fileprivate let highWaterPosition: Int
        /// Live bytes held by this snapshot, for logging.
        public var byteCount: Int {
            k.reduce(0) { $0 + $1.count } + v.reduce(0) { $0 + $1.count }
        }
        public var tokenPosition: Int { position }
    }

    /// Bytes that must be preserved for `layer`. Ring layers wrap, so the whole
    /// ring is live; linear layers only ever hold `[0, position)`.
    private func liveByteCount(layer: Int) -> Int {
        let tokens = ringCapacity(layer: layer) > 0
            ? capacityTokens[layer]
            : min(position, capacityTokens[layer])
        return tokens * strides[layer]
    }

    public func snapshot() -> Snapshot {
        var ks: [Data] = []
        var vs: [Data] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        for layer in 0..<config.numLayers {
            let count = liveByteCount(layer: layer)
            ks.append(Data(bytes: kBuffers[layer].contents(), count: count))
            vs.append(Data(bytes: vBuffers[layer].contents(), count: count))
        }
        return Snapshot(
            k: ks,
            v: vs,
            position: position,
            highWaterPosition: highWaterPosition)
    }

    /// How far the cursor may be wound back without reading slots the ring has
    /// already recycled. Restoring bytes then stepping back by at most this
    /// much keeps `[position - slidingWindow, position]` intact.
    public var safeRewindTokens: Int {
        guard fp16RingEnabled else { return maxContext }
        var slack = maxContext
        for layer in 0..<config.numLayers where kinds[layer] == .swa {
            let capacity = capacityTokens[layer]
            // A ring sized for the whole context never recycles a slot, so
            // nothing behind the cursor can have been overwritten.
            guard capacity < maxContext else { continue }
            slack = min(slack, max(0, capacity - slidingWindowTokens))
        }
        return slack
    }

    /// Reinstate `snapshot`, optionally stepping the cursor back by `rewoundBy`
    /// tokens. A resume needs at least one token left to prefill, so replaying
    /// the exact prompt that produced the snapshot rewinds by 1.
    public func restore(_ snapshot: Snapshot, rewoundBy rewind: Int = 0) {
        precondition(snapshot.k.count == config.numLayers
                        && snapshot.v.count == config.numLayers,
                     "snapshot layer count does not match this cache")
        precondition(snapshot.position <= maxContext,
                     "snapshot position \(snapshot.position) exceeds maxContext \(maxContext)")
        precondition(rewind >= 0 && rewind <= snapshot.position,
                     "rewind \(rewind) outside snapshot of \(snapshot.position) tokens")
        precondition(rewind <= safeRewindTokens,
                     "rewind \(rewind) exceeds safe ring slack \(safeRewindTokens)")
        for layer in 0..<config.numLayers {
            copy(snapshot.k[layer], into: kBuffers[layer])
            copy(snapshot.v[layer], into: vBuffers[layer])
        }
        position = snapshot.position - rewind
        // The restored bytes belong to the snapshot's lineage, not to tokens
        // generated after it. Keeping the abandoned high-water mark would make
        // the upstream rewind guard reject otherwise safe future rewinds.
        highWaterPosition = snapshot.highWaterPosition
    }

    private func copy(_ data: Data, into buffer: MTLBuffer) {
        guard !data.isEmpty else { return }
        precondition(data.count <= buffer.length,
                     "snapshot slice larger than its destination buffer")
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            buffer.contents().copyMemory(from: base, byteCount: raw.count)
        }
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        position = 0
        highWaterPosition = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        position % capacityTokens[layer]
    }

    private func ringStartSlot(layer: Int, validTokenCount: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        let capacity = capacityTokens[layer]
        guard validTokenCount > capacity else { return 0 }
        return validTokenCount % capacity
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
