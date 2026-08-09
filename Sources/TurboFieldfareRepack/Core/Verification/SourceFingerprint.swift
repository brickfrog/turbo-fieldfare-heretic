import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = [
        // Upstream pinned source, spelled out so this table stays valid even
        // when SupportedModelSource points somewhere else.
        "mlx-community/gemma-4-26b-a4b-it-4bit":
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        // Local experiment: abliterated fine-tune of the same base. Identical
        // ArchConfig shape; differs only in quantization, which puts the dense
        // shared-expert MLP at 8 bits instead of 4. Not upstream-validated.
        "mlx-community/gemma-4-26B-A4B-it-heretic-4bit":
            "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
    ]

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }
}
