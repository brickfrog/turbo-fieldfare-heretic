import Foundation

public enum SupportedModelSource {
    public static let displayName = "Gemma 4 26B-A4B IT heretic 4-bit"
    public static let repoID = "mlx-community/gemma-4-26B-A4B-it-heretic-4bit"
    public static let revision = "1f1ae59040a78b1d717618f44c82757255925d26"
    public static let sourceIndexSHA256 =
        "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80"
    public static let approximateDownloadBytes: UInt64 = 15_608_838_516
    public static let installedBytes: UInt64 = 15_300_000_000
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}
