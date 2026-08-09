import Darwin
import Foundation
import TurboFieldfareServerCore

// Local change: stdout is fully buffered when redirected to a file, so the
// ready banner and per-request logs never appeared in `nohup` logs. Line
// buffering costs nothing at this volume and makes the log usable live.
setvbuf(stdout, nil, _IOLBF, 0)

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode,
        runtime: try arguments.resolvedRuntimeConfiguration(forceLogitsHead: true))
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend)
    _ = try await server.start(port: arguments.port)
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue) expert_cache=\(arguments.expertCacheSlots)/\(arguments.expertCachePolicy.rawValue) prefill=\(arguments.prefillPolicy.rawValue)/\(arguments.prefillChunkTokens) rdadvise=\(arguments.rdadvisePolicy.rawValue)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
