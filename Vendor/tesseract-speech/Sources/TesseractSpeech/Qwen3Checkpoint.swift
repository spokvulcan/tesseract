// TesseractSpeech — what a Qwen3-TTS checkpoint directory must hold before
// `Qwen3Synthesizer` loads it. Disk only: no MLX, no network.
//
// One rule, two readers: the synthesizer checks it before the engine takes
// the GPU lease, and the app's Model Catalog uses it to decide whether the
// Voice Engine counts as downloaded. So the Models page can't call a
// checkpoint downloaded that the engine would then refuse.

import Foundation

public enum Qwen3Checkpoint {

    /// The files `directory` lacks for `Qwen3TTSModel.fromModelDirectory` to
    /// build a working model, as paths relative to it. Empty means complete.
    ///
    /// Present means a non-empty regular file. Sizes against the hub listing
    /// are the download manager's verify, not this check.
    public static func missingFiles(in directory: URL) -> [String] {
        var missing: [String] = []

        if !isJSONObject(directory.appendingPathComponent("config.json")) {
            missing.append("config.json")
        }
        missing += missingWeights(in: directory)

        // Text tokenizer: tokenizer.json, or the vocab.json + merges.txt the
        // vendor generates it from on first load. Without either the model
        // loads with no tokenizer and can't generate.
        if !hasContent(directory.appendingPathComponent("tokenizer_config.json")) {
            missing.append("tokenizer_config.json")
        }
        if !hasContent(directory.appendingPathComponent("tokenizer.json")) {
            for name in ["vocab.json", "merges.txt"]
            where !hasContent(directory.appendingPathComponent(name)) {
                missing.append(name)
            }
        }

        // The speech tokenizer decodes codec frames to audio. The vendor
        // loads without it and then can't produce sound.
        let speechTokenizer = directory.appendingPathComponent("speech_tokenizer")
        if !hasContent(speechTokenizer.appendingPathComponent("config.json")) {
            missing.append("speech_tokenizer/config.json")
        }
        if !hasSafetensors(in: speechTokenizer) {
            missing.append("speech_tokenizer/model.safetensors")
        }

        return missing
    }

    /// Top-level weights. With an index, every shard it names; without one,
    /// at least one `.safetensors` file (the vendor loads them all).
    private static func missingWeights(in directory: URL) -> [String] {
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String],
            !weightMap.isEmpty
        {
            return Set(weightMap.values).sorted().filter {
                !hasContent(directory.appendingPathComponent($0))
            }
        }
        return hasSafetensors(in: directory) ? [] : ["model.safetensors"]
    }

    private static func hasSafetensors(in directory: URL) -> Bool {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil))
            ?? []
        return files.contains { $0.pathExtension == "safetensors" && hasContent($0) }
    }

    private static func hasContent(_ file: URL) -> Bool {
        let values = try? file.resolvingSymlinksInPath()
            .resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        return values?.isRegularFile == true && (values?.fileSize ?? 0) > 0
    }

    private static func isJSONObject(_ file: URL) -> Bool {
        guard hasContent(file), let data = try? Data(contentsOf: file) else { return false }
        return (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
    }
}
