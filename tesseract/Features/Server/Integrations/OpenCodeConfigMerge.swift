//
//  OpenCodeConfigMerge.swift
//  tesseract
//

import Foundation

/// The Config Merge for the OpenCode Integration (CONTEXT.md → Client
/// integrations): regenerates the `provider.tesseract` block from an
/// `IntegrationSnapshot` while preserving everything else in the client's
/// file. The block is generated output — owned by the merge, replaced
/// wholesale on every run, so re-runs are idempotent, stale models drop out,
/// and renames propagate. A pure function of (existing bytes, snapshot);
/// writes nothing — the setup script owns backup and file replacement.
nonisolated enum OpenCodeConfigMerge {

    static let providerKey = "tesseract"

    struct Output: Equatable, Sendable {
        let configData: Data
        /// True when the existing file was present but unparseable — the merge
        /// started from a fresh config. The caller surfaces the warning; the
        /// setup script has already backed the original up.
        let replacedCorruptInput: Bool
    }

    static func merge(existingConfig: Data?, snapshot: IntegrationSnapshot) -> Output {
        var root: [String: Any]
        var replacedCorruptInput = false
        if let data = existingConfig, !data.isEmpty {
            // OpenCode parses every config file as JSONC regardless of
            // extension — accept the same, or a legal commented config would
            // be misclassified as corrupt and replaced.
            // Lossy UTF-8 decode is intentional — a stray invalid byte must not nil the result.
            // swiftlint:disable:next optional_data_string_conversion
            let sanitized = JSONCSanitizer.sanitize(String(decoding: data, as: UTF8.self))
            if let parsed = (try? JSONSerialization.jsonObject(with: Data(sanitized.utf8)))
                as? [String: Any]
            {
                root = parsed
            } else {
                replacedCorruptInput = true
                root = freshRoot()
            }
        } else {
            root = freshRoot()
        }

        var provider = (root["provider"] as? [String: Any]) ?? [:]
        provider[providerKey] = providerBlock(snapshot: snapshot)
        root["provider"] = provider

        // Running setup is an explicit "use Tesseract now" — the default
        // model is always set. Except with nothing downloaded: pointing the
        // default at a model the server would 404 helps nobody, so the
        // existing default (if any) is left alone.
        if let defaultModelID = snapshot.defaultModelID {
            root["model"] = "\(providerKey)/\(defaultModelID)"
        }

        var data =
            (try? JSONSerialization.data(
                withJSONObject: root,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            )) ?? Data("{}".utf8)
        data.append(Data("\n".utf8))
        return Output(configData: data, replacedCorruptInput: replacedCorruptInput)
    }

    // MARK: - Private

    private static func freshRoot() -> [String: Any] {
        ["$schema": "https://opencode.ai/config.json"]
    }

    private static func providerBlock(snapshot: IntegrationSnapshot) -> [String: Any] {
        var models: [String: Any] = [:]
        for model in snapshot.models {
            var entry: [String: Any] = [
                "name": "\(model.displayName) (Tesseract)",
                "modalities": [
                    "input": model.visionCapable ? ["text", "image"] : ["text"],
                    "output": ["text"],
                ],
                "limit": [
                    "context": model.contextLength,
                    "output": model.contextLength,
                ],
            ]
            if model.visionCapable {
                entry["attachment"] = true
            }
            models[model.id] = entry
        }
        return [
            "npm": "@ai-sdk/openai-compatible",
            "name": "Tesseract",
            "options": ["baseURL": "http://127.0.0.1:\(snapshot.port)/v1"],
            "models": models,
        ]
    }
}
