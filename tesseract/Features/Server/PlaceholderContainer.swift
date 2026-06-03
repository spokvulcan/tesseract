//
//  PlaceholderContainer.swift
//  tesseract
//
//  Neutral codec for the SSD prefix-cache tier's placeholder on-disk
//  container — the length-prefixed binary format that wraps a
//  `SnapshotPayload`'s tensor blobs behind a JSON header carrying the
//  full `PersistedSnapshotDescriptor`.
//
//  Deliberately **MLX-free**. It owns only the bytes ↔ header/blob
//  boundary: encode a payload into the container layout, and parse the
//  header back out (either from an in-memory `Data` or by streaming
//  just the header section off a file handle). The Metal-affine step
//  that reconstructs `MLXArray`s from the blob slices stays in
//  `SSDSnapshotStore.decodePlaceholderContainer`, inside
//  `container.perform`.
//
//  Sharing the codec here is what lets the **Snapshot Ledger**'s
//  corrupt-manifest directory-walk rebuild extract descriptors from the
//  per-file headers without depending on `SSDSnapshotStore`, and lets
//  the store's writer/`loadSync` paths encode/parse without depending on
//  the ledger — no store↔ledger cycle through the on-disk format.
//
//  ## Layout
//
//  ```
//  [8 bytes little-endian UInt64: header JSON length]
//  [header JSON bytes]
//  [concatenated array data blobs]
//  ```
//

import Foundation

// MARK: - SSDLoadError

/// Errors thrown by the placeholder-container codec. The header-parse
/// variants (`truncatedHeader`, `invalidHeader`) originate here; the
/// blob/dtype variants (`truncatedBlob`, `unknownDType`) are thrown by
/// the store's Metal-affine `decodePlaceholderContainer`. All variants
/// map to a terminal `loadSync` failure that drops the descriptor and
/// the on-disk file before reporting a miss.
nonisolated enum SSDLoadError: Error {
    case truncatedHeader
    case truncatedBlob
    case invalidHeader(String)
    case unknownDType(String)
}

// MARK: - Placeholder container header

/// Codable header for the placeholder container. Pinned with the
/// full `PersistedSnapshotDescriptor` so a directory walk can
/// rebuild the authoritative manifest after a `manifest.json`
/// corruption: every descriptor field needed to reconstruct the
/// radix-tree shape + LRU bookkeeping survives in each file.
///
/// `PartitionMeta` is deliberately NOT duplicated per file — it
/// lives in `partitions/{digest}/_meta.json` so the rebuild can
/// validate the partition's fingerprint without paying per-file
/// duplication. See `SnapshotLedger.registerPartition(_:digest:)` and
/// `SnapshotLedger`'s `rebuildManifestFromDirectoryWalk`.
nonisolated struct PlaceholderContainerHeader: Codable, Sendable {
    let formatKind: String
    let schemaVersion: Int
    let descriptor: PersistedSnapshotDescriptor
    let layers: [Layer]

    nonisolated struct Layer: Codable, Sendable {
        let className: String
        let metaState: [String]
        let offset: Int
        let arrays: [ArrayEntry]

        enum CodingKeys: String, CodingKey {
            case className = "class_name"
            case metaState = "meta_state"
            case offset
            case arrays
        }
    }

    nonisolated struct ArrayEntry: Codable, Sendable {
        let dtype: String
        let shape: [Int]
        let byteOffset: Int
        let byteSize: Int

        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case byteOffset = "byte_offset"
            case byteSize = "byte_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case formatKind = "format_kind"
        case schemaVersion = "schema_version"
        case descriptor
        case layers
    }

    /// Parse the 8-byte length prefix + JSON header from an in-memory
    /// container. Returned `blobsStart` points at the first byte after
    /// the header (same as `data.startIndex + 8 + headerLength` when
    /// `data` is a fresh `Data`). Used by the store's
    /// `decodePlaceholderContainer` (full hydration); does **not** touch
    /// the tensor payload.
    nonisolated static func parse(
        from data: Data
    ) throws -> (header: PlaceholderContainerHeader, blobsStart: Int) {
        guard data.count >= 8 else { throw SSDLoadError.truncatedHeader }
        let headerLength = data.prefix(8).withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        let headerEnd = 8 + Int(headerLength)
        guard headerEnd <= data.count else { throw SSDLoadError.truncatedHeader }
        let headerData = data[8..<headerEnd]
        let header: PlaceholderContainerHeader
        do {
            header = try JSONDecoder().decode(
                PlaceholderContainerHeader.self,
                from: headerData
            )
        } catch {
            throw SSDLoadError.invalidHeader(String(describing: error))
        }
        return (header, headerEnd)
    }

    /// Read only the header section of an on-disk container file via a
    /// streaming `FileHandle`, sidestepping the tensor payload so the
    /// directory-walk rebuild stays fast even with hundreds of
    /// snapshots. Returns `nil` on any read or decode failure; the
    /// caller deletes the file.
    nonisolated static func readHeaderOnly(
        from url: URL
    ) -> PlaceholderContainerHeader? {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            return nil
        }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8),
              lengthData.count == 8
        else { return nil }
        let headerLength = lengthData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLength <= UInt64(Int.max),
              let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength)
        else { return nil }
        return try? JSONDecoder().decode(
            PlaceholderContainerHeader.self,
            from: headerData
        )
    }
}

// MARK: - Encode

/// Serialize a payload into a single byte blob following the
/// placeholder container layout. The header carries the full descriptor
/// so warm start can rebuild the manifest from a directory walk when
/// `manifest.json` is corrupt. Shared by the store's `writePayload`.
nonisolated func encodePlaceholderContainer(
    payload: SnapshotPayload,
    descriptor: PersistedSnapshotDescriptor
) throws -> Data {
    var layerHeaders: [PlaceholderContainerHeader.Layer] = []
    layerHeaders.reserveCapacity(payload.layers.count)
    var blobs: [Data] = []
    var runningByteOffset = 0

    for layer in payload.layers {
        var arrayEntries: [PlaceholderContainerHeader.ArrayEntry] = []
        arrayEntries.reserveCapacity(layer.state.count)
        for array in layer.state {
            arrayEntries.append(.init(
                dtype: array.dtype,
                shape: array.shape,
                byteOffset: runningByteOffset,
                byteSize: array.data.count
            ))
            runningByteOffset += array.data.count
            blobs.append(array.data)
        }
        layerHeaders.append(.init(
            className: layer.className,
            metaState: layer.metaState,
            offset: layer.offset,
            arrays: arrayEntries
        ))
    }

    let header = PlaceholderContainerHeader(
        formatKind: "tesseract-cache-v1",
        schemaVersion: SnapshotManifestSchema.currentVersion,
        descriptor: descriptor,
        layers: layerHeaders
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let headerData = try encoder.encode(header)

    var out = Data(capacity: 8 + headerData.count + runningByteOffset)
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { out.append(contentsOf: $0) }
    out.append(headerData)
    for blob in blobs {
        out.append(blob)
    }
    return out
}
