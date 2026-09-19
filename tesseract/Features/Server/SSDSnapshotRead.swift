import Darwin
import Foundation

/// #532's pre-registered experiment. Production always uses `.mapped`;
/// only the owner-run benchmark selects another arm at store construction.
nonisolated enum SSDSnapshotReadArm: String, CaseIterable, Codable, Sendable {
    case mapped, sequentialMap, positional
}

/// One segment's owned host bytes and the work actually performed on them.
/// Mapping setup alone excludes page faults; the store adds header parsing
/// and every contributing layer's MLX copy before publishing the measurement.
nonisolated struct SSDReadSegment {
    let name: String
    let data: Data
    var readSeconds: Double
    var copySeconds = 0.0
    var materializedBytes = 0

    init(url: URL, arm: SSDSnapshotReadArm) throws {
        let start = ContinuousClock.now
        name = url.lastPathComponent
        switch arm {
        case .mapped:
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        case .sequentialMap, .positional:
            data = try Self.readOwnedBuffer(url: url, arm: arm)
        }
        readSeconds = start.duration(to: .now).seconds
    }

    private static func readOwnedBuffer(url: URL, arm: SSDSnapshotReadArm) throws -> Data {
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw posixError() }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw posixError() }
        guard info.st_mode & S_IFMT == S_IFREG, let count = Int(exactly: info.st_size), count >= 0
        else { throw POSIXError(.EINVAL) }
        guard count > 0 else { return Data() }

        if arm == .sequentialMap {
            let mapped = mmap(nil, count, PROT_READ, MAP_PRIVATE, fd, 0)
            guard let mapped, mapped != MAP_FAILED else { throw posixError() }
            // Apply before header parsing or the first MLX copy. A failed
            // advice is a failed arm, never silently labelled as advised.
            guard madvise(mapped, count, MADV_SEQUENTIAL) == 0 else {
                let error = posixError()
                munmap(mapped, count)
                throw error
            }
            return Data(
                bytesNoCopy: mapped, count: count,
                deallocator: .custom { pointer, size in
                    munmap(pointer, size)
                })
        }

        var buffer: UnsafeMutableRawPointer?
        let alignment = Int(getpagesize())
        let result = posix_memalign(&buffer, alignment, count)
        guard result == 0, let buffer else {
            throw POSIXError(POSIXErrorCode(rawValue: result) ?? .ENOMEM)
        }
        var transferred = 0
        defer { if transferred < count { free(buffer) } }
        let chunkBytes = 8 * 1_024 * 1_024
        while transferred < count {
            let readCount = pread(
                fd, buffer.advanced(by: transferred),
                min(chunkBytes, count - transferred), off_t(transferred))
            if readCount < 0 {
                if errno == EINTR { continue }
                throw posixError()
            }
            guard readCount > 0 else { throw POSIXError(.EIO) }
            transferred += readCount
        }
        // Data owns the page-aligned allocation; materializeLayerArrays
        // performs the same single host-to-MLX copy as the mapped arms.
        return Data(bytesNoCopy: buffer, count: count, deallocator: .free)
    }

    private static func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
