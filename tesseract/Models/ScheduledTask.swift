//
//  ScheduledTask.swift
//  tesseract
//

import Foundation

// MARK: - ScheduledTaskError

nonisolated enum ScheduledTaskError: LocalizedError, Sendable {
    case invalidCronExpression(String, underlying: String)

    var errorDescription: String? {
        switch self {
        case .invalidCronExpression(let expr, let underlying):
            "Invalid cron expression '\(expr)': \(underlying)"
        }
    }
}

// MARK: - TaskCreator

nonisolated enum TaskCreator: Sendable, Equatable {
    case user
    case agent(reason: String)
}

nonisolated extension TaskCreator: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, reason
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "user":
            self = .user
        case "agent":
            self = .agent(reason: try container.decode(String.self, forKey: .reason))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "Unknown TaskCreator type: \(type)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .user:
            try container.encode("user", forKey: .type)
        case .agent(let reason):
            try container.encode("agent", forKey: .type)
            try container.encode(reason, forKey: .reason)
        }
    }
}

// MARK: - TaskRunResult

nonisolated enum TaskRunResult: Sendable, Equatable {
    case success(summary: String)
    case noActionNeeded
    case error(message: String)
    case interrupted
    case missed(at: Date)

    var displaySummary: String {
        switch self {
        case .success(let s): s
        case .noActionNeeded: "No action needed"
        case .error(let m): "Error: \(m)"
        case .interrupted: "Interrupted"
        case .missed(let at): "Missed at \(at)"
        }
    }
}

nonisolated extension TaskRunResult: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, summary, message, at
    }

    private nonisolated(unsafe) static let iso8601Formatter: ISO8601DateFormatter = {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fmt
    }()

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "success":
            self = .success(summary: try container.decode(String.self, forKey: .summary))
        case "noActionNeeded":
            self = .noActionNeeded
        case "error":
            self = .error(message: try container.decode(String.self, forKey: .message))
        case "interrupted":
            self = .interrupted
        case "missed":
            let dateString = try container.decode(String.self, forKey: .at)
            guard let date = Self.iso8601Formatter.date(from: dateString) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .at, in: container,
                    debugDescription: "Invalid ISO8601 date: \(dateString)"
                )
            }
            self = .missed(at: date)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "Unknown TaskRunResult type: \(type)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .success(let summary):
            try container.encode("success", forKey: .type)
            try container.encode(summary, forKey: .summary)
        case .noActionNeeded:
            try container.encode("noActionNeeded", forKey: .type)
        case .error(let message):
            try container.encode("error", forKey: .type)
            try container.encode(message, forKey: .message)
        case .interrupted:
            try container.encode("interrupted", forKey: .type)
        case .missed(let date):
            try container.encode("missed", forKey: .type)
            try container.encode(Self.iso8601Formatter.string(from: date), forKey: .at)
        }
    }
}

// MARK: - ScheduledTask

nonisolated struct ScheduledTask: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    var name: String
    var description: String
    var cronExpression: String
    var prompt: String
    var enabled: Bool
    var createdBy: TaskCreator
    var createdAt: Date
    var lastRunAt: Date?
    var lastRunResult: TaskRunResult?
    var nextRunAt: Date?
    var runCount: Int
    var maxRuns: Int?
    var tags: [String]
    var notifyUser: Bool
    var speakResult: Bool
    var sessionId: UUID
}

// MARK: - ScheduledTask Extensions

nonisolated extension ScheduledTask {

    var parsedCronExpression: CronExpression? {
        try? CronExpression(parsing: cronExpression)
    }

    func computeNextRunAt(after date: Date = Date(), in timeZone: TimeZone = .current) -> Date? {
        parsedCronExpression?.nextOccurrence(after: date, in: timeZone)
    }

    var humanReadableSchedule: String {
        parsedCronExpression?.humanReadable ?? cronExpression
    }

    var isExhausted: Bool {
        if let maxRuns { return runCount >= maxRuns }
        return false
    }

    static func create(
        name: String,
        cronExpression: String,
        prompt: String,
        description: String = "",
        enabled: Bool = true,
        createdBy: TaskCreator = .user,
        maxRuns: Int? = nil,
        tags: [String] = [],
        notifyUser: Bool = true,
        speakResult: Bool = false,
        sessionId: UUID = UUID()
    ) throws -> ScheduledTask {
        let parsed: CronExpression
        do {
            parsed = try CronExpression(parsing: cronExpression)
        } catch {
            throw ScheduledTaskError.invalidCronExpression(
                cronExpression, underlying: error.localizedDescription
            )
        }

        let now = Date()
        let nextRun = parsed.nextOccurrence(after: now)

        return ScheduledTask(
            id: UUID(),
            name: name,
            description: description,
            cronExpression: cronExpression,
            prompt: prompt,
            enabled: enabled,
            createdBy: createdBy,
            createdAt: now,
            lastRunAt: nil,
            lastRunResult: nil,
            nextRunAt: nextRun,
            runCount: 0,
            maxRuns: maxRuns,
            tags: tags,
            notifyUser: notifyUser,
            speakResult: speakResult,
            sessionId: sessionId
        )
    }
}

// MARK: - TaskRun

nonisolated struct TaskRun: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    let taskId: UUID
    let sessionId: UUID
    var startedAt: Date
    var completedAt: Date?
    var durationSeconds: Int?
    var result: TaskRunResult
    var summary: String
    var notifiedUser: Bool
    var spokeResult: Bool
    var tokensUsed: Int?
}

// MARK: - ScheduledTaskSummary

nonisolated struct ScheduledTaskSummary: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    var name: String
    var cronExpression: String
    var enabled: Bool
    var nextRunAt: Date?
    var createdBy: TaskCreator
    var sessionId: UUID

    init(from task: ScheduledTask) {
        self.id = task.id
        self.name = task.name
        self.cronExpression = task.cronExpression
        self.enabled = task.enabled
        self.nextRunAt = task.nextRunAt
        self.createdBy = task.createdBy
        self.sessionId = task.sessionId
    }

    init(
        id: UUID,
        name: String,
        cronExpression: String,
        enabled: Bool,
        nextRunAt: Date?,
        createdBy: TaskCreator,
        sessionId: UUID
    ) {
        self.id = id
        self.name = name
        self.cronExpression = cronExpression
        self.enabled = enabled
        self.nextRunAt = nextRunAt
        self.createdBy = createdBy
        self.sessionId = sessionId
    }
}

// MARK: - ScheduledTaskIndex

nonisolated struct ScheduledTaskIndex: Codable, Sendable {
    let version: Int
    var tasks: [ScheduledTaskSummary]
}
