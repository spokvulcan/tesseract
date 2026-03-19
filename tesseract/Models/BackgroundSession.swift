//
//  BackgroundSession.swift
//  tesseract
//

import Foundation

// MARK: - SessionType

nonisolated enum SessionType: String, Codable, Sendable {
    case heartbeat
    case cron
}

// MARK: - BackgroundSession

nonisolated struct BackgroundSession: Codable, Identifiable, Sendable {
    let id: UUID
    var sessionType: SessionType
    var displayName: String
    var taskId: UUID
    var messages: [TaggedMessage]
    var lastRunAt: Date?
    var createdAt: Date
}

// MARK: - BackgroundSessionSummary

nonisolated struct BackgroundSessionSummary: Codable, Identifiable, Sendable {
    let id: UUID
    var displayName: String
    var sessionType: SessionType
    var taskId: UUID
    var lastRunAt: Date?
    var createdAt: Date
    var messageCount: Int

    init(from session: BackgroundSession) {
        self.id = session.id
        self.displayName = session.displayName
        self.sessionType = session.sessionType
        self.taskId = session.taskId
        self.lastRunAt = session.lastRunAt
        self.createdAt = session.createdAt
        self.messageCount = session.messages.count
    }
}

// MARK: - BackgroundSessionIndex

nonisolated struct BackgroundSessionIndex: Codable, Sendable {
    let version: Int
    var sessions: [BackgroundSessionSummary]
}
