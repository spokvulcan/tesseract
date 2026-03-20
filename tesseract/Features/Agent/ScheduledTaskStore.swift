//
//  ScheduledTaskStore.swift
//  tesseract
//

import Combine
import Foundation

@MainActor
final class ScheduledTaskStore: ObservableObject {

    @Published private(set) var tasks: [ScheduledTaskSummary] = []

    private let baseDir: URL
    private let tasksDir: URL
    private let runsDir: URL

    private static let storageVersion = 1

    // MARK: - Init

    init() {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        baseDir = appSupport
            .appendingPathComponent("Tesseract Agent/agent/scheduled-tasks", isDirectory: true)
        tasksDir = baseDir.appendingPathComponent("tasks", isDirectory: true)
        runsDir = baseDir.appendingPathComponent("runs", isDirectory: true)
        commonInit()
        seedHeartbeatIfNeeded()
    }

    /// Test-friendly initializer that uses a custom base directory.
    init(baseDirectory: URL) {
        baseDir = baseDirectory
        tasksDir = baseDir.appendingPathComponent("tasks", isDirectory: true)
        runsDir = baseDir.appendingPathComponent("runs", isDirectory: true)
        commonInit()
    }

    private func commonInit() {
        ensureDirectories()
        migrateStorageVersionIfNeeded()
        loadOrResetIndex()
    }

    // MARK: - Heartbeat

    static let defaultHeartbeatTemplate = """
    # Heartbeat Checklist

    - Read tasks.md and check for tasks that have been pending too long — nudge the user to complete or reschedule them
    - Review scheduled tasks with cron_list and flag any that are failing or paused
    - Check if the user has completed any tasks today — if not, encourage them to pick one and start
    - Read memories.md for any time-sensitive commitments or deadlines the user mentioned
    - If it's late in the day and no tasks were completed, suggest a quick end-of-day review
    """

    var heartbeatFileURL: URL {
        baseDir.appendingPathComponent("heartbeat.md")
    }

    func loadHeartbeatChecklist() -> String? {
        let url = heartbeatFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    private func seedHeartbeatIfNeeded() {
        let url = heartbeatFileURL
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            try Self.defaultHeartbeatTemplate.write(to: url, atomically: true, encoding: .utf8)
            Log.agent.info("Seeded default heartbeat checklist at \(url.path)")
        } catch {
            Log.agent.error("Failed to seed heartbeat checklist: \(error)")
        }
    }

    // MARK: - Public API

    func loadTask(id: UUID) -> ScheduledTask? {
        let fileURL = taskFileURL(for: id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(ScheduledTask.self, from: data)
        } catch {
            Log.agent.error("Failed to load scheduled task \(id): \(error)")
            return nil
        }
    }

    /// Loads all full tasks, reconciling `nextRunAt` from the index (the
    /// authoritative source for missed-run detection).  A crash between
    /// writing the task file and the index can leave them divergent;
    /// the index value wins so the scheduler recovers correctly.
    func loadAll() -> [ScheduledTask] {
        let summaryByID = Dictionary(uniqueKeysWithValues: tasks.map { ($0.id, $0) })
        return tasks.compactMap { summary -> ScheduledTask? in
            guard var task = loadTask(id: summary.id) else { return nil }
            task.nextRunAt = summaryByID[task.id]?.nextRunAt
            return task
        }
    }

    func save(_ task: ScheduledTask) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(task)
            try data.write(to: taskFileURL(for: task.id), options: .atomic)
        } catch {
            Log.agent.error("Failed to save scheduled task \(task.id): \(error)")
            return
        }

        let summary = ScheduledTaskSummary(from: task)
        if let idx = tasks.firstIndex(where: { $0.id == task.id }) {
            tasks[idx] = summary
        } else {
            tasks.append(summary)
        }
        saveIndex()
    }

    func delete(id: UUID) {
        try? FileManager.default.removeItem(at: taskFileURL(for: id))

        let taskRunsDir = runsDir.appendingPathComponent(id.uuidString, isDirectory: true)
        try? FileManager.default.removeItem(at: taskRunsDir)

        tasks.removeAll { $0.id == id }
        saveIndex()
    }

    func saveRun(_ run: TaskRun) {
        let taskRunsDir = runsDir.appendingPathComponent(run.taskId.uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: taskRunsDir, withIntermediateDirectories: true)

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(run)
            let fileURL = taskRunsDir.appendingPathComponent("\(run.id.uuidString).json")
            try data.write(to: fileURL, options: .atomic)
        } catch {
            Log.agent.error("Failed to save task run \(run.id) for task \(run.taskId): \(error)")
        }
    }

    func markRunNotified(runId: UUID, taskId: UUID) {
        updateRun(runId: runId, taskId: taskId, { $0.notifiedUser = true }, operation: "mark as notified")
    }

    func markRunSpoken(runId: UUID, taskId: UUID) {
        updateRun(runId: runId, taskId: taskId, { $0.spokeResult = true }, operation: "mark as spoken")
    }

    func markRunCompletion(runId: UUID, taskId: UUID, notified: Bool, spoken: Bool) {
        guard notified || spoken else { return }
        updateRun(runId: runId, taskId: taskId, { run in
            if notified { run.notifiedUser = true }
            if spoken { run.spokeResult = true }
        }, operation: "update completion flags")
    }

    private func updateRun(runId: UUID, taskId: UUID, _ mutate: (inout TaskRun) -> Void, operation: String) {
        let taskRunsDir = runsDir.appendingPathComponent(taskId.uuidString, isDirectory: true)
        let fileURL = taskRunsDir.appendingPathComponent("\(runId.uuidString).json")

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            var run = try decoder.decode(TaskRun.self, from: data)
            mutate(&run)
            saveRun(run)
        } catch {
            Log.agent.error("Failed to \(operation) run \(runId): \(error)")
        }
    }

    func loadRuns(for taskId: UUID) -> [TaskRun] {
        let taskRunsDir = runsDir.appendingPathComponent(taskId.uuidString, isDirectory: true)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: taskRunsDir,
            includingPropertiesForKeys: nil
        ) else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> TaskRun? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(TaskRun.self, from: data)
            }
            .sorted { $0.startedAt > $1.startedAt }
    }

    func updateAfterRun(taskId: UUID, run: TaskRun) {
        saveRun(run)

        guard var task = loadTask(id: taskId) else {
            Log.agent.error("Cannot update after run — task \(taskId) not found")
            return
        }

        task.lastRunAt = run.startedAt
        task.lastRunResult = run.result
        task.runCount += 1
        task.nextRunAt = task.computeNextRunAt()

        if task.isExhausted {
            task.enabled = false
        }

        save(task)
    }

    // MARK: - Private

    private func ensureDirectories() {
        try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: tasksDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: runsDir, withIntermediateDirectories: true)
    }

    private var versionFileURL: URL {
        baseDir.appendingPathComponent(".storage_version")
    }

    private func migrateStorageVersionIfNeeded() {
        let currentOnDisk = (try? String(contentsOf: versionFileURL, encoding: .utf8))
            .flatMap(Int.init) ?? 0

        if currentOnDisk == Self.storageVersion { return }

        Log.agent.info(
            "Scheduled task storage version mismatch (disk=\(currentOnDisk), app=\(Self.storageVersion)) — clearing data"
        )
        try? FileManager.default.removeItem(at: baseDir)
        ensureDirectories()
        try? String(Self.storageVersion).write(to: versionFileURL, atomically: true, encoding: .utf8)
        tasks = []
    }

    private var indexURL: URL {
        baseDir.appendingPathComponent("index.json")
    }

    private func taskFileURL(for id: UUID) -> URL {
        tasksDir.appendingPathComponent("\(id.uuidString).json")
    }

    /// Checks whether a task file exists AND decodes successfully.
    private func canLoadTask(id: UUID) -> Bool {
        let fileURL = taskFileURL(for: id)
        guard let data = try? Data(contentsOf: fileURL) else { return false }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode(ScheduledTask.self, from: data)) != nil
    }

    private func loadOrResetIndex() {
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return }
        do {
            let data = try Data(contentsOf: indexURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let index = try decoder.decode(ScheduledTaskIndex.self, from: data)

            let valid = index.tasks.filter { canLoadTask(id: $0.id) }
            tasks = valid

            if valid.count < index.tasks.count {
                Log.agent.info(
                    "Pruned \(index.tasks.count - valid.count) orphaned/corrupt scheduled task(s) from index"
                )
                saveIndex()
            }
        } catch {
            Log.agent.error("Failed to decode scheduled task index: \(error)")
            tasks = []
            saveIndex()
        }
    }

    private func saveIndex() {
        do {
            let index = ScheduledTaskIndex(version: Self.storageVersion, tasks: tasks)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(index)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            Log.agent.error("Failed to save scheduled task index: \(error)")
        }
    }
}
