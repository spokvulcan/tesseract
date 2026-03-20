//
//  SchedulingActor.swift
//  tesseract
//

import Foundation

/// Off-MainActor actor that owns the 60-second polling loop, due-task detection,
/// sequential execution queue, missed-run recovery, and safety controls.
///
/// Communicates with `ScheduledTaskStore` (`@MainActor`) via `await`.
/// Follows the `ContextManager` actor pattern.
actor SchedulingActor {

    // MARK: - Constants

    static let maxConsecutiveFailures = 5
    static let maxActiveTasks = 50
    static let maxAgentTasksPerTurn = 10
    static let minimumIntervalSeconds: TimeInterval = 300
    static let missedRunCatchUpThreshold: TimeInterval = 3600
    static let heartbeatSessionId = UUID(uuidString: "00000001-0000-0000-0000-000000000001")!
    static let defaultHeartbeatIntervalSeconds: TimeInterval = 1800

    // MARK: - Dependencies

    private let taskStore: ScheduledTaskStore
    private let executeTask: @Sendable (ScheduledTask) async -> TaskRunResult
    private let executeHeartbeat: @Sendable (ScheduledTask) async -> TaskRunResult
    private let persistInFlightSession: @Sendable () async -> Void
    private var onRunningTaskChanged: (@MainActor @Sendable (UUID?) -> Void)?

    // MARK: - State

    private var pollTask: Task<Void, Never>?
    private(set) var isPaused: Bool = false
    private(set) var currentlyRunningTask: ScheduledTask?
    private var currentRunStartedAt: Date?
    private var isShuttingDown: Bool = false
    private var executionQueue: [ScheduledTask] = []
    private var isDraining: Bool = false
    private var consecutiveFailureCount: [UUID: Int] = [:]
    private var caughtUpTaskIds: Set<UUID> = []
    private var heartbeatTask: Task<Void, Never>?
    private var onHeartbeatStatusChanged: (@MainActor @Sendable (HeartbeatStatus) -> Void)?

    // MARK: - Init

    init(
        taskStore: ScheduledTaskStore,
        executeTask: @escaping @Sendable (ScheduledTask) async -> TaskRunResult,
        executeHeartbeat: @escaping @Sendable (ScheduledTask) async -> TaskRunResult,
        persistInFlightSession: @escaping @Sendable () async -> Void
    ) {
        self.taskStore = taskStore
        self.executeTask = executeTask
        self.executeHeartbeat = executeHeartbeat
        self.persistInFlightSession = persistInFlightSession
    }

    // MARK: - Polling

    func startPolling() {
        guard pollTask == nil else { return }
        Log.agent.info("SchedulingActor: polling started")
        pollTask = Task {
            await detectMissedRuns()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                guard !Task.isCancelled else { break }
                await checkAndRunDueTasks()
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    func setOnRunningTaskChanged(_ callback: @escaping @MainActor @Sendable (UUID?) -> Void) {
        onRunningTaskChanged = callback
    }

    func setOnHeartbeatStatusChanged(_ callback: @escaping @MainActor @Sendable (HeartbeatStatus) -> Void) {
        onHeartbeatStatusChanged = callback
    }

    // MARK: - Heartbeat

    func startHeartbeat(intervalSeconds: TimeInterval = SchedulingActor.defaultHeartbeatIntervalSeconds) {
        guard heartbeatTask == nil else { return }
        let clamped = max(intervalSeconds, Self.minimumIntervalSeconds)
        if clamped != intervalSeconds {
            Log.agent.warning("SchedulingActor: heartbeat interval \(Int(intervalSeconds))s below minimum, clamped to \(Int(clamped))s")
        }
        Log.agent.info("SchedulingActor: heartbeat started (interval: \(Int(clamped))s)")
        heartbeatTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(clamped))
                guard !Task.isCancelled else { break }
                guard !isPaused else { continue }
                await enqueueHeartbeat()
            }
        }
    }

    func stopHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = nil
    }

    /// Builds the synthetic heartbeat task and drops it into the shared execution queue.
    /// Actual execution is serialized with cron tasks via `processQueue`.
    private func enqueueHeartbeat() async {
        guard let checklist = await taskStore.loadHeartbeatChecklist() else {
            Log.agent.debug("SchedulingActor: heartbeat skipped — no heartbeat.md")
            return
        }

        let syntheticTask = ScheduledTask(
            id: Self.heartbeatSessionId,
            name: "Heartbeat",
            description: "Periodic heartbeat evaluation",
            cronExpression: "",
            prompt: checklist,
            enabled: true,
            createdBy: .user,
            createdAt: Date(),
            lastRunAt: nil, lastRunResult: nil, nextRunAt: nil,
            runCount: 0, maxRuns: nil, tags: [],
            notifyUser: false, speakResult: false,
            sessionId: Self.heartbeatSessionId
        )

        enqueue(syntheticTask)
    }

    func pause() { isPaused = true }
    func resume() {
        isPaused = false
        if !executionQueue.isEmpty && !isDraining {
            Task { await processQueue() }
        }
    }

    // MARK: - Due-Task Detection

    func checkAndRunDueTasks(now: Date = Date()) async {
        guard !isPaused else { return }
        let allTasks = await taskStore.loadAll()
        let dueTasks = allTasks.filter { $0.enabled && !$0.isExhausted && isDue($0, at: now) }
        for task in dueTasks { enqueue(task) }
    }

    private func isDue(_ task: ScheduledTask, at now: Date) -> Bool {
        guard let nextRun = task.nextRunAt else { return false }
        return nextRun <= now
    }

    // MARK: - Sequential Execution Queue

    private func enqueue(_ task: ScheduledTask) {
        guard !executionQueue.contains(where: { $0.id == task.id }) else { return }
        guard currentlyRunningTask?.id != task.id else { return }
        executionQueue.append(task)
        if !isDraining {
            Task { await processQueue() }
        }
    }

    private func processQueue() async {
        guard !isDraining else { return }
        isDraining = true
        defer { isDraining = false }
        while let task = executionQueue.first {
            guard !isPaused else { break }
            executionQueue.removeFirst()

            if task.id == Self.heartbeatSessionId {
                // Heartbeat: synthetic task, not in store — run directly
                await runHeartbeat(task)
            } else {
                // Cron: re-read from store to catch pause/delete that happened after enqueue
                guard let current = await taskStore.loadTask(id: task.id),
                      current.enabled, !current.isExhausted else { continue }
                await runTask(current)
            }
        }
    }

    private func runTask(_ task: ScheduledTask) async {
        currentlyRunningTask = task
        currentRunStartedAt = Date()
        await onRunningTaskChanged?(task.id)

        let startedAt = currentRunStartedAt!
        let result = await executeTask(task)

        // If shutdown persisted an .interrupted record while we were suspended,
        // skip the normal updateAfterRun to avoid a duplicate run record.
        guard !isShuttingDown else {
            currentlyRunningTask = nil
            currentRunStartedAt = nil
            await onRunningTaskChanged?(nil)
            return
        }

        let completedAt = Date()
        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: startedAt, completedAt: completedAt,
            durationSeconds: Int(completedAt.timeIntervalSince(startedAt)),
            result: result, summary: result.displaySummary,
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )

        await taskStore.updateAfterRun(taskId: task.id, run: run)
        await updateFailureTracking(taskId: task.id, result: result)
        currentlyRunningTask = nil
        currentRunStartedAt = nil
        await onRunningTaskChanged?(nil)
    }

    // MARK: - Heartbeat Execution

    /// Executes a heartbeat task within the shared queue, tracking it as the
    /// currently-running task so shutdown can detect and handle it.
    private func runHeartbeat(_ task: ScheduledTask) async {
        currentlyRunningTask = task
        currentRunStartedAt = Date()
        await onHeartbeatStatusChanged?(.checking)

        let result = await executeHeartbeat(task)

        guard !isShuttingDown else {
            currentlyRunningTask = nil
            currentRunStartedAt = nil
            return
        }

        await onHeartbeatStatusChanged?(.lastRun(Date()))
        Log.agent.info("SchedulingActor: heartbeat completed — \(result.displaySummary)")
        currentlyRunningTask = nil
        currentRunStartedAt = nil
    }

    // MARK: - Missed-Run Detection

    func detectMissedRuns(now: Date = Date()) async {
        guard !isPaused else { return }
        let allTasks = await taskStore.loadAll()
        for task in allTasks where task.enabled && !task.isExhausted {
            guard let nextRunAt = task.nextRunAt, nextRunAt < now else { continue }
            let missedBy = now.timeIntervalSince(nextRunAt)

            if missedBy < Self.missedRunCatchUpThreshold {
                if !caughtUpTaskIds.contains(task.id) {
                    caughtUpTaskIds.insert(task.id)
                    enqueue(task)
                }
            } else {
                let missedResult = TaskRunResult.missed(at: nextRunAt)
                let missedRun = TaskRun(
                    id: UUID(), taskId: task.id, sessionId: task.sessionId,
                    startedAt: nextRunAt, completedAt: nextRunAt, durationSeconds: 0,
                    result: missedResult, summary: missedResult.displaySummary,
                    notifiedUser: false, spokeResult: false, tokensUsed: nil
                )
                await taskStore.saveRun(missedRun)
            }

            // AFTER evaluating miss, advance nextRunAt
            var updated = task
            updated.nextRunAt = task.computeNextRunAt(after: now)
            await taskStore.save(updated)
        }
    }

    // MARK: - Safety Controls

    private func updateFailureTracking(taskId: UUID, result: TaskRunResult) async {
        switch result {
        case .error:
            let count = (consecutiveFailureCount[taskId] ?? 0) + 1
            consecutiveFailureCount[taskId] = count
            if count >= Self.maxConsecutiveFailures {
                await autoPauseTask(taskId: taskId)
            }
        case .success, .noActionNeeded:
            consecutiveFailureCount[taskId] = 0
        case .interrupted, .missed:
            break
        }
    }

    private func autoPauseTask(taskId: UUID) async {
        guard var task = await taskStore.loadTask(id: taskId) else { return }
        task.enabled = false
        await taskStore.save(task)
        Log.agent.info(
            "SchedulingActor: auto-paused task '\(task.name)' after \(Self.maxConsecutiveFailures) consecutive failures"
        )
    }

    // MARK: - Shutdown

    /// Persists an `.interrupted` run record for the currently-running task (if any)
    /// and advances `nextRunAt` so the same occurrence isn't replayed as missed on next launch.
    /// Sets `isShuttingDown` so that `runTask` skips its own `updateAfterRun` if it resumes.
    func persistInterruptedTask() async {
        isShuttingDown = true
        guard let task = currentlyRunningTask else { return }

        // Flush accumulated session transcript before the app terminates.
        // This preserves messages from the partial run for both cron and heartbeat.
        await persistInFlightSession()

        // Heartbeat is a synthetic task with no store entry — session flush above
        // is sufficient. No TaskRun record to save.
        if task.id == Self.heartbeatSessionId {
            Log.agent.info("SchedulingActor: heartbeat interrupted during shutdown")
            return
        }

        let now = Date()
        let startedAt = currentRunStartedAt ?? now
        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: startedAt, completedAt: now,
            durationSeconds: Int(now.timeIntervalSince(startedAt)),
            result: .interrupted, summary: TaskRunResult.interrupted.displaySummary,
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        await taskStore.updateAfterRun(taskId: task.id, run: run)
        Log.agent.info("SchedulingActor: persisted interrupted run for '\(task.name)'")
    }

    // MARK: - State Queries

    var queueDepth: Int { executionQueue.count }
    func consecutiveFailures(for taskId: UUID) -> Int { consecutiveFailureCount[taskId] ?? 0 }
}
