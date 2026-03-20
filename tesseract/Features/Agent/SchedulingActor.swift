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

    // MARK: - Dependencies

    private let taskStore: ScheduledTaskStore
    private let executeTask: @Sendable (ScheduledTask) async -> TaskRunResult
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

    // MARK: - Init

    init(
        taskStore: ScheduledTaskStore,
        executeTask: @escaping @Sendable (ScheduledTask) async -> TaskRunResult
    ) {
        self.taskStore = taskStore
        self.executeTask = executeTask
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
            // Re-read from store to catch pause/delete that happened after enqueue
            guard let current = await taskStore.loadTask(id: task.id),
                  current.enabled, !current.isExhausted else { continue }
            await runTask(current)
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
