//
//  SchedulingService.swift
//  tesseract
//

import Combine
import Foundation

// MARK: - SchedulingError

nonisolated enum SchedulingError: LocalizedError, Sendable {
    case maxActiveTasksReached(limit: Int)
    case agentTurnLimitReached(limit: Int)

    var errorDescription: String? {
        switch self {
        case .maxActiveTasksReached(let limit):
            "Maximum active tasks limit reached (\(limit)). Disable or delete existing tasks first."
        case .agentTurnLimitReached(let limit):
            "Agent limit reached: maximum \(limit) tasks per conversation turn."
        }
    }
}

// MARK: - SchedulingService

@Observable @MainActor
final class SchedulingService {

    // MARK: - Observable UI State

    private(set) var tasks: [ScheduledTask] = []
    private(set) var currentlyRunningTaskId: UUID? = nil
    private(set) var isPaused: Bool = false
    private(set) var heartbeatStatus: HeartbeatStatus = .idle
    private(set) var unreadResultCount: Int = 0
    private(set) var runHistory: [UUID: [TaskRun]] = [:]
    private var agentTasksCreatedThisTurn: Int = 0

    private static let globalPauseKey = "scheduling.globalPause"

    // MARK: - Dependencies

    @ObservationIgnored private let schedulingActor: SchedulingActor
    @ObservationIgnored private let taskStore: ScheduledTaskStore
    @ObservationIgnored private var storeSink: AnyCancellable?

    // MARK: - Init

    init(actor: SchedulingActor, store: ScheduledTaskStore) {
        self.schedulingActor = actor
        self.taskStore = store
    }

    // MARK: - Lifecycle

    func start() async {
        tasks = taskStore.loadAll()

        let wasPaused = UserDefaults.standard.bool(forKey: Self.globalPauseKey)
        if wasPaused {
            isPaused = true
            await schedulingActor.pause()
        } else {
            isPaused = await schedulingActor.isPaused
        }

        await schedulingActor.setOnRunningTaskChanged { [weak self] taskId in
            self?.currentlyRunningTaskId = taskId
        }

        storeSink = taskStore.$tasks
            .dropFirst()
            .sink { [weak self] _ in
                self?.syncFromStore()
            }

        await schedulingActor.startPolling()
    }

    func stop() async {
        cancelStoreSink()
        await schedulingActor.stopPolling()
    }

    /// Cancel the Combine subscription synchronously. Called from `applicationWillTerminate`
    /// (already on MainActor) before blocking on actor work to avoid deadlock.
    func cancelStoreSink() {
        storeSink?.cancel()
        storeSink = nil
    }

    // MARK: - State Sync

    private func syncFromStore() {
        tasks = taskStore.loadAll()

        let taskIds = Set(tasks.map(\.id))
        for key in runHistory.keys {
            if !taskIds.contains(key) {
                runHistory[key] = nil
            } else if runHistory[key] != nil {
                // Refresh cached histories so completed runs appear live
                runHistory[key] = taskStore.loadRuns(for: key)
            }
        }
    }

    // MARK: - Task CRUD

    func createTask(_ task: ScheduledTask) throws {
        try enforceActiveTaskLimit()
        if task.createdBy.isAgent {
            if agentTasksCreatedThisTurn >= SchedulingActor.maxAgentTasksPerTurn {
                throw SchedulingError.agentTurnLimitReached(limit: SchedulingActor.maxAgentTasksPerTurn)
            }
            agentTasksCreatedThisTurn += 1
        }
        taskStore.save(task)
        syncFromStore()
    }

    func updateTask(_ task: ScheduledTask) throws {
        // If enabling a previously-disabled task, enforce the active cap
        if task.enabled {
            let wasEnabled = tasks.first(where: { $0.id == task.id })?.enabled ?? false
            if !wasEnabled {
                try enforceActiveTaskLimit()
            }
        }
        taskStore.save(task)
        syncFromStore()
    }

    func deleteTask(id: UUID) {
        taskStore.delete(id: id)
        syncFromStore()
    }

    func pauseTask(id: UUID) {
        guard var task = taskStore.loadTask(id: id) else { return }
        task.enabled = false
        taskStore.save(task)
        syncFromStore()
    }

    func resumeTask(id: UUID) throws {
        guard var task = taskStore.loadTask(id: id) else { return }
        try enforceActiveTaskLimit()
        task.enabled = true
        task.nextRunAt = task.computeNextRunAt()
        taskStore.save(task)
        syncFromStore()
    }

    // MARK: - Safety Checks

    private func enforceActiveTaskLimit() throws {
        let activeCount = tasks.filter(\.enabled).count
        if activeCount >= SchedulingActor.maxActiveTasks {
            throw SchedulingError.maxActiveTasksReached(limit: SchedulingActor.maxActiveTasks)
        }
    }

    // MARK: - Global Pause

    func pauseAll() {
        isPaused = true
        UserDefaults.standard.set(true, forKey: Self.globalPauseKey)
        Task { await schedulingActor.pause() }
    }

    func resumeAll() {
        isPaused = false
        UserDefaults.standard.set(false, forKey: Self.globalPauseKey)
        Task { await schedulingActor.resume() }
    }

    // MARK: - Agent Turn Counter

    func resetAgentTurnCounter() {
        agentTasksCreatedThisTurn = 0
    }

    // MARK: - Run History

    func loadRunHistory(for taskId: UUID) {
        runHistory[taskId] = taskStore.loadRuns(for: taskId)
    }
}
