//
//  SchedulingService.swift
//  tesseract
//

import Combine
import Foundation

@Observable @MainActor
final class SchedulingService {

    // MARK: - Observable UI State

    private(set) var tasks: [ScheduledTask] = []
    private(set) var currentlyRunningTaskId: UUID? = nil
    private(set) var isPaused: Bool = false
    private(set) var heartbeatStatus: HeartbeatStatus = .idle
    private(set) var unreadResultCount: Int = 0
    private(set) var runHistory: [UUID: [TaskRun]] = [:]

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
        isPaused = await schedulingActor.isPaused

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
        storeSink?.cancel()
        storeSink = nil
        await schedulingActor.stopPolling()
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

    func createTask(_ task: ScheduledTask) {
        taskStore.save(task)
        syncFromStore()
    }

    func updateTask(_ task: ScheduledTask) {
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

    func resumeTask(id: UUID) {
        guard var task = taskStore.loadTask(id: id) else { return }
        task.enabled = true
        task.nextRunAt = task.computeNextRunAt()
        taskStore.save(task)
        syncFromStore()
    }

    // MARK: - Global Pause

    func pauseAll() {
        isPaused = true
        Task { await schedulingActor.pause() }
    }

    func resumeAll() {
        isPaused = false
        Task { await schedulingActor.resume() }
    }

    // MARK: - Run History

    func loadRunHistory(for taskId: UUID) {
        runHistory[taskId] = taskStore.loadRuns(for: taskId)
    }
}
