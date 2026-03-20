import SwiftUI

struct ScheduledTasksView: View {
    @Environment(SchedulingService.self) private var schedulingService
    @Environment(AgentCoordinator.self) private var agentCoordinator
    @Environment(SettingsManager.self) private var settings
    
    @State private var showingCreateSheet = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                
                header
                
                if !schedulingService.isPaused {
                    heartbeatCard
                } else {
                    pausedBanner
                }
                
                tasksList
                
                runHistorySection
            }
            .padding(Theme.Spacing.xl)
            .frame(maxWidth: Theme.Layout.contentMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Scheduled Tasks")
        .sheet(isPresented: $showingCreateSheet) {
            Text("Create Task Sheet (Coming Soon)")
                .padding()
        }
        .onAppear {
            schedulingService.markResultsRead()
        }
    }
    
    // MARK: - Header
    
    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Scheduled Tasks")
                    .font(.title)
                    .fontWeight(.semibold)
                Text("Background awareness and recurring agent tasks")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            HStack(spacing: 12) {
                Button {
                    if schedulingService.isPaused {
                        schedulingService.resumeAll()
                    } else {
                        schedulingService.pauseAll()
                    }
                } label: {
                    Label(
                        schedulingService.isPaused ? "Resume All" : "Pause All",
                        systemImage: schedulingService.isPaused ? "play.fill" : "pause.fill"
                    )
                }
                .buttonStyle(.bordered)
                .tint(schedulingService.isPaused ? .green : .orange)
                
                Button {
                    showingCreateSheet = true
                } label: {
                    Label("New Task", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }
    
    // MARK: - Paused Banner
    
    private var pausedBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "pause.circle.fill")
                .font(.title2)
                .foregroundStyle(.orange)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("Scheduling Paused")
                    .font(.headline)
                Text("Heartbeat and scheduled tasks will not run.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }
    
    // MARK: - Heartbeat Card
    
    private var heartbeatCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(settings.heartbeatEnabled ? Color.blue.opacity(0.15) : Color.secondary.opacity(0.1))
                            .frame(width: 36, height: 36)
                        Image(systemName: "waveform.path.ecg")
                            .foregroundStyle(settings.heartbeatEnabled ? .blue : .secondary)
                            .symbolEffect(.pulse, options: .repeating, isActive: schedulingService.heartbeatStatus == .checking)
                    }
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Heartbeat")
                            .font(.headline)
                        if settings.heartbeatEnabled {
                            Text("every \(settings.heartbeatIntervalMinutes)m")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("Disabled")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                
                Spacer()
                
                Toggle("", isOn: Bindable(settings).heartbeatEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
            }
            
            if settings.heartbeatEnabled {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Status")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textCase(.uppercase)
                        
                        switch schedulingService.heartbeatStatus {
                        case .idle:
                            Text("Waiting...")
                        case .checking:
                            Text("Checking...")
                                .foregroundStyle(.blue)
                        case .lastRun(let date):
                            Text("Last run: \(date.formatted(.relative(presentation: .named)))")
                        }
                    }
                    Spacer()
                }
            }
        }
        .padding(Theme.Spacing.lg)
        .glassEffect(in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 4)
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.quaternary, lineWidth: 0.5)
        }
    }
    
    // MARK: - Tasks List
    
    private var tasksList: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("Tasks")
                .font(.headline)
                .padding(.bottom, 4)
            
            if schedulingService.tasks.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "calendar.badge.clock")
                        .font(.system(size: 32))
                        .foregroundStyle(.quaternary)
                    Text("No scheduled tasks")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 40)
                .background(.quinary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
            } else {
                ForEach(schedulingService.tasks) { task in
                    TaskRowView(task: task)
                }
            }
        }
    }
    
    // MARK: - Run History
    
    private var runHistorySection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("Recent Run History")
                .font(.headline)
                .padding(.bottom, 4)
            
            let allRuns = schedulingService.runHistory.values.flatMap { $0 }.sorted(by: { $0.startedAt > $1.startedAt })
            
            if allRuns.isEmpty {
                Text("No runs yet")
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 20)
                    .frame(maxWidth: .infinity, alignment: .center)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(allRuns.prefix(10))) { run in
                        RunHistoryRowView(run: run, taskName: taskName(for: run.taskId))
                        if run.id != allRuns.prefix(10).last?.id {
                            Divider().padding(.leading, 12)
                        }
                    }
                }
                .padding(.vertical, 8)
                .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .shadow(color: .black.opacity(0.03), radius: 4, x: 0, y: 2)
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(.quaternary, lineWidth: 0.5)
                }
            }
        }
    }
    
    private func taskName(for id: UUID) -> String {
        if let task = schedulingService.tasks.first(where: { $0.id == id }) {
            return task.name
        }
        return "Unknown Task"
    }
}

// MARK: - Task Row View

struct TaskRowView: View {
    let task: ScheduledTask
    @Environment(SchedulingService.self) private var schedulingService
    @Environment(AgentCoordinator.self) private var agentCoordinator
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(task.name)
                            .font(.headline)
                        
                        if task.createdBy.isAgent {
                            Text("Agent")
                                .font(.caption2)
                                .fontWeight(.medium)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.purple.opacity(0.2))
                                .foregroundStyle(.purple)
                                .clipShape(Capsule())
                        }
                    }
                    
                    Text(task.humanReadableSchedule)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button {
                    if task.enabled {
                        schedulingService.pauseTask(id: task.id)
                    } else {
                        try? schedulingService.resumeTask(id: task.id)
                    }
                } label: {
                    Text(task.enabled ? "Pause" : "Resume")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Next Run")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .textCase(.uppercase)
                    if task.enabled, let next = task.nextRunAt {
                        Text(next.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption)
                    } else {
                        Text("Paused")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Last Run")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .textCase(.uppercase)
                    if let last = task.lastRunAt {
                        Text(last.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption)
                    } else {
                        Text("Never")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            
            Divider()
            
            HStack {
                Button {
                    Task {
                        await agentCoordinator.openBackgroundSession(id: task.sessionId)
                        if let appDelegate = NSApp.delegate as? AppDelegate {
                            appDelegate.navigateToAgent()
                        }
                    }
                } label: {
                    Label("View Session", systemImage: "message")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tint)
                .font(.callout)
                
                Spacer()
                
                Button(role: .destructive) {
                    schedulingService.deleteTask(id: task.id)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red.opacity(0.8))
            }
        }
        .padding(Theme.Spacing.md)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .opacity(task.enabled ? 1.0 : 0.6)
        .shadow(color: .black.opacity(0.04), radius: 6, x: 0, y: 3)
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.quaternary, lineWidth: 0.5)
        }
    }
}

// MARK: - Run History Row

struct RunHistoryRowView: View {
    let run: TaskRun
    let taskName: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Time
            Text(run.startedAt.formatted(date: .omitted, time: .shortened))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .leading)
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(taskName)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    Spacer()
                    
                    resultBadge
                }
                
                Text(run.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
    
    @ViewBuilder
    private var resultBadge: some View {
        switch run.result {
        case .success:
            Text("Done")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.green.opacity(0.2))
                .foregroundStyle(.green)
                .clipShape(Capsule())
        case .noActionNeeded:
            Text("OK")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.secondary.opacity(0.2))
                .foregroundStyle(.secondary)
                .clipShape(Capsule())
        case .error:
            Text("Error")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.red.opacity(0.2))
                .foregroundStyle(.red)
                .clipShape(Capsule())
        case .interrupted:
            Text("Interrupted")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.orange.opacity(0.2))
                .foregroundStyle(.orange)
                .clipShape(Capsule())
        case .missed:
            Text("Missed")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.gray.opacity(0.2))
                .foregroundStyle(.gray)
                .clipShape(Capsule())
        }
    }
}
