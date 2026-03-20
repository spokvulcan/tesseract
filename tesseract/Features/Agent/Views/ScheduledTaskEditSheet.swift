import SwiftUI

struct ScheduledTaskEditSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(SchedulingService.self) private var schedulingService
    
    let taskToEdit: ScheduledTask?
    
    @State private var name: String = ""
    @State private var description: String = ""
    @State private var cronExpression: String = ""
    @State private var prompt: String = ""
    @State private var notifyUser: Bool = true
    @State private var speakResult: Bool = false
    
    @State private var hasMaxRuns: Bool = false
    @State private var maxRuns: Int = 1
    
    @State private var errorMessage: String?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text(taskToEdit == nil ? "New Task" : "Edit Task")
                    .font(.title3)
                    .fontWeight(.semibold)
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, Theme.Spacing.xl)
            .padding(.vertical, Theme.Spacing.lg)
            
            Divider()
            
            ScrollView {
                VStack(spacing: Theme.Spacing.xl) {
                    
                    // Basic Info
                    VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                        Text("Basic Info")
                            .font(.headline)
                        
                        VStack(spacing: 0) {
                            TextField("Task Name", text: $name)
                                .textFieldStyle(.plain)
                                .font(.body)
                                .padding(12)
                            
                            Divider().padding(.leading, 12)
                            
                            TextField("Description (Optional)", text: $description)
                                .textFieldStyle(.plain)
                                .font(.body)
                                .padding(12)
                        }
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(.quaternary, lineWidth: 0.5)
                        }
                    }
                    
                    // Schedule
                    VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                        Text("Schedule")
                            .font(.headline)
                        
                        VStack(alignment: .leading, spacing: 0) {
                            HStack {
                                Image(systemName: "clock")
                                    .foregroundStyle(.secondary)
                                TextField("Cron Expression (e.g. 0 9 * * *)", text: $cronExpression)
                                    .textFieldStyle(.plain)
                                    .font(.body.monospaced())
                            }
                            .padding(12)
                        }
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(
                                    cronError != nil
                                        ? AnyShapeStyle(Color.red.opacity(0.5))
                                        : AnyShapeStyle(.quaternary),
                                    lineWidth: cronError != nil ? 1 : 0.5
                                )
                        }
                        
                        if let error = cronError {
                            Text(error)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .padding(.horizontal, 4)
                        } else if let next = nextRun {
                            Text("Next run: \(next.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 4)
                        }
                        
                        VStack(alignment: .leading, spacing: 0) {
                            Toggle(isOn: $hasMaxRuns.animation()) {
                                Text("Limit Number of Runs")
                            }
                            .toggleStyle(.switch)
                            .padding(12)
                            
                            if hasMaxRuns {
                                Divider().padding(.leading, 12)
                                
                                HStack {
                                    Text("Maximum runs:")
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Stepper(value: Binding(
                                        get: { Double(maxRuns) },
                                        set: { maxRuns = Int($0) }
                                    ), in: 1...1000) {
                                        Text("\(maxRuns)")
                                            .font(.body.monospacedDigit())
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(Color.secondary.opacity(0.2))
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                    }
                                }
                                .padding(12)
                            }
                        }
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(.quaternary, lineWidth: 0.5)
                        }
                    }
                    
                    // Instruction
                    VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                        Text("Agent Instruction")
                            .font(.headline)
                        
                        TextEditor(text: $prompt)
                            .font(.body)
                            .frame(minHeight: 120)
                            .scrollContentBackground(.hidden)
                            .padding(8)
                            .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .strokeBorder(.quaternary, lineWidth: 0.5)
                            }
                    }
                    
                    // Actions
                    VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                        Text("Actions")
                            .font(.headline)
                        
                        VStack(spacing: 0) {
                            Toggle(isOn: $notifyUser) {
                                HStack {
                                    Label("Send Notification", systemImage: "app.badge")
                                    Spacer()
                                }
                            }
                            .toggleStyle(.switch)
                            .padding(12)
                            
                            Divider().padding(.leading, 12)
                            
                            Toggle(isOn: $speakResult) {
                                HStack {
                                    Label("Speak Result via TTS", systemImage: "speaker.wave.2")
                                    Spacer()
                                }
                            }
                            .toggleStyle(.switch)
                            .padding(12)
                        }
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(.quaternary, lineWidth: 0.5)
                        }
                    }
                }
                .padding(Theme.Spacing.xl)
            }
            
            Divider()
            
            // Footer
            HStack {
                if let errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
                
                Spacer()
                
                Button("Cancel") {
                    dismiss()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.quinary, in: Capsule())
                
                Button("Save Task") {
                    save()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isValid ? Color.accentColor : Color.secondary.opacity(0.2), in: Capsule())
                .foregroundStyle(isValid ? .white : .secondary)
                .disabled(!isValid)
            }
            .padding(Theme.Spacing.xl)
        }
        .frame(width: 500, height: 700)
        .onAppear {
            if let task = taskToEdit {
                name = task.name
                description = task.description
                cronExpression = task.cronExpression
                prompt = task.prompt
                notifyUser = task.notifyUser
                speakResult = task.speakResult
                if let mr = task.maxRuns {
                    hasMaxRuns = true
                    maxRuns = mr
                }
            } else {
                cronExpression = "0 9 * * *"
            }
        }
    }
    
    private var cronError: String? {
        if cronExpression.isEmpty { return nil }
        do {
            _ = try CronExpression(parsing: cronExpression)
            return nil
        } catch {
            return error.localizedDescription
        }
    }
    
    private var nextRun: Date? {
        guard let expr = try? CronExpression(parsing: cronExpression) else { return nil }
        return expr.nextOccurrence(after: Date())
    }
    
    private var isValid: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty &&
        !prompt.trimmingCharacters(in: .whitespaces).isEmpty &&
        cronError == nil
    }
    
    private func save() {
        do {
            let finalMaxRuns = hasMaxRuns ? maxRuns : nil
            if let taskId = taskToEdit?.id {
                guard var task = schedulingService.loadTask(id: taskId) else {
                    errorMessage = "Task no longer exists."
                    return
                }

                let didChangeCron = task.cronExpression != cronExpression
                task.name = name
                task.description = description
                task.cronExpression = cronExpression
                task.prompt = prompt
                task.notifyUser = notifyUser
                task.speakResult = speakResult
                task.maxRuns = finalMaxRuns
                if didChangeCron {
                    task.nextRunAt = task.computeNextRunAt()
                }
                try schedulingService.updateTask(task)
            } else {
                let newTask = try ScheduledTask.create(
                    name: name,
                    cronExpression: cronExpression,
                    prompt: prompt,
                    description: description,
                    enabled: true,
                    createdBy: .user,
                    maxRuns: finalMaxRuns,
                    notifyUser: notifyUser,
                    speakResult: speakResult
                )
                try schedulingService.createTask(newTask)
            }
            dismiss()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
