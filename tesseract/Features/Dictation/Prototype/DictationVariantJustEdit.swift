//
//  DictationVariantJustEdit.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  D · Just Edit. No new gesture: fix the words where they landed, the way
//  you fix any text. For a minute after a take the app watches that field
//  through Accessibility; when your edit settles it learns the change and
//  says so, with Undo. Where it can't see the text (terminals, many
//  Electron apps) it says that too, and ⌃⌥Space opens the fix bar instead.
//  The page is a journal of what it learned and where it can watch.
//

import SwiftUI

@MainActor
enum JustEditFlow {
    static func showWatching(_ lab: DictationLab, take: LabTake) {
        guard let watcher = lab.watcher else { return }
        lab.panels.showCard(
            WatchCard(lab: lab, watcher: watcher, take: take), size: CGSize(width: 480, height: 64),
            duration: .seconds(5))
    }
}

struct WatchCard: View {
    let lab: DictationLab
    let watcher: LabEditWatcher
    let take: LabTake

    var body: some View {
        HStack(spacing: 10) {
            switch watcher.state {
            case .watching(let app, let until):
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    let left = max(0, until.timeIntervalSince(context.date))
                    HStack(spacing: 10) {
                        CountdownRing(fraction: left / 60)
                        Text("Edit it in \(app) and I'll learn the fix")
                            .font(.system(size: 12, weight: .semibold))
                        Spacer(minLength: 4)
                        Text("\(Int(left)) s")
                            .font(.system(size: 12))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            case .blind(let app):
                Image(systemName: "eye.slash")
                    .foregroundStyle(.secondary)
                Text("Can't see the text in \(app)")
                    .font(.system(size: 12, weight: .semibold))
                Spacer(minLength: 4)
                Text("\(LabStyle.fixShortcut) to fix")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            case .idle:
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.green)
                Text("Inserted in \(take.appName)")
                    .font(.system(size: 12, weight: .semibold))
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

struct CountdownRing: View {
    let fraction: Double
    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 2)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 14, height: 14)
        .animation(.linear(duration: 1), value: fraction)
    }
}

// MARK: - Page

struct JustEditPage: View {
    let lab: DictationLab

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: LabStyle.rhythm) {
                if let watcher = lab.watcher { liveRow(watcher) }
                Text("What I learned from your edits")
                    .font(.system(size: LabStyle.body, weight: .semibold))
                if lab.lessons.isEmpty {
                    Text(
                        "Dictate into an app, then fix a word the way you'd fix any text. When you're done, I learn the change and tell you. Where I can't see the text, press \(LabStyle.fixShortcut)."
                    )
                    .font(.system(size: LabStyle.body))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(lab.lessons) { lesson in
                    LessonRow(lab: lab, lesson: lesson)
                    Divider().opacity(0.5)
                }
                if let watcher = lab.watcher, !watcher.apps.isEmpty {
                    Text("Where I can watch")
                        .font(.system(size: LabStyle.body, weight: .semibold))
                        .padding(.top, 8)
                    ForEach(watcher.apps.sorted(by: { $0.key < $1.key }), id: \.key) {
                        app, canSee in
                        HStack(spacing: 8) {
                            Image(systemName: canSee ? "eye" : "eye.slash")
                                .foregroundStyle(canSee ? Color.accentColor : .secondary)
                                .frame(width: 18)
                            Text(app).font(.system(size: LabStyle.body))
                            Text(
                                canSee ? "your edits teach me" : "use \(LabStyle.fixShortcut) here"
                            )
                            .font(.system(size: LabStyle.body))
                            .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .frame(maxWidth: LabStyle.column, alignment: .leading)
            .padding(24)
            .frame(maxWidth: .infinity)
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                LabStatusLine(extra: "edit in place, or \(LabStyle.fixShortcut)")
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }

    @ViewBuilder
    private func liveRow(_ watcher: LabEditWatcher) -> some View {
        if case .watching(let app, let until) = watcher.state {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let left = max(0, until.timeIntervalSince(context.date))
                HStack(spacing: 10) {
                    CountdownRing(fraction: left / 60)
                    Text("Watching \(app) for your edit")
                        .font(.system(size: LabStyle.body, weight: .medium))
                    Text("\(Int(left)) s left")
                        .font(.system(size: LabStyle.body))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            .padding(.bottom, 8)
        }
    }
}

struct LessonRow: View {
    let lab: DictationLab
    let lesson: LabLesson

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(lesson.date, format: .dateTime.hour().minute())
                .font(.system(size: LabStyle.body))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
                .frame(width: 64, alignment: .trailing)
            VStack(alignment: .leading, spacing: 4) {
                if lesson.learned, !lesson.undone {
                    HeardMeant(heard: lesson.hunk.before, meant: lesson.hunk.after)
                } else {
                    HStack(spacing: 6) {
                        Text(lesson.hunk.before)
                            .strikethrough()
                            .foregroundStyle(.secondary)
                        Image(systemName: "arrow.right").font(.system(size: 10)).foregroundStyle(
                            .tertiary)
                        Text(lesson.hunk.after).foregroundStyle(.secondary)
                    }
                    .font(.system(size: LabStyle.body))
                }
                Text(caption)
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
        .contextMenu {
            if lesson.learned, !lesson.undone {
                Button("Undo, don't learn this") { lab.undo([lesson.id]) }
            }
        }
    }

    private var caption: String {
        let source = lesson.source.rawValue.lowercased()
        if lesson.undone {
            return "\(source) in \(lesson.appName) · undone, won't be learned again"
        }
        if !lesson.learned {
            return "\(source) in \(lesson.appName) · looked like a rewrite, not learned"
        }
        return "\(source) in \(lesson.appName)"
    }
}
