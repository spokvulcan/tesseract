//
//  MarkdownGallery.swift
//  tesseract
//
//  The Markdown Gallery: a live instrument for the chat's markdown
//  rendering. An editable source pane — pre-filled with a canonical document
//  exercising every construct the renderer supports — feeds the exact chat
//  render stack (`ChatMarkdownView`), so what the gallery shows is what the
//  transcript does; it cannot drift. The appearance switch renders Light,
//  Dark, or both side by side for palette work. Always compiled: the gallery
//  is the living style reference, and the dev loop runs Release builds.
//

import SwiftUI

/// Which appearance(s) the preview renders.
private enum GalleryAppearance: String, CaseIterable, Identifiable {
    case light = "Light"
    case dark = "Dark"
    case both = "Both"

    var id: String { rawValue }
}

struct MarkdownGalleryView: View {
    @State private var source = MarkdownGalleryDocument.canonical
    @State private var appearance: GalleryAppearance = .both

    var body: some View {
        HSplitView {
            editor
            preview
        }
        .toolbar {
            ToolbarItem {
                Picker("Appearance", selection: $appearance) {
                    ForEach(GalleryAppearance.allCases) { choice in
                        Text(choice.rawValue).tag(choice)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
        .navigationTitle("Markdown Gallery")
        // Edits are ephemeral by contract: the scene's @State survives a
        // close/reopen (measured — macOS keeps singleton Window scene state
        // alive), so each open restores the canonical document explicitly.
        .onAppear { source = MarkdownGalleryDocument.canonical }
    }

    private var editor: some View {
        TextEditor(text: $source)
            .font(.system(size: 12, design: .monospaced))
            .frame(minWidth: 240, idealWidth: 320, maxWidth: 480)
    }

    @ViewBuilder
    private var preview: some View {
        switch appearance {
        case .light:
            previewPane(.light)
        case .dark:
            previewPane(.dark)
        case .both:
            HStack(spacing: 0) {
                previewPane(.light)
                Divider()
                previewPane(.dark)
            }
        }
    }

    /// One rendered copy of the source, forced to a color scheme. The chat's
    /// readable-column metrics apply so line lengths match the transcript.
    private func previewPane(_ scheme: ColorScheme) -> some View {
        ScrollView {
            ChatMarkdownView(text: source)
                .frame(maxWidth: ChatLayout.columnMaxWidth, alignment: .leading)
                .padding(20)
        }
        .frame(minWidth: 300, maxWidth: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .environment(\.colorScheme, scheme)
    }
}

/// The canonical document: one construct per section, ordered roughly as the
/// chat meets them. Edits in the gallery are ephemeral — the document resets
/// on every open. No image syntax: the renderer would try to load it, and the
/// assistant never emits images into prose.
enum MarkdownGalleryDocument {
    static let canonical = #"""
        # Heading 1 — Document Title

        ## Heading 2 — Section

        ### Heading 3 — Subsection

        #### Heading 4

        ##### Heading 5

        ###### Heading 6

        A paragraph with **bold**, *italic*, ***bold italic***, ~~strikethrough~~,
        a [link to the repo](https://github.com/spokvulcan/tesseract), and an
        autolink: <https://example.com>.

        Inline code, short and long: `x`, `scripts/dev.sh dev-release`, and a
        wrapping span `PrefixCacheManager.resolveSnapshot(for:tokens:allowPartial:)`
        inside running prose. Punctuation hugs the chip: `let x = 1`, `y`?

        ---

        - Unordered list, level one
          - Level two, with `inline code`
            - Level three, with a [link](https://example.com)
        - Back to level one with **bold**

        1. Ordered list — **Bold Term**: explanation follows in plain prose
        2. Second ordinal with *emphasis*
           1. Nested ordinal
        3. Third

        Task-list syntax (the renderer has no checkbox support — this shows the
        literal fallback the chat would produce):

        - [ ] Task list: an unchecked item
        - [x] Task list: a checked item

        > A single-line blockquote with `inline code` and **bold**.

        > A multi-paragraph blockquote. First paragraph.
        >
        > Second paragraph, with a nested quote below.
        > > The nested quote.

        ```swift
        /// A Swift fence: keywords, types, strings, numbers.
        func resolve(_ path: TokenPath) -> Snapshot? {
            let budget = max(0, capacity - 4_096)
            return cache.first { $0.matches(path) && $0.cost < budget }
        }
        ```

        ```python
        # A Python fence.
        def hit_rate(hits: int, misses: int) -> float:
            return hits / max(1, hits + misses)
        ```

        ```json
        { "model": "qwen3.6-35b", "temperature": 0.7, "stream": true }
        ```

        ```
        A plain fence with no language — falls back to the base code color.
        ```

        | Column | Aligned Left | Centered | Right |
        | ------ | :----------- | :------: | ----: |
        | Row 1  | `code`       | **bold** |  1234 |
        | Row 2  | plain        | *italic* |    56 |

        A final paragraph after a hard break:\
        this line follows the backslash break, and \*escaped asterisks\* stay
        literal.
        """#
}
