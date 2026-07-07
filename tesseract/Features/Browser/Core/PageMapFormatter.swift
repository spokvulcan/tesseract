import Foundation

// MARK: - PageMapFormatter

/// Pure transform: decoded **Page Map** elements → the compact outline the
/// agent reads. Kept free of WebKit and I/O so it is unit-testable in isolation
/// (the second, lower test seam for the browser feature).
///
/// Output shape, one element per line:
/// ```
/// # Results                     (heading, level 1)
/// [1] link "Home"
/// [3] textbox "Search" placeholder="Search…"
/// [4] button "Go" (disabled)
/// ```
/// The leading `[n]` is the interaction ref for `click`/`type`. Headings render
/// as Markdown-style `#` scaffolding with no ref.
nonisolated enum PageMapFormatter {

    /// Render the outline.
    /// - Parameters:
    ///   - elements: decoded map, in document order.
    ///   - maxElements: display cap on **ref-bearing** rows; headings never
    ///     count against it (they are cheap scaffolding). When the cap trims
    ///     rows, a steering line names how many were dropped.
    static func format(_ elements: [PageMapElement], maxElements: Int = 200) -> String {
        guard !elements.isEmpty else {
            return "No interactive elements found on this page. "
                + "It may still have readable content — use read_page."
        }

        var lines: [String] = []
        var interactiveShown = 0
        var interactiveDropped = 0

        for element in elements {
            if element.role == "heading" {
                let hashes = String(repeating: "#", count: min(max(element.level ?? 1, 1), 6))
                lines.append("\(hashes) \(element.name)")
                continue
            }

            guard interactiveShown < maxElements else {
                interactiveDropped += 1
                continue
            }
            lines.append(renderInteractive(element))
            interactiveShown += 1
        }

        if interactiveDropped > 0 {
            lines.append(
                "… \(interactiveDropped) more interactive element(s) not shown "
                    + "(raise max_elements or use find to locate a specific control).")
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Private

    private static func renderInteractive(_ element: PageMapElement) -> String {
        var line = "[\(element.ref ?? 0)] \(element.role)"

        let name = element.name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty {
            line += " \"\(name)\""
        }

        if let value = element.value, !value.isEmpty {
            line += " value=\"\(value)\""
        }
        if let placeholder = element.placeholder, !placeholder.isEmpty,
            placeholder != element.name
        {
            line += " placeholder=\"\(placeholder)\""
        }

        var flags: [String] = []
        if element.checked == true { flags.append("checked") }
        if element.disabled == true { flags.append("disabled") }
        if !flags.isEmpty {
            line += " (\(flags.joined(separator: ", ")))"
        }

        return line
    }
}
