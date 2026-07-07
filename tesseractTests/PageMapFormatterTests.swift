import Foundation
import Testing

@testable import Tesseract_Agent

/// Lower browser test seam: the **Page Map** outline is a pure transform over
/// decoded elements, so it is exercised without WebKit.
struct PageMapFormatterTests {

    private func element(
        ref: Int? = nil,
        role: String,
        name: String,
        value: String? = nil,
        placeholder: String? = nil,
        level: Int? = nil,
        disabled: Bool? = nil,
        checked: Bool? = nil
    ) -> PageMapElement {
        PageMapElement(
            ref: ref, role: role, name: name, value: value, placeholder: placeholder,
            level: level, disabled: disabled, checked: checked)
    }

    @Test
    func emptyMapExplainsItself() {
        let out = PageMapFormatter.format([])
        #expect(out.contains("No interactive elements"))
        #expect(out.contains("read_page"))
    }

    @Test
    func interactiveRowCarriesRefRoleAndName() {
        let out = PageMapFormatter.format([element(ref: 1, role: "link", name: "Home")])
        #expect(out == "[1] link \"Home\"")
    }

    @Test
    func headingsRenderAsMarkdownWithoutARef() {
        let out = PageMapFormatter.format([
            element(role: "heading", name: "Results", level: 2),
            element(ref: 1, role: "link", name: "First"),
        ])
        let lines = out.split(separator: "\n").map(String.init)
        #expect(lines[0] == "## Results")
        #expect(lines[1] == "[1] link \"First\"")
        // Headings never carry an interaction ref.
        #expect(!lines[0].contains("["))
    }

    @Test
    func placeholderAndFlagsAreSurfaced() {
        let out = PageMapFormatter.format([
            element(
                ref: 3, role: "textbox", name: "Search", placeholder: "Search…"),
            element(ref: 4, role: "button", name: "Go", disabled: true),
            element(ref: 5, role: "checkbox", name: "Remember me", checked: true),
        ])
        #expect(out.contains("[3] textbox \"Search\" placeholder=\"Search…\""))
        #expect(out.contains("[4] button \"Go\" (disabled)"))
        #expect(out.contains("[5] checkbox \"Remember me\" (checked)"))
    }

    @Test
    func placeholderEqualToNameIsNotDuplicated() {
        let out = PageMapFormatter.format([
            element(ref: 1, role: "textbox", name: "Email", placeholder: "Email")
        ])
        #expect(out == "[1] textbox \"Email\"")
    }

    @Test
    func maxElementsCapsInteractiveRowsButNotHeadings() {
        var elements: [PageMapElement] = [element(role: "heading", name: "Top", level: 1)]
        for i in 1...10 {
            elements.append(element(ref: i, role: "link", name: "L\(i)"))
        }
        let out = PageMapFormatter.format(elements, maxElements: 4)
        let lines = out.split(separator: "\n").map(String.init)
        // 1 heading + 4 interactive + 1 steering line.
        #expect(lines.filter { $0.hasPrefix("[") }.count == 4)
        #expect(lines.first == "# Top")
        #expect(out.contains("6 more interactive element"))
    }
}
