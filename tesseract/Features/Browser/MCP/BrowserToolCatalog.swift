import Foundation

// MARK: - BrowserToolSpec

/// Static description of one browser tool — the single source of truth for its
/// name, agent-facing description, and parameter schema. The MCP server turns
/// these into `tools/list` entries; the executor keys off `name`.
nonisolated struct BrowserToolSpec: Sendable {
    let name: String
    let description: String
    let inputSchema: JSONSchema
}

// MARK: - BrowserToolCatalog

/// The lean ~12-tool surface (ADR-0026). Deliberately small: benchmark evidence
/// (AgentOccam) and Anthropic's tool-writing guidance both show tight,
/// pretraining-aligned verbs beat sprawling tool sets. Reads default to the
/// token-cheap **Page Read**; the **Page Map** is requested only to interact.
nonisolated enum BrowserToolCatalog {

    static let all: [BrowserToolSpec] = [
        navigate, back, readPage, pageMap, click, type, find, tabs, evaluate,
        screenshot, search, fetch,
    ]

    static func spec(named name: String) -> BrowserToolSpec? {
        all.first { $0.name == name }
    }

    // MARK: Specs

    static let navigate = BrowserToolSpec(
        name: "navigate",
        description:
            "Load a URL in your authenticated browser tab. Returns the final URL and page "
            + "title only — call read_page to read the content or page_map to interact.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "url": PropertySchema(
                    type: "string",
                    description: "URL to open (a bare domain is upgraded to https://).")
            ],
            required: ["url"]))

    static let back = BrowserToolSpec(
        name: "back",
        description: "Go back one entry in the current tab's history.",
        inputSchema: JSONSchema(type: "object", properties: [:], required: []))

    static let readPage = BrowserToolSpec(
        name: "read_page",
        description:
            "Read the current page as clean, distilled Markdown (the token-cheap default for "
            + "consuming a page). Long pages are paginated: if the result ends with a cursor "
            + "hint, call again with that cursor for the next chunk.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "max_chars": PropertySchema(
                    type: "integer",
                    description: "Max characters to return this call (default 20000)."),
                "cursor": PropertySchema(
                    type: "integer",
                    description: "Character offset to resume from (from a prior page's hint)."),
            ],
            required: []))

    static let pageMap = BrowserToolSpec(
        name: "page_map",
        description:
            "List the page's interactive elements (links, buttons, fields) with stable [ref] "
            + "numbers and structural headings. Request this only when you need to click or "
            + "type — the [ref] values feed click and type.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "max_elements": PropertySchema(
                    type: "integer",
                    description: "Max interactive elements to list (default 200).")
            ],
            required: []))

    static let click = BrowserToolSpec(
        name: "click",
        description:
            "Click the element with the given [ref] from the latest page_map. Returns where "
            + "the page ended up after any resulting navigation.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "ref": PropertySchema(
                    type: "integer", description: "Element ref from page_map.")
            ],
            required: ["ref"]))

    static let type = BrowserToolSpec(
        name: "type",
        description:
            "Type text into the field with the given [ref] from the latest page_map. Set "
            + "submit=true to press Enter / submit the form afterward.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "ref": PropertySchema(
                    type: "integer", description: "Field ref from page_map."),
                "text": PropertySchema(type: "string", description: "Text to enter."),
                "submit": PropertySchema(
                    type: "boolean",
                    description: "Press Enter / submit after typing (default false)."),
            ],
            required: ["ref", "text"]))

    static let find = BrowserToolSpec(
        name: "find",
        description:
            "Find occurrences of a string in the current page's visible text and return short "
            + "surrounding snippets — a cheap way to locate content without re-reading the page.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "query": PropertySchema(type: "string", description: "Text to search for."),
                "max_results": PropertySchema(
                    type: "integer", description: "Max snippets to return (default 10)."),
            ],
            required: ["query"]))

    static let tabs = BrowserToolSpec(
        name: "tabs",
        description:
            "Manage this session's tabs. action=list shows them; open starts a tab (optionally "
            + "at a url); select/close act on a tab_id from the listing.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "action": PropertySchema(
                    type: "string", description: "list | open | select | close",
                    enumValues: ["list", "open", "select", "close"]),
                "url": PropertySchema(
                    type: "string", description: "URL for action=open (optional)."),
                "tab_id": PropertySchema(
                    type: "string", description: "Tab id for action=select/close."),
            ],
            required: ["action"]))

    static let evaluate = BrowserToolSpec(
        name: "evaluate",
        description:
            "Run JavaScript in the current page (isolated from the page's own scripts) and "
            + "return the result. Use `return <expr>` to extract exactly what you need in a few "
            + "tokens instead of re-reading the page.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "script": PropertySchema(
                    type: "string",
                    description: "JavaScript function body; use `return` to yield a value.")
            ],
            required: ["script"]))

    static let screenshot = BrowserToolSpec(
        name: "screenshot",
        description:
            "Capture a PNG screenshot of the current page — a vision fallback for content that "
            + "doesn't read well as text (charts, canvas, visual layout).",
        inputSchema: JSONSchema(type: "object", properties: [:], required: []))

    static let search = BrowserToolSpec(
        name: "search",
        description:
            "Search the web anonymously (cookieless, outside your logged-in profile) and return "
            + "titles, URLs, and snippets. Snippets are only hints for picking pages — open the "
            + "result URLs with fetch (or navigate + read_page) to read the real content.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "query": PropertySchema(type: "string", description: "Search query."),
                "max_results": PropertySchema(
                    type: "integer", description: "1–10 results (default 5)."),
            ],
            required: ["query"]))

    static let fetch = BrowserToolSpec(
        name: "fetch",
        description:
            "Fetch a URL anonymously (cookieless, outside your profile) and return it as "
            + "distilled Markdown. Use navigate + read_page instead when a page needs your "
            + "login.",
        inputSchema: JSONSchema(
            type: "object",
            properties: [
                "url": PropertySchema(type: "string", description: "URL to fetch."),
                "max_chars": PropertySchema(
                    type: "integer", description: "Max characters to return (default 20000)."),
            ],
            required: ["url"]))
}
