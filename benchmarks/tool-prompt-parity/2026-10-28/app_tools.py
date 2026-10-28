"""The app's file, profile and agenda tools, as Python dicts in the order the
Swift sources write them (AgentToolDefinition.toolSpec's envelope order, and
each JSONSchema's properties in their literal order)."""


def prop(type_, description, enum=None, items=None):
    d = {"type": type_, "description": description}
    if enum is not None:
        d["enum"] = enum
    if items is not None:
        d["items"] = items
    return d


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


APP_TOOLS = [
    tool(
        "read",
        "Read the contents of a file. Returns numbered lines (cat -n style). For images, returns a note and the image data directly.",
        {
            "path": prop("string", "File path relative to working directory"),
            "offset": prop("integer", "Line number to start from (1-indexed)"),
            "limit": prop("integer", "Maximum number of lines to return"),
        },
        ["path"],
    ),
    tool(
        "write",
        "Write content to a file. Creates parent directories if needed. By default, appends to existing files. Set overwrite to true to replace the entire file (requires reading the file first).",
        {
            "path": prop("string", "File path to write"),
            "content": prop("string", "Content to write"),
            "overwrite": prop("boolean", "If true, replace the entire file instead of appending. Requires reading the file first. Default: false"),
        },
        ["path", "content"],
    ),
    tool(
        "edit",
        "Edit a file by replacing an exact text match. old_text must match exactly once in the file. Add surrounding context lines to old_text to disambiguate multiple matches.",
        {
            "path": prop("string", "File path to edit"),
            "old_text": prop("string", "Exact text to find and replace"),
            "new_text": prop("string", "Replacement text"),
        },
        ["path", "old_text", "new_text"],
    ),
    tool(
        "ls",
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).",
        {
            "path": prop("string", "Directory to list (default: current directory)"),
            "limit": prop("integer", "Maximum number of entries to return (default: 500)"),
        },
        [],
    ),
    tool(
        "remember",
        "Save one lasting fact about the owner to their Profile when they ask you to remember something (\"remember that I…\"). Write it as a short third-person fact: \"Goes to the gym on Mondays and Thursdays.\" One fact per call. Never save what they didn't ask you to remember.",
        {
            "fact": prop("string", "The fact, third person, one sentence."),
            "area": prop("string", "Optional Area it belongs to."),
        },
        ["fact"],
    ),
    tool(
        "forget",
        "Remove a fact from the owner's Profile when they say it's wrong or ask you to forget it. Give the fact's id (from recall) or its words.",
        {"fact": prop("string", "The fact's id, or its words.")},
        ["fact"],
    ),
    tool(
        "recall",
        "Search what the owner told you: their Profile and past conversations. Use it when they ask about something from before (\"what did I say about X last week?\") or when you need a detail you don't have. Answers with dated snippets.",
        {
            "query": prop("string", "What to look for."),
            "days": prop("integer", "Only the last N days."),
        },
        ["query"],
    ),
    tool(
        "agenda",
        "Read the owner's calendar events and reminders (Apple Calendar and Reminders) for a range of days, with the ids the other agenda tools take. With today in range it also lists overdue reminders, the Inbox (reminders with no date) and what is done today.",
        {
            "from": prop("string", "First day: YYYY-MM-DD, \"today\" or \"tomorrow\". Default: today."),
            "days": prop("integer", "How many days, 1–14. Default: 1."),
        },
        [],
    ),
    tool(
        "add_reminder",
        "Add a reminder to Apple Reminders — for anything the owner needs to do or asks to be reminded of. Give it a due time only when the owner gave one or one follows from their words; a reminder with a time alerts on their Mac, phone and watch. Without a due time it lands in the Inbox or in its Area undated.",
        {
            "title": prop("string", "What to do, short: \"Call the dentist\"."),
            "area": prop("string", "The owner's Area (a Reminders list), e.g. Work or Health. Omit for the Inbox."),
            "due": prop("string", "Local time YYYY-MM-DDTHH:MM, or YYYY-MM-DD for a whole day."),
            "after_event": prop("string", "Instead of due: the calendar event it follows (\"the 1:1\", or an event id); due becomes that event's end."),
            "notes": prop("string", "Optional details."),
        },
        ["title"],
    ),
    tool(
        "update_reminder",
        "Change one reminder by id (from the agenda tool): complete it or reopen it, rename it, move it to another time or Area, or clear its date.",
        {
            "id": prop("string", "The reminder's id."),
            "completed": prop("boolean", "true completes it, false reopens it."),
            "title": prop("string", "A new title."),
            "area": prop("string", "Move it to this Area."),
            "due": prop("string", "New local time YYYY-MM-DDTHH:MM, a whole day YYYY-MM-DD, or \"none\" to clear the date."),
            "after_event": prop("string", "Instead of due: re-time it to the end of this calendar event."),
            "notes": prop("string", "Replace its notes."),
        },
        ["id"],
    ),
    tool(
        "add_event",
        "Add an event to the owner's calendar (Apple Calendar), in the default calendar unless one is named. Use it for time the owner wants to block or a meeting they describe.",
        {
            "title": prop("string", "The event's title."),
            "start": prop("string", "Local start YYYY-MM-DDTHH:MM, or YYYY-MM-DD for an all-day event."),
            "end": prop("string", "Local end YYYY-MM-DDTHH:MM."),
            "duration_minutes": prop("integer", "Instead of end. Default: 30."),
            "calendar": prop("string", "A calendar's name."),
            "location": prop("string", "Where."),
            "notes": prop("string", "Details."),
        },
        ["title", "start"],
    ),
    tool(
        "move_event",
        "Move one calendar event (by id, from the agenda tool) to a new time. It keeps its length unless an end is given. Only this occurrence of a repeating event moves.",
        {
            "id": prop("string", "The event's id."),
            "start": prop("string", "New local start YYYY-MM-DDTHH:MM."),
            "end": prop("string", "New local end YYYY-MM-DDTHH:MM."),
        },
        ["id", "start"],
    ),
    tool(
        "delete_event",
        "Delete one calendar event (by id, from the agenda tool) when the owner asks: a block, a slot or an appointment they keep for themselves. Only this occurrence of a repeating event goes. A meeting with other people invited can't be deleted here; tell the owner to decline or cancel it in Calendar. The change can be undone.",
        {"id": prop("string", "The event's id.")},
        ["id"],
    ),
]

# Keys in the order transformers' get_json_schema and the app's own builders
# write them; anything else follows, sorted.
CONVENTIONAL_ORDER = [
    "type", "function", "name", "description", "parameters", "properties", "required",
    "enum", "items",
]


def sort_keys(value):
    if isinstance(value, dict):
        return {k: sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [sort_keys(v) for v in value]
    return value


def rank_keys(value, parent=None):
    """Conventional order for schema keywords; property names (the keys of a
    `properties` object) sorted, since that order is what the app loses."""
    if isinstance(value, dict):
        if parent == "properties":
            keys = sorted(value)
        else:
            known = [k for k in CONVENTIONAL_ORDER if k in value]
            keys = known + sorted(k for k in value if k not in CONVENTIONAL_ORDER)
        return {k: rank_keys(value[k], k) for k in keys}
    if isinstance(value, list):
        return [rank_keys(v) for v in value]
    return value
