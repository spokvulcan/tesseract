#!/usr/bin/env python3
"""Fails when app code is used only by tests.

A type or function in tesseract/ that no app code references but a test does
is dead code its tests keep alive: ServerRunState outlived its last caller by
three months that way. Each one is either deleted with its tests or listed in
scripts/test-only-code.allow with the reason it stays (a test seam, a slice in
progress, a caller in another target).

A source scan, not a compiler: it counts identifier references outside
comments, so a name shared with live code elsewhere hides a dead one, never the
reverse. Runs in about a second, with no build.

Usage: scripts/check-test-only-code.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECLARING = ["tesseract"]
APP = ["tesseract", "tesseract-ios", "tesseract-share"]
TESTS = ["tesseractTests", "tesseractUITests"]
ALLOW = ROOT / "scripts" / "test-only-code.allow"

DECLARATION = re.compile(
    r"^(?P<attrs>(?:\s*@\w+(?:\([^)]*\))?)*)\s*(?P<mods>(?:\w+\s+)*?)"
    r"(?P<kind>class|struct|enum|actor|protocol|typealias|func)\s+(?P<name>[A-Za-z_]\w*)"
)
EXTENSION = re.compile(r"\bextension\s+([A-Za-z_]\w*)")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
# Called by the runtime, a framework or the compiler, not by name.
RUNTIME_NAMES = {"main", "body", "callAsFunction", "hash", "encode", "makeBody"}
RUNTIME_ATTRS = ("@objc", "@IBAction", "@main", "@NSApplicationDelegateAdaptor")


def swift_files(dirs):
    for d in dirs:
        yield from sorted((ROOT / d).rglob("*.swift"))


def without_comments(source):
    """Blanks // and /* */ comments, keeping line numbers and string literals."""
    out, i, n = [], 0, len(source)
    in_string = False
    while i < n:
        c = source[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(source[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
        else:
            out.append(c)
            i += 1
    return "".join(out)


def count_identifiers(dirs):
    counts = {}
    for path in swift_files(dirs):
        for name in IDENTIFIER.findall(without_comments(path.read_text(errors="replace"))):
            counts[name] = counts.get(name, 0) + 1
    return counts


def declarations():
    """(name, kind, path, line) for every non-private declaration in tesseract/."""
    found, extended = [], {}
    for path in swift_files(DECLARING):
        lines = without_comments(path.read_text(errors="replace")).split("\n")
        for number, line in enumerate(lines, 1):
            for name in EXTENSION.findall(line):
                extended[name] = extended.get(name, 0) + 1
            match = DECLARATION.match(line)
            if not match:
                continue
            mods, attrs, name = match["mods"].split(), match["attrs"], match["name"]
            if "private" in mods or "fileprivate" in mods or "override" in mods:
                continue
            context = " ".join(lines[max(0, number - 3):number])
            if name in RUNTIME_NAMES or any(a in context for a in RUNTIME_ATTRS):
                continue
            if name.endswith(("ForTesting", "ForTests")):
                continue
            found.append((name, match["kind"], path.relative_to(ROOT), number))
    return found, extended


def main():
    allowed = {}
    for line in ALLOW.read_text().splitlines() if ALLOW.exists() else []:
        entry = line.split("#", 1)[0].strip()
        if entry:
            allowed[entry] = line
    found, extended = declarations()
    app, tests = count_identifiers(APP), count_identifiers(TESTS)
    declared = {}
    for name, *_ in found:
        declared[name] = declared.get(name, 0) + 1

    test_only, seen = [], set()
    for name, kind, path, number in found:
        if name in seen:
            continue
        seen.add(name)
        # Each declaration and extension is one occurrence of the name itself.
        uses = app.get(name, 0) - declared[name] - extended.get(name, 0)
        if uses <= 0 and tests.get(name, 0) > 0:
            test_only.append((name, kind, path, number, tests[name]))

    failures = 0
    for name, kind, path, number, test_refs in test_only:
        if name in allowed:
            continue
        failures += 1
        print(f"{path}:{number}: {kind} {name} is used only by tests ({test_refs} references)")
    flagged = {name for name, *_ in test_only}
    for name in sorted(set(allowed) - flagged):
        failures += 1
        print(f"{ALLOW.relative_to(ROOT)}: {name} is no longer used only by tests: remove the entry")
    if failures:
        print(
            "\nDelete code that only tests use, with its tests, or list it in "
            f"{ALLOW.relative_to(ROOT)} with the reason it stays."
        )
        return 1
    print(f"test-only code: none outside {ALLOW.relative_to(ROOT)} ({len(allowed)} listed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
