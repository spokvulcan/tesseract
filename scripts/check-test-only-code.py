#!/usr/bin/env python3
"""Fails when app code is used only by tests.

A type or function in tesseract/ that no app code references but a test does
is dead code its tests keep alive: ServerRunState outlived its last caller by
three months that way. Each one is either deleted with its tests or listed in
scripts/test-only-code.allow with the reason it stays (a test seam, a slice in
progress, a caller in another target).

Two passes. The name pass is a source scan: it counts identifier references
outside comments, so a name shared with live code elsewhere (`stop`, `flush`)
hides a dead one, never the reverse; it runs in about a second, with no build.
The index pass reads the compiler's index of this checkout's last build and
asks of each declaration's own symbol whether app code references it, so a
shared name hides nothing: it reports `Type.member`. It runs when the build's
index is newer than every source (after `scripts/test.sh`, in about three
seconds) or on the store `--index-store` names (CI), and is skipped otherwise.
An override or a protocol witness is called through another symbol, so the
index pass leaves it to the name pass, as it does a name the phone app or the
share extension spells (the index may hold no build of theirs).

Allow entries: a bare name (`flushManifestForTesting`) allows that name in
either pass; `Type.member` allows one member, and `Type.*` every member of
a test-only type, in the index pass.

Usage: scripts/check-test-only-code.py [--index-store <Index.noindex/DataStore>]
"""

import ctypes
import os
import pathlib
import plistlib
import re
import subprocess
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
            found.append((name, match["kind"], path.relative_to(ROOT), number))
    return found, extended


# MARK: - The index pass

DECLARATION_ROLES = 1 << 0 | 1 << 1
REFERENCE_ROLE = 1 << 2
CHILD_OF = 1 << 9
OVERRIDE_OF = 1 << 11
RECORD_DEPENDENCY = 2
PROTOCOL_KIND = 8


class _StringRef(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("length", ctypes.c_size_t)]

    def text(self):
        return ctypes.string_at(self.data, self.length).decode() if self.length else ""


def _index_library():
    swift = subprocess.run(["xcrun", "--find", "swift"], capture_output=True, text=True)
    path = pathlib.Path(swift.stdout.strip()).parent.parent / "lib" / "libIndexStore.dylib"
    lib = ctypes.CDLL(str(path))
    pointer, string = ctypes.c_void_p, _StringRef

    def bind(name, restype, *argtypes):
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, list(argtypes)
        return function

    applier = lambda *types: ctypes.CFUNCTYPE(ctypes.c_bool, pointer, *types)
    lib.UNITS, lib.EACH = applier(string), applier(pointer)
    error = ctypes.POINTER(pointer)
    api = {
        "store": bind("indexstore_store_create", pointer, ctypes.c_char_p, error),
        "units": bind("indexstore_store_units_apply_f", ctypes.c_bool, pointer, ctypes.c_uint,
                      pointer, lib.UNITS),
        "unit": bind("indexstore_unit_reader_create", pointer, pointer, ctypes.c_char_p, error),
        "unit_done": bind("indexstore_unit_reader_dispose", None, pointer),
        "dependencies": bind("indexstore_unit_reader_dependencies_apply_f", ctypes.c_bool,
                             pointer, pointer, lib.EACH),
        "dependency_kind": bind("indexstore_unit_dependency_get_kind", ctypes.c_int, pointer),
        "dependency_path": bind("indexstore_unit_dependency_get_filepath", string, pointer),
        "dependency_name": bind("indexstore_unit_dependency_get_name", string, pointer),
        "record": bind("indexstore_record_reader_create", pointer, pointer, ctypes.c_char_p, error),
        "record_done": bind("indexstore_record_reader_dispose", None, pointer),
        "occurrences": bind("indexstore_record_reader_occurrences_apply_f", ctypes.c_bool,
                            pointer, pointer, lib.EACH),
        "symbol": bind("indexstore_occurrence_get_symbol", pointer, pointer),
        "roles": bind("indexstore_occurrence_get_roles", ctypes.c_uint64, pointer),
        "line": bind("indexstore_occurrence_get_line_col", None, pointer,
                     ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)),
        "relations": bind("indexstore_occurrence_relations_apply_f", ctypes.c_bool, pointer,
                          pointer, lib.EACH),
        "relation_roles": bind("indexstore_symbol_relation_get_roles", ctypes.c_uint64, pointer),
        "relation_symbol": bind("indexstore_symbol_relation_get_symbol", pointer, pointer),
        "usr": bind("indexstore_symbol_get_usr", string, pointer),
        "kind": bind("indexstore_symbol_get_kind", ctypes.c_uint64, pointer),
        "name": bind("indexstore_symbol_get_name", string, pointer),
    }
    return lib, api


def build_index_store(explicit):
    """This checkout's index store and "", or None and why the pass is skipped."""
    if explicit:
        store = pathlib.Path(explicit)
    else:
        project = str(ROOT / "tesseract.xcodeproj")
        store = None
        derived = pathlib.Path.home() / "Library/Developer/Xcode/DerivedData"
        for folder in sorted(derived.glob("tesseract-*")):
            try:
                with open(folder / "info.plist", "rb") as plist:
                    if plistlib.load(plist).get("WorkspacePath") == project:
                        store = folder / "Index.noindex" / "DataStore"
                        break
            except (OSError, plistlib.InvalidFileException):
                continue
        if store is None:
            return None, "no build of this checkout yet"
    units = store / "v5" / "units"
    if not units.is_dir():
        return None, f"no index at {store}"
    indexed = max((entry.stat().st_mtime for entry in os.scandir(units)), default=0)
    edited = max((path.stat().st_mtime for path in swift_files(DECLARING + TESTS)), default=0)
    if edited > indexed and not explicit:
        return None, "the build's index is older than the sources: build to include it"
    return store, ""


def index_test_only(store, found):
    """(path, line, Type.member) for each found declaration whose own symbol
    tests reference and app code doesn't."""
    lib, api = _index_library()
    sites = {(str(path), number): name for name, _, path, number in found}
    names = set(sites.values())
    error = ctypes.c_void_p()
    handle = api["store"](str(store).encode(), ctypes.byref(error))
    if not handle:
        return []
    root = str(ROOT) + "/"
    unit_names, records = [], {}
    on_unit = lib.UNITS(lambda _, name: unit_names.append(name.text()) or True)
    api["units"](handle, 0, None, on_unit)

    def on_dependency(_, dependency):
        if api["dependency_kind"](dependency) == RECORD_DEPENDENCY:
            path = api["dependency_path"](dependency).text()
            if path.startswith(root):
                records.setdefault(api["dependency_name"](dependency).text(), path[len(root):])
        return True

    each_dependency = lib.EACH(on_dependency)
    for unit in unit_names:
        reader = api["unit"](handle, unit.encode(), ctypes.byref(error))
        if reader:
            api["dependencies"](reader, None, each_dependency)
            api["unit_done"](reader)

    defined, witnesses, referenced, requirements = {}, set(), {}, set()
    line, column = ctypes.c_uint(), ctypes.c_uint()
    relation = {"override": False, "parent": "", "in_protocol": False}

    def on_relation(_, edge):
        roles = api["relation_roles"](edge)
        relation["override"] |= bool(roles & OVERRIDE_OF)
        if roles & CHILD_OF and not relation["parent"]:
            parent = api["relation_symbol"](edge)
            relation["parent"] = api["name"](parent).text()
            relation["in_protocol"] = api["kind"](parent) == PROTOCOL_KIND
        return True

    each_relation = lib.EACH(on_relation)
    for record, path in records.items():
        reader = api["record"](handle, record.encode(), ctypes.byref(error))
        if not reader:
            continue
        top = path.split("/", 1)[0]

        def on_occurrence(_, occurrence, path=path, top=top):
            roles = api["roles"](occurrence)
            if roles & DECLARATION_ROLES:
                api["line"](occurrence, ctypes.byref(line), ctypes.byref(column))
                name = sites.get((path, line.value))
                if name is None:
                    return True
                symbol = api["symbol"](occurrence)
                if api["name"](symbol).text().split("(", 1)[0] != name:
                    return True
                relation.update(override=False, parent="", in_protocol=False)
                api["relations"](occurrence, None, each_relation)
                usr = api["usr"](symbol).text()
                if relation["override"]:
                    witnesses.add(usr)
                if relation["in_protocol"]:
                    requirements.add(name)
                parent = relation["parent"]
                defined[usr] = (path, line.value, f"{parent}.{name}" if parent else name)
            elif roles & REFERENCE_ROLE:
                symbol = api["symbol"](occurrence)
                if api["name"](symbol).text().split("(", 1)[0] in names:
                    referenced.setdefault(api["usr"](symbol).text(), set()).add(top)
            return True

        each_occurrence = lib.EACH(on_occurrence)
        api["occurrences"](reader, None, each_occurrence)
        api["record_done"](reader)

    elsewhere = count_identifiers([d for d in APP if d not in DECLARING])
    found_here = []
    for usr, (path, number, qualified) in sorted(defined.items(), key=lambda item: item[1]):
        where = referenced.get(usr, set())
        if usr in witnesses or not where & set(TESTS) or where & set(APP):
            continue
        # A protocol's requirement of the same name may be how the app calls
        # it, through a conformance declared elsewhere: the name pass's call.
        if qualified.rsplit(".", 1)[-1] in requirements:
            continue
        if elsewhere.get(qualified.rsplit(".", 1)[-1]):
            continue
        found_here.append((path, number, qualified))
    return found_here


def main():
    explicit_store = None
    if "--index-store" in sys.argv:
        explicit_store = sys.argv[sys.argv.index("--index-store") + 1]
    allowed = {}
    for line in ALLOW.read_text().splitlines() if ALLOW.exists() else []:
        entry = line.split("#", 1)[0].strip()
        if entry:
            allowed[entry] = line
    members = {entry for entry in allowed if "." in entry}
    allowed = {entry: line for entry, line in allowed.items() if "." not in entry}
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

    store, why = build_index_store(explicit_store)
    if store is None and explicit_store:
        failures += 1
        print(f"index pass: {why}")
    elif store is None:
        print(f"index pass skipped: {why}")
    else:
        used = set()
        for path, number, qualified in index_test_only(store, found):
            container, _, member = qualified.rpartition(".")
            allowing = [entry for entry in (qualified, f"{container}.*") if entry in members]
            if member in allowed or allowing or member in flagged:
                used.update(allowing)
                continue
            failures += 1
            print(f"{path}:{number}: {qualified} is used only by tests (the build's index)")
        for entry in sorted(members - used):
            failures += 1
            print(f"{ALLOW.relative_to(ROOT)}: {entry} is no longer used only by tests: remove the entry")
    if failures:
        print(
            "\nDelete code that only tests use, with its tests, or list it in "
            f"{ALLOW.relative_to(ROOT)} with the reason it stays."
        )
        return 1
    print(f"test-only code: none outside {ALLOW.relative_to(ROOT)} ({len(allowed) + len(members)} listed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
