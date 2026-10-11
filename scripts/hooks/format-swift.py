#!/usr/bin/env python3
"""Claude Code PostToolUse hook (matcher: Edit|MultiEdit|Write).

Formats a Swift file the moment an agent writes it, with the configuration
and the exclusions of the commit hook's swift-format (.pre-commit-config.yaml),
so `scripts/check.sh` finds nothing left to reformat and a commit never fails
on formatting alone.

Reads the hook JSON on stdin and always exits 0: a file swift-format can't
parse yet, mid-edit, stays as written, and the commit hook still checks it.
"""
import json
import pathlib
import re
import subprocess
import sys

# The commit hook's `exclude`, relative to the checkout's root.
EXCLUDED = re.compile(r"^(Vendor/|benchmarks/gather-sweep/|.*/\.build/)")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not path.endswith(".swift"):
        return
    file = pathlib.Path(path).resolve()
    # The checkout the file belongs to: its nearest folder with the
    # configuration, whichever worktree that is.
    root = next((p for p in file.parents if (p / ".swift-format").is_file()), None)
    if root is None or EXCLUDED.match(str(file.relative_to(root))):
        return
    try:
        subprocess.run(
            ["xcrun", "swift-format", "format", "--in-place", "--configuration",
             ".swift-format", str(file)],
            cwd=root, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


if __name__ == "__main__":
    main()
