#!/usr/bin/env python3
"""Claude Code PreToolUse hook (matcher: Bash).

Guards `gh pr create` / `gh pr edit --title`: the title must be a
Conventional Commit, because this repo squash-merges and the PR title becomes
the commit on main (see .github/workflows/pr-title.yml and docs/releasing.md).
Same type list as amannn/action-semantic-pull-request's default.

Reads the hook JSON on stdin. Exits 0 silently to allow; prints a deny
decision to block. Commands that are not `gh pr create|edit` pass through.
"""
import json
import re
import shlex
import sys

TYPES = "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
PATTERN = re.compile(rf"^({TYPES})(\([\w\$\.\-\*/ ]+\))?!?: \S.*$")


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bgh\s+pr\s+(create|edit)\b", command):
        return

    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()

    subcommand = None
    title = None
    for index, token in enumerate(argv):
        if subcommand is None:
            if (token == "gh" and argv[index + 1:index + 2] == ["pr"]
                    and argv[index + 2:index + 3] in (["create"], ["edit"])):
                subcommand = argv[index + 2]
            continue
        if token in ("--title", "-t"):
            title = argv[index + 1] if index + 1 < len(argv) else ""
        elif token.startswith("--title="):
            title = token[len("--title="):]

    if subcommand is None:
        return
    if title is None:
        if subcommand == "edit":
            return  # body/labels only; nothing to check
        deny("gh pr create needs an explicit --title in Conventional Commits "
             "form (e.g. `fix(server): ...`). This repo squash-merges: the PR "
             "title becomes the commit on main and the changelog line.")
    if not PATTERN.match(title.strip()):
        deny(f"PR title is not a Conventional Commit: {title!r}. Required: "
             f"<type>(<scope>)?: <summary>, type in "
             f"[{TYPES.replace('|', ', ')}]. This repo squash-merges: the PR "
             "title becomes the commit on main and the changelog line; "
             "pr-title.yml would reject it.")


if __name__ == "__main__":
    main()
