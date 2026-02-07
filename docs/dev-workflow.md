# Dev Workflow

`scripts/dev.sh` automates the build-kill-run cycle so you can iterate on code changes without manually using Xcode.

## Commands

| Command | What it does |
|---------|-------------|
| `scripts/dev.sh build` | Build the project via `xcodebuild`. Shows only errors, warnings, and the build result. |
| `scripts/dev.sh run` | Kill any running tesseract process and launch the most recent build. |
| `scripts/dev.sh dev` | Build + kill + run in one shot. **This is the main command.** |
| `scripts/dev.sh clean` | Run `xcodebuild clean` and delete DerivedData for tesseract. |
| `scripts/dev.sh log` | Tail the macOS system log filtered to tesseract's subsystem/process. |

## Typical Development Loop

```
1. Edit code in your editor / via Claude Code
2. scripts/dev.sh dev        # builds, kills old app, launches new one
3. Test the change manually
4. Repeat
```

If you only need to relaunch without rebuilding (e.g., testing a different runtime condition):

```
scripts/dev.sh run
```

## When You Still Need Xcode

- Changing build settings, schemes, or signing configuration
- Editing the storyboard or asset catalog
- Debugging with breakpoints or Instruments
- Resolving Swift package dependency issues
