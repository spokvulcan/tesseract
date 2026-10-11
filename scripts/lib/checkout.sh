# Sourced by scripts/test.sh, scripts/dev.sh and scripts/bench.sh: where this
# checkout's build lives. Every worktree has its own DerivedData, so a glob
# over DerivedData/tesseract-* names whichever checkout sorts first: it ran a
# stale sibling's binary once (#165), and a stale-module clean through it
# emptied another worktree's cache.

# This checkout's DerivedData: Xcode's default location, whose info.plist
# names the project it belongs to. Prints nothing and fails until the project
# at $1 has built.
checkout_derived_data() {
    local project="$1" dir
    for dir in "$HOME"/Library/Developer/Xcode/DerivedData/tesseract-*; do
        [ -f "$dir/info.plist" ] || continue
        if [ "$(plutil -extract WorkspacePath raw "$dir/info.plist" 2>/dev/null)" = "$project" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# The products directory xcodebuild builds configuration $3 of scheme $2 of
# project $1 into. awk reads xcodebuild's whole output: stopping at the first
# match ends xcodebuild with SIGPIPE, which `pipefail` makes the caller's
# exit (141).
built_products_dir() {
    local project="$1" scheme="$2" configuration="$3"
    xcodebuild -project "$project" -scheme "$scheme" \
        -configuration "$configuration" -destination 'platform=macOS' \
        -skipPackagePluginValidation -showBuildSettings 2>/dev/null \
        | awk -F' = ' '/[[:space:]]BUILT_PRODUCTS_DIR =/ && !found { print $2; found = 1 }'
}
