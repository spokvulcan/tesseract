//! tesseract-highlight — syntax highlighting and diff computation for the
//! Tesseract transcript's Tool Panels (ADR-0029).
//!
//! The FFI is deliberately narrow and returns plain data: styled spans as
//! (text, role) pairs and diff rows as (kind, line numbers, segments). Swift
//! owns all rendering, theming, and UI; no color values cross this boundary —
//! only named `TokenRole`s, which Swift maps to the Code Accent Palette.

use std::sync::OnceLock;

use syntect::easy::HighlightLines;
use syntect::highlighting::{
    Color, ScopeSelectors, StyleModifier, Theme, ThemeItem, ThemeSettings,
};
use syntect::parsing::{SyntaxReference, SyntaxSet};

uniffi::setup_scaffolding!();

/// Inputs larger than this are returned as plain text — the transcript never
/// needs to highlight megabytes, and pathological inputs must not stall a
/// panel render.
const MAX_HIGHLIGHT_BYTES: usize = 2 * 1024 * 1024;

// MARK: - FFI types

/// The named color roles of the Code Accent Palette. Swift resolves each to a
/// system-semantic color; this crate never sees actual colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum TokenRole {
    Plain,
    Keyword,
    StringLit,
    Number,
    Constant,
    Comment,
    TypeName,
    FunctionName,
    Attribute,
    VariableName,
}

/// One run of same-role text within a line. Spans carry their text directly
/// (not offsets) so the FFI contract cannot go out of sync with Swift's
/// string indexing.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct HighlightSpan {
    pub text: String,
    pub role: TokenRole,
}

/// One line of highlighted code, without its trailing newline.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct HighlightedLine {
    pub spans: Vec<HighlightSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum DiffRowKind {
    Context,
    Added,
    Removed,
}

/// One run of text within a diff row. `emphasized` marks the word-level
/// changed range inside a modified line (similar's inline diff).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct DiffSegment {
    pub text: String,
    pub role: TokenRole,
    pub emphasized: bool,
}

/// One row of a line diff. Line numbers are 1-based; `old_line` is absent on
/// added rows, `new_line` on removed rows.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct DiffRow {
    pub kind: DiffRowKind,
    pub old_line: Option<u32>,
    pub new_line: Option<u32>,
    pub segments: Vec<DiffSegment>,
}

// MARK: - Entry points

/// Highlight `code` line by line. `language_hint` is a fence token ("swift"),
/// an extension ("rs"), or a file path ("Sources/Foo.swift"); unknown hints
/// fall back to plain text (single Plain span per line).
#[uniffi::export]
pub fn highlight(code: String, language_hint: String) -> Vec<HighlightedLine> {
    highlight_lines(&code, &language_hint)
        .into_iter()
        .map(|spans| HighlightedLine { spans: spans_to_ffi(spans) })
        .collect()
}

/// Line-diff `old` against `new` (word-level inline emphasis on modified
/// lines), highlighting both sides with `language_hint`. Rows come back in
/// document order with 1-based line numbers on each side.
#[uniffi::export]
pub fn diff_highlighted(old: String, new: String, language_hint: String) -> Vec<DiffRow> {
    let old_lines = highlight_lines(&old, &language_hint);
    let new_lines = highlight_lines(&new, &language_hint);

    let diff = similar::TextDiff::from_lines(&old, &new);
    let mut rows = Vec::new();

    for op in diff.ops() {
        for change in diff.iter_inline_changes(op) {
            let (kind, line_spans) = match change.tag() {
                similar::ChangeTag::Equal => {
                    (DiffRowKind::Context, lookup(&new_lines, change.new_index()))
                }
                similar::ChangeTag::Delete => {
                    (DiffRowKind::Removed, lookup(&old_lines, change.old_index()))
                }
                similar::ChangeTag::Insert => {
                    (DiffRowKind::Added, lookup(&new_lines, change.new_index()))
                }
            };

            // The inline runs cover the row's text in order; merge them with
            // the highlighted spans at the union of their boundaries.
            let runs: Vec<(bool, String)> = change
                .iter_strings_lossy()
                .map(|(emphasized, value)| (emphasized, value.into_owned()))
                .collect();

            rows.push(DiffRow {
                kind,
                old_line: change.old_index().map(|i| i as u32 + 1),
                new_line: change.new_index().map(|i| i as u32 + 1),
                segments: merge_segments(line_spans, &runs),
            });
        }
    }

    rows
}

// MARK: - Highlighting internals

fn syntax_set() -> &'static SyntaxSet {
    static SET: OnceLock<SyntaxSet> = OnceLock::new();
    // bat's extended grammar set (newline variant) — syntect's own defaults
    // lack Swift, the one language this transcript shows most.
    SET.get_or_init(|| two_face::syntax::extra_newlines())
}

/// A theme whose only purpose is scope matching: each item's foreground red
/// channel encodes a `TokenRole` discriminant, so syntect's own selector
/// specificity decides the winning role per token.
fn role_theme() -> &'static Theme {
    static THEME: OnceLock<Theme> = OnceLock::new();
    THEME.get_or_init(|| {
        let items: &[(&str, TokenRole)] = &[
            ("comment, punctuation.definition.comment", TokenRole::Comment),
            ("string", TokenRole::StringLit),
            ("constant.numeric", TokenRole::Number),
            (
                "constant.language, constant.character, constant.other",
                TokenRole::Constant,
            ),
            (
                "keyword, storage.type, storage.modifier, keyword.operator",
                TokenRole::Keyword,
            ),
            (
                "entity.name.type, entity.name.class, entity.name.struct, entity.name.enum, \
                 entity.other.inherited-class, support.type, support.class",
                TokenRole::TypeName,
            ),
            (
                "entity.name.function, support.function, variable.function",
                TokenRole::FunctionName,
            ),
            (
                "entity.other.attribute-name, meta.annotation.identifier, \
                 punctuation.definition.annotation",
                TokenRole::Attribute,
            ),
            (
                "variable.parameter, variable.language, variable.other.member",
                TokenRole::VariableName,
            ),
        ];

        let scopes = items
            .iter()
            .filter_map(|(selector, role)| {
                let selectors: ScopeSelectors = selector.parse().ok()?;
                Some(ThemeItem {
                    scope: selectors,
                    style: StyleModifier {
                        foreground: Some(role_color(*role)),
                        background: None,
                        font_style: None,
                    },
                })
            })
            .collect();

        Theme {
            settings: ThemeSettings {
                foreground: Some(role_color(TokenRole::Plain)),
                ..ThemeSettings::default()
            },
            scopes,
            ..Theme::default()
        }
    })
}

fn role_color(role: TokenRole) -> Color {
    Color { r: role as u8, g: 0, b: 0, a: 0xFF }
}

fn role_from_color(color: Color) -> TokenRole {
    match color.r {
        1 => TokenRole::Keyword,
        2 => TokenRole::StringLit,
        3 => TokenRole::Number,
        4 => TokenRole::Constant,
        5 => TokenRole::Comment,
        6 => TokenRole::TypeName,
        7 => TokenRole::FunctionName,
        8 => TokenRole::Attribute,
        9 => TokenRole::VariableName,
        _ => TokenRole::Plain,
    }
}

fn find_syntax<'s>(set: &'s SyntaxSet, hint: &str) -> Option<&'s SyntaxReference> {
    let trimmed = hint.trim();
    if trimmed.is_empty() {
        return None;
    }
    // A path hint reduces to its last component ("Sources/Foo.swift" → "Foo.swift").
    let name = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if let Some(syntax) = set.find_syntax_by_token(name) {
        return Some(syntax);
    }
    let extension = name.rsplit('.').next().unwrap_or(name);
    set.find_syntax_by_token(extension)
}

/// Highlight into per-line role spans; trailing newlines are stripped from
/// span text. Unknown language or oversized input → one Plain span per line.
fn highlight_lines(code: &str, language_hint: &str) -> Vec<Vec<(TokenRole, String)>> {
    let plain = |code: &str| -> Vec<Vec<(TokenRole, String)>> {
        split_lines(code)
            .map(|line| {
                if line.is_empty() {
                    Vec::new()
                } else {
                    vec![(TokenRole::Plain, line.to_string())]
                }
            })
            .collect()
    };

    if code.len() > MAX_HIGHLIGHT_BYTES {
        return plain(code);
    }
    let set = syntax_set();
    let Some(syntax) = find_syntax(set, language_hint) else {
        return plain(code);
    };

    let mut highlighter = HighlightLines::new(syntax, role_theme());
    let mut lines = Vec::new();
    for line in syntect::util::LinesWithEndings::from(code) {
        match highlighter.highlight_line(line, set) {
            Ok(regions) => {
                let mut spans: Vec<(TokenRole, String)> = Vec::new();
                for (style, text) in regions {
                    let text = text.trim_end_matches('\n');
                    if text.is_empty() {
                        continue;
                    }
                    let role = role_from_color(style.foreground);
                    match spans.last_mut() {
                        Some((last_role, last_text)) if *last_role == role => {
                            last_text.push_str(text)
                        }
                        _ => spans.push((role, text.to_string())),
                    }
                }
                lines.push(spans);
            }
            Err(_) => lines.push(vec![(TokenRole::Plain, line.trim_end_matches('\n').to_string())]),
        }
    }
    if code.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

/// `str::lines` that preserves a final empty line only when the text is empty
/// — mirrors `LinesWithEndings` semantics for the plain-text path.
fn split_lines(code: &str) -> impl Iterator<Item = &str> {
    let mut lines: Vec<&str> = code.lines().collect();
    if code.is_empty() {
        lines.push("");
    }
    lines.into_iter()
}

fn lookup(lines: &[Vec<(TokenRole, String)>], index: Option<usize>) -> Vec<(TokenRole, String)> {
    index.and_then(|i| lines.get(i)).cloned().unwrap_or_default()
}

fn spans_to_ffi(spans: Vec<(TokenRole, String)>) -> Vec<HighlightSpan> {
    spans
        .into_iter()
        .map(|(role, text)| HighlightSpan { text, role })
        .collect()
}

// MARK: - Segment merging

/// Merge a line's highlight spans with its inline-diff emphasis runs by
/// splitting at the union of both boundary sets. If the two texts disagree
/// (highlighting unavailable, lossy decode), the runs win with Plain role —
/// the diff shape is the load-bearing part.
fn merge_segments(spans: Vec<(TokenRole, String)>, runs: &[(bool, String)]) -> Vec<DiffSegment> {
    let run_text: String = runs
        .iter()
        .map(|(_, text)| text.trim_end_matches('\n'))
        .collect();
    let span_text: String = spans.iter().map(|(_, text)| text.as_str()).collect();

    if span_text != run_text {
        return runs
            .iter()
            .filter_map(|(emphasized, text)| {
                let text = text.trim_end_matches('\n');
                (!text.is_empty()).then(|| DiffSegment {
                    text: text.to_string(),
                    role: TokenRole::Plain,
                    emphasized: *emphasized,
                })
            })
            .collect();
    }

    // Per-byte attribute arrays, then re-segment at attribute changes. Both
    // inputs split only at char boundaries, so every emitted boundary is one.
    let bytes = run_text.len();
    let mut roles = vec![TokenRole::Plain; bytes];
    let mut cursor = 0;
    for (role, text) in &spans {
        roles[cursor..cursor + text.len()].fill(*role);
        cursor += text.len();
    }
    let mut emphasis = vec![false; bytes];
    cursor = 0;
    for (emphasized, text) in runs {
        let len = text.trim_end_matches('\n').len();
        emphasis[cursor..cursor + len].fill(*emphasized);
        cursor += len;
    }

    let mut segments: Vec<DiffSegment> = Vec::new();
    let mut start = 0;
    for i in 1..=bytes {
        if i == bytes || roles[i] != roles[start] || emphasis[i] != emphasis[start] {
            segments.push(DiffSegment {
                text: run_text[start..i].to_string(),
                role: roles[start],
                emphasized: emphasis[start],
            });
            start = i;
        }
    }
    segments
}

// MARK: - Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(line: &HighlightedLine) -> String {
        line.spans.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn highlights_swift_keywords_and_strings() {
        let code = "let name = \"tesseract\" // local\n";
        let lines = highlight(code.into(), "swift".into());
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        assert!(spans.iter().any(|s| s.text == "let" && s.role == TokenRole::Keyword));
        assert!(spans
            .iter()
            .any(|s| s.text.contains("tesseract") && s.role == TokenRole::StringLit));
        assert!(spans
            .iter()
            .any(|s| s.text.contains("local") && s.role == TokenRole::Comment));
        assert_eq!(joined(&lines[0]), "let name = \"tesseract\" // local");
    }

    #[test]
    fn path_hints_resolve_by_extension() {
        let by_path = highlight("let x = 1".into(), "Sources/Foo.swift".into());
        assert!(by_path[0].spans.iter().any(|s| s.role == TokenRole::Keyword));
    }

    #[test]
    fn unknown_language_is_plain_and_lossless() {
        let code = "some ¬unicode∆ text\nsecond line";
        let lines = highlight(code.into(), "not-a-language".into());
        assert_eq!(lines.len(), 2);
        assert!(lines
            .iter()
            .all(|l| l.spans.iter().all(|s| s.role == TokenRole::Plain)));
        assert_eq!(joined(&lines[0]), "some ¬unicode∆ text");
        assert_eq!(joined(&lines[1]), "second line");
    }

    #[test]
    fn empty_input_yields_single_empty_line() {
        let lines = highlight(String::new(), "swift".into());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].spans.is_empty());
    }

    #[test]
    fn no_trailing_newline_keeps_last_line() {
        let lines = highlight("a\nb".into(), "txt".into());
        assert_eq!(lines.len(), 2);
        assert_eq!(joined(&lines[1]), "b");
    }

    #[test]
    fn diff_reports_line_numbers_and_kinds() {
        let old = "one\ntwo\nthree\n";
        let new = "one\nTWO\nthree\n";
        let rows = diff_highlighted(old.into(), new.into(), "txt".into());
        let kinds: Vec<DiffRowKind> = rows.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                DiffRowKind::Context,
                DiffRowKind::Removed,
                DiffRowKind::Added,
                DiffRowKind::Context
            ]
        );
        assert_eq!(rows[1].old_line, Some(2));
        assert_eq!(rows[1].new_line, None);
        assert_eq!(rows[2].new_line, Some(2));
        assert_eq!(rows[3].old_line, Some(3));
        assert_eq!(rows[3].new_line, Some(3));
    }

    #[test]
    fn diff_marks_inline_word_changes() {
        let old = "let count = compute(from: base)\n";
        let new = "let count = compute(from: offset)\n";
        let rows = diff_highlighted(old.into(), new.into(), "swift".into());
        let added: Vec<&DiffRow> = rows.iter().filter(|r| r.kind == DiffRowKind::Added).collect();
        assert_eq!(added.len(), 1);
        let emphasized: String = added[0]
            .segments
            .iter()
            .filter(|s| s.emphasized)
            .map(|s| s.text.as_str())
            .collect();
        assert!(emphasized.contains("offset"), "emphasized was: {emphasized:?}");
        // Highlighting and emphasis coexist on the same row.
        assert!(added[0].segments.iter().any(|s| s.role == TokenRole::Keyword));
        let row_text: String = added[0].segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(row_text, "let count = compute(from: offset)");
    }

    #[test]
    fn diff_handles_empty_sides() {
        let rows = diff_highlighted(String::new(), "new line\n".into(), "txt".into());
        assert!(rows.iter().any(|r| r.kind == DiffRowKind::Added));
        let rows = diff_highlighted("gone\n".into(), String::new(), "txt".into());
        assert!(rows.iter().any(|r| r.kind == DiffRowKind::Removed));
    }

    #[test]
    fn diff_survives_unicode_and_missing_final_newline() {
        let old = "café ≠ cafe";
        let new = "café == cafe";
        let rows = diff_highlighted(old.into(), new.into(), "txt".into());
        let texts: Vec<String> = rows
            .iter()
            .map(|r| r.segments.iter().map(|s| s.text.as_str()).collect())
            .collect();
        assert!(texts.iter().any(|t| t == "café == cafe"));
        assert!(texts.iter().any(|t| t == "café ≠ cafe"));
    }

    #[test]
    fn oversized_input_falls_back_to_plain() {
        let code = "x".repeat(MAX_HIGHLIGHT_BYTES + 1);
        let lines = highlight(code, "swift".into());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].spans[0].role, TokenRole::Plain);
    }
}
