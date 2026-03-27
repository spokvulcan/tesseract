---
name: web-research
description: Use this skill when answering questions that require current, factual, or verifiable information from the web.
---

# Web Research

## Core Rule

**Search snippets are navigation aids, not data sources.** They are often outdated, truncated, or misleading. Never cite a snippet as fact. Always fetch the actual page.

## Workflow

### 1. Plan search queries

Before searching, decide on 2-3 queries from different angles:
- Official/primary source query (e.g., "Swift 6.2 release notes")
- Community/analysis query (e.g., "Swift 6.2 new features overview")
- Stats/data query if needed (e.g., "product X benchmark comparison 2026")

### 2. Search and select URLs

Run `web_search` for your first query. Scan results for:
- **Official sources** (docs, patch notes, manufacturer sites) — prefer these
- **Recent dates** in titles or snippets — prefer these
- **Reputable secondary sources** (major news outlets, established wikis, stats sites)

Do NOT extract data from snippets. Use them only to choose which URLs to fetch.

### 3. Fetch actual pages (MANDATORY)

Fetch **at least 2-3 pages** using `web_fetch` before answering. This is not optional.
- Start with the most authoritative URL from search results
- If the first fetch gives thin content or fails, try alternative URLs
- Use `max_chars: 20000` to keep context manageable; use higher values only when you need comprehensive detail from a single long page

### 4. Check freshness

When reading fetched content, look for:
- Publication or update dates
- Version numbers, patch numbers, timestamps
- Language like "as of [date]" or "last updated"

If the newest source is more than 6 months old for a time-sensitive topic, tell the user: "The most recent source I found is from [date]."

### 5. Cross-reference

Compare facts across your fetched sources. If sources disagree:
- State the disagreement
- Prefer the more authoritative or more recent source
- Never silently pick one version

### 6. Cite sources

Every factual claim must have a source URL. Format:
- Inline: "The base damage was reduced to 300 ([source](url))"
- Or a Sources section at the end with numbered references

## Error Recovery

- **Fetch fails or returns thin content**: Try the next URL from search results. If multiple fetches fail, run a refined search query.
- **No relevant results**: Try alternative query phrasing. Broaden or narrow the search terms.
- **All sources are old**: State this clearly. Do NOT fill gaps with your training data — say "I couldn't find current information on [topic]."

## What NOT To Do

- Do NOT answer factual questions from search snippets alone
- Do NOT present your training data as current facts when the user asked for current information
- Do NOT run many searches without fetching any pages — this is the failure mode this skill exists to prevent
- Do NOT assume a snippet's date reflects current reality
