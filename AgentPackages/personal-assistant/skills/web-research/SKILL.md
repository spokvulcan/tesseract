---
name: web-research
description: "MANDATORY when the user asks about news, current events, latest updates, prices, weather, scores, releases, or any question requiring up-to-date or verifiable web information. Load this skill BEFORE calling web_search or web_fetch."
---

# Web Research

**You MUST follow this workflow whenever you use `web_search` or `web_fetch`.** Do not skip steps.

## Core Rule

**Search snippets are navigation aids, not data sources.** They are often outdated, truncated, or misleading. Never cite a snippet as fact. Always fetch the actual page.

## Workflow

### 1. Search with 2-3 queries from different angles

- Primary source query (e.g., "Swift 6.2 release notes site:swift.org")
- Community/analysis query (e.g., "Swift 6.2 new features overview")
- Data query if relevant (e.g., "product X benchmark 2026")

Run `web_search` for each. Scan results for official sources, recent dates, and reputable outlets. Do NOT extract data from snippets — use them only to pick URLs.

### 2. Fetch at least 2-3 pages (MANDATORY)

Use `web_fetch` on the best URLs. This step is not optional.
- Start with the most authoritative URL
- If a fetch returns thin content, try the next URL
- Use `max_chars: 20000` to keep context manageable

### 3. Check freshness

Look for publication dates, version numbers, and "last updated" timestamps in fetched content. If the newest source is over 6 months old for a time-sensitive topic, tell the user.

### 4. Cross-reference and cite

Compare facts across fetched pages. If sources disagree, state the disagreement and prefer the more authoritative or recent one. Every factual claim must have a source URL — either inline or in a Sources section.

## Error Recovery

- **Fetch fails**: Try the next URL. If multiple fail, run a refined search.
- **No results**: Rephrase the query. Broaden or narrow terms.
- **All sources are old**: Say so. Do NOT fill gaps with training data.

## What NOT To Do

- Do NOT answer from search snippets alone — always fetch actual pages
- Do NOT present training data as current facts
- Do NOT call web_search repeatedly without fetching any pages
- Do NOT assume a snippet's date reflects current reality
