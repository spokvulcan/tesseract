---
name: web-research
description: "MANDATORY when the user asks about news, current events, latest updates, prices, weather, scores, releases, or any question requiring up-to-date or verifiable web information. Load this skill BEFORE using the browser tools (browser.search, browser.fetch, browser.navigate)."
---

# Web Research

**You MUST follow this workflow whenever you research the web with the browser tools.** Do not skip steps.

Your web access is the real browser (the `browser.*` tools). There is one surface for everything — search, read, and interact:

- `browser.search` — anonymous web search; returns titles, URLs, and snippets.
- `browser.fetch` — anonymously fetch a URL and read it as clean Markdown (cookieless, outside your logged-in profile).
- `browser.navigate` + `browser.read_page` — open a page in your authenticated browser and read it (use this when a page needs your login or is blocked anonymously).
- `browser.page_map` + `browser.click` + `browser.type` — interact: follow links, expand sections, run a site's own search, page through results, fill forms.
- `browser.find` — locate a string on the current page without re-reading it all.

## Core Rule

**Search snippets are navigation aids, not data sources.** They are often outdated, truncated, or misleading. Never cite a snippet as fact. Always open and read the actual page.

## Workflow

### 1. Search with 2-3 queries from different angles

- Primary/official source query (e.g., "Swift 6.2 release notes site:swift.org")
- Community/analysis query (e.g., "Swift 6.2 new features overview")
- Data query if relevant (e.g., "product X benchmark 2026")

Run `browser.search` for each. Scan results for official sources, recent dates, and reputable outlets. Do NOT extract data from snippets — use them only to pick URLs.

### 2. Read at least 2-3 pages (MANDATORY)

Open the best URLs and read their real content. This step is not optional.

- Start with the most authoritative URL.
- Prefer `browser.fetch` for a quick anonymous read of a static/public page.
- If a fetch returns thin content, is blocked, or the page needs your login or renders dynamically, escalate: `browser.navigate` to it, then `browser.read_page`. Long pages paginate — follow the cursor hint for more.

### 3. Interact when the answer is behind a click

Don't stop at the first page if the information lives one interaction away. Use `browser.page_map` to see the links, buttons, and fields, then `browser.click` / `browser.type` to:

- follow a "docs" / "changelog" / "pricing" link,
- run the site's own in-page search or filter,
- page through result lists,
- expand collapsed sections or accept a consent gate.

Re-read with `browser.read_page` after each interaction. This is how you get the most out of a source instead of quoting a summary of it.

### 4. Check freshness

Look for publication dates, version numbers, and "last updated" timestamps in the pages you read. If the newest source is over 6 months old for a time-sensitive topic, tell the user.

### 5. Cross-reference and cite

Compare facts across the pages you read. If sources disagree, state the disagreement and prefer the more authoritative or recent one. Every factual claim must have a source URL — either inline or in a Sources section.

## Error Recovery

- **Fetch fails or is thin/blocked**: escalate to `browser.navigate` + `browser.read_page` (your authenticated browser), or try the next URL.
- **Search returns no structured results**: it falls back to the results page text — read it, then rephrase the query (broaden or narrow terms).
- **Page needs a login you have**: `browser.navigate` uses your logged-in sessions; open the page directly rather than fetching it anonymously.
- **All sources are old**: say so. Do NOT fill gaps with training data.

## What NOT To Do

- Do NOT answer from search snippets alone — always open and read the pages.
- Do NOT present training data as current facts.
- Do NOT run `browser.search` repeatedly without reading any pages.
- Do NOT give up at the first page when the answer is one click deeper — interact with the site.
