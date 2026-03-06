# Pi Skills System — How It Works

## Core Design: Progressive Disclosure

The model does **NOT** get full skill content in the system prompt. It only gets a lightweight catalog — name, description, and file path — wrapped in XML:

```xml
<available_skills>
  <skill>
    <name>brave-search</name>
    <description>Web search via Brave Search API.</description>
    <location>/home/user/.pi/agent/skills/brave-search/SKILL.md</location>
  </skill>
</available_skills>
```

The instruction preceding this block is the key guidance:

```
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory
(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.
```

The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory.

<available_skills>

## How It Works

1. **Model-driven activation** — No trigger/match logic exists. The model reads the descriptions, decides if a task matches, and calls `read` on the SKILL.md path to get full instructions. The docs acknowledge this is imperfect: _"models don't always do this"_.

2. **User-explicit activation** — User types `/skill:name args`. The harness reads the SKILL.md, strips frontmatter, wraps the body in `<skill>` XML tags, and injects it as the user message:

   ```xml
   <skill name="brave-search" location="/path/to/SKILL.md">
   References are relative to /path/to/brave-search.

   [full skill body here]
   </skill>

   user's query here
   ```

3. **`disable-model-invocation: true`** — Hides the skill from the system prompt entirely. Only accessible via explicit `/skill:name`.

4. **Conditional inclusion** — Skills are only added to the system prompt if the `read` tool is available (otherwise the model couldn't load them).

## SKILL.md File Format

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: skill-name
description: Short description of what this skill does
---

# Skill Name

Usage instructions, examples, etc.
Scripts are in: {baseDir}/
```

**Required fields:** `name`, `description`
**Optional fields:** `license`, `compatibility`, `metadata`, `allowed-tools`, `disable-model-invocation`

## Skill Discovery Locations

### Pi Coding Agent

- User-level: `~/.pi/agent/skills/`
- Project-level: `.pi/skills/` in working directory

### Mom (Slack Bot)

- Workspace-level (global): `${workspacePath}/skills/`
- Channel-specific: `${channelPath}/skills/` (overrides workspace on name collision)

## Mom's Addition

Mom uses the same `formatSkillsForPrompt()` function but also teaches the model to **create** new skills — it includes the SKILL.md template format with frontmatter spec so the bot can build its own reusable CLI tools.

## Key Takeaway

The pattern is: **small catalog in system prompt, full content loaded on demand via `read` tool**. ~50-100 tokens per skill upfront instead of potentially thousands. The model is trusted to match task-to-skill by description alone.
