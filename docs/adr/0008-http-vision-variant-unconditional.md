---
status: accepted
---

# HTTP requests load the vision variant for vision-capable models; the chat toggle stays chat-only

This records a decision from the 2026-06-11 client-integrations grilling (see
`CONTEXT.md` → **Client integrations**; PRD: #74). It deliberately reverses the
previous contract — "Vision mode is always sourced from settings — HTTP requests cannot
override it" — for the HTTP path only.

The trigger: generated client configs (the **Config Merge**) advertise image
input per model. A config snapshot cannot track a runtime toggle: with
`visionModeEnabled` off, a client that was told the model accepts images sends
them into a text-variant load and they silently go nowhere — the same failure
class observed with OpenCode on 2026-06-11, then on the client side. So
capability advertising is static (from `vision_config` in the model's own
`config.json`) and the server must honor what it advertises: an HTTP request
for a vision-capable model loads the vision variant, always.

Considered and rejected: *sticky upgrade on first image* (text variant until an
image arrives, then reload-and-stay) — saves the vision tower's RAM for
pure-text sessions but pays a full model reload, and with it the warm RAM
prefix cache, at the worst possible moment: deep into a long session, the
instant the first screenshot is pasted. *Honoring the toggle on the HTTP path*
— keeps generated configs honest only at generation time and reintroduces
silent image-dropping whenever the toggle flips.

Consequence for the shared LLM slot: chat with the toggle off "wants" the text
variant while HTTP wants vision. To avoid reload thrash between callers, the
vision variant satisfies text-only requests — load-state matching upgrades but
never downgrades. Corollary: flipping the chat toggle off no longer frees the
vision tower immediately; the memory returns at the next natural unload.
