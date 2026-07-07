import Foundation

// MARK: - PageMapElement

/// One salient element in a **Page Map** — the interaction representation of a
/// page. Decoded from the JSON the injected map script returns. Interactive
/// elements carry a `ref` (a stable integer written onto the DOM node as a
/// `data-tesseract-ref` attribute) that `click`/`type` later resolve; purely
/// structural rows (headings) carry no ref.
nonisolated struct PageMapElement: Codable, Sendable, Equatable {
    /// Stable interaction handle, or nil for structural rows (headings).
    let ref: Int?
    /// Coarse role: `link`, `button`, `textbox`, `checkbox`, `radio`,
    /// `combobox`, `heading`, … — derived from tag + ARIA role.
    let role: String
    /// Accessible name (label / text / placeholder / alt / title / value).
    let name: String
    /// Current value for form fields (may duplicate `name` for some inputs).
    let value: String?
    /// Placeholder text for inputs, surfaced as a hint.
    let placeholder: String?
    /// Heading level (1–6) when `role == "heading"`.
    let level: Int?
    /// Whether the control is disabled.
    let disabled: Bool?
    /// Whether a checkbox/radio is checked.
    let checked: Bool?
}

// MARK: - PageMapScript

/// The JavaScript that builds a **Page Map** in the live DOM.
///
/// Runs in an isolated `WKContentWorld` (shares the DOM, not the page's JS
/// globals). It assigns each interactive element a sequential `ref`, writes it
/// as a `data-tesseract-ref` attribute so a later `click(ref:)` can resolve the
/// same node, and returns a compact JSON array. Structural headings are
/// included (ref-less) so the agent has document scaffolding. Hidden and
/// zero-size elements are skipped.
///
/// The script caps its own output at `hardElementCap` to keep the round-trip
/// bounded; `PageMapFormatter` applies the display cap on top.
nonisolated enum PageMapScript {

    /// Upper bound on elements the script will emit, independent of the display
    /// cap — a guard against pathological pages, not the token budget.
    static let hardElementCap = 800

    static let source = #"""
        const HARD_CAP = 800;
        const NAME_CAP = 400;

        // Visibility is judged from computed style + attributes only, never
        // geometry: getBoundingClientRect/getClientRects depend on layout, which a
        // page that isn't attached to a sized view has not performed. Style-based
        // checks (display/visibility) come from the cascade and are reliable
        // whether or not the page has laid out.
        function isVisible(el) {
          if (!el || el.nodeType !== 1) return false;
          if (el.getAttribute('aria-hidden') === 'true') return false;
          if (el.hasAttribute('hidden')) return false;
          const style = window.getComputedStyle(el);
          if (!style) return true;
          if (style.visibility === 'hidden' || style.display === 'none') return false;
          return true;
        }

        function clip(s) {
          if (!s) return '';
          s = String(s).replace(/\s+/g, ' ').trim();
          return s.length > NAME_CAP ? s.slice(0, NAME_CAP) + '…' : s;
        }

        function labelFor(el) {
          const aria = el.getAttribute('aria-label');
          if (aria) return clip(aria);
          const labelledby = el.getAttribute('aria-labelledby');
          if (labelledby) {
            const parts = labelledby.split(/\s+/).map(id => {
              const n = document.getElementById(id);
              return n ? n.textContent : '';
            });
            const joined = clip(parts.join(' '));
            if (joined) return joined;
          }
          if (el.id) {
            const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            if (lab && lab.textContent) return clip(lab.textContent);
          }
          const closestLabel = el.closest('label');
          if (closestLabel && closestLabel.textContent) return clip(closestLabel.textContent);
          const alt = el.getAttribute('alt');
          if (alt) return clip(alt);
          const title = el.getAttribute('title');
          if (title) return clip(title);
          const placeholder = el.getAttribute('placeholder');
          if (placeholder) return clip(placeholder);
          if (el.value && (el.tagName === 'INPUT' || el.tagName === 'BUTTON')) return clip(el.value);
          return clip(el.textContent);
        }

        function roleFor(el) {
          const explicit = el.getAttribute('role');
          if (explicit) return explicit;
          const tag = el.tagName.toLowerCase();
          if (tag === 'a') return 'link';
          if (tag === 'button') return 'button';
          if (tag === 'select') return 'combobox';
          if (tag === 'textarea') return 'textbox';
          if (tag === 'input') {
            const t = (el.getAttribute('type') || 'text').toLowerCase();
            if (t === 'checkbox') return 'checkbox';
            if (t === 'radio') return 'radio';
            if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
            if (t === 'range') return 'slider';
            return 'textbox';
          }
          if (el.isContentEditable) return 'textbox';
          return 'button';
        }

        const INTERACTIVE = [
          'a[href]', 'button', 'input:not([type=hidden])', 'textarea', 'select',
          '[role=button]', '[role=link]', '[role=textbox]', '[role=checkbox]',
          '[role=radio]', '[role=tab]', '[role=menuitem]', '[role=combobox]',
          '[contenteditable=""]', '[contenteditable=true]', '[onclick]', '[tabindex]'
        ].join(',');

        // Merge interactive + heading elements in document order.
        const wanted = document.querySelectorAll(INTERACTIVE + ',h1,h2,h3,h4,h5,h6');
        const out = [];
        let ref = 0;
        const seen = new WeakSet();

        for (const el of wanted) {
          if (seen.has(el)) continue;
          seen.add(el);
          if (out.length >= HARD_CAP) break;
          if (!isVisible(el)) continue;

          const tag = el.tagName.toLowerCase();
          if (/^h[1-6]$/.test(tag)) {
            const name = clip(el.textContent);
            if (name) out.push({ role: 'heading', name: name, level: parseInt(tag[1], 10) });
            continue;
          }

          const name = labelFor(el);
          // Drop nameless, actionless noise (e.g. bare tabindex wrappers).
          if (!name && !el.getAttribute('href') && tag !== 'input') continue;

          ref += 1;
          el.setAttribute('data-tesseract-ref', String(ref));

          const item = { ref: ref, role: roleFor(el), name: name };
          const type = el.getAttribute('type');
          if (el.disabled === true) item.disabled = true;
          if (el.checked === true) item.checked = true;
          const ph = el.getAttribute('placeholder');
          if (ph) item.placeholder = clip(ph);
          if (el.value && (tag === 'input' || tag === 'select' || tag === 'textarea')) {
            const v = clip(el.value);
            if (v && v !== name) item.value = v;
          }
          out.push(item);
        }

        return JSON.stringify(out);
        """#
}
