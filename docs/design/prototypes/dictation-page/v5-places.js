/* PROTOTYPE, THROWAWAY. Variant 5, Places.
   Your words belong to where you use them. Each place (an app, or the project folder a terminal is in) listens
   for its own words: read from the project's files and learned from your fixes there. So "cloud" becomes Claude
   in Terminal · tesseract and stays "cloud" in Notes. The overlay always says which place is listening.
   Fix: ⌃⌥Space puts a letter over each word of the last take; press the letter, then the number of the right word. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'places', num: 5, name: 'Places',
  thesis: 'Your words belong to where you use them. Each project listens for its own names; a fix in one place never leaks into another.',
  steps: [
    { id: 'dictate', text: 'Hold <b>`</b> and dictate into Terminal. Tesseract is right already: it’s in the project' },
    { id: 'fix', text: 'Press <b>Tab</b>, the letter over “cloud”, then <b>1</b>. Claude is learned for this project' },
    { id: 'scoped', text: 'Click Notes and dictate. “cloud” stays cloud there' },
    { id: 'move', text: 'On the page, drag a word to Everywhere to use it in every app' },
  ],
};

V.css = `
:root{--pl-term:#3d3d44;--pl-on:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--pl-term:#a4a4ad;--pl-on:#1b1b1d}}
:root[data-theme="dark"]{--pl-term:#a4a4ad;--pl-on:#1b1b1d}
.pl{padding:6px 26px 120px;display:flex;flex-direction:column;gap:18px;max-width:1180px;margin:0 auto}
.pl-here{display:flex;align-items:center;gap:12px;padding:4px 2px}
.pl-here .pin{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:var(--here-c,#3a3a3f);color:#fff;box-shadow:0 4px 12px color-mix(in srgb,var(--here-c,#000) 35%,transparent);transition:background .35s}
.pl-here .txt{font:600 20px/1.2 var(--display);letter-spacing:-.015em}
.pl-here .txt span{color:var(--ink3);font-weight:500}
.pl-here .txt code{font:500 15px var(--mono);color:var(--ink2);letter-spacing:0}
.pl-here .n{margin-left:auto;font:500 13px var(--sans);color:var(--ink2);display:flex;align-items:center;gap:8px}
.pl-here .n i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent)}
.pl-board{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;align-items:start}
.pl-col{position:relative;border-radius:16px;background:var(--fill);box-shadow:inset 0 0 0 1px var(--line);padding:0 12px 12px;min-height:300px;transition:box-shadow .3s,background .3s,transform .3s}
.pl-col.here{background:var(--win-bg);box-shadow:inset 0 0 0 2px var(--c),0 10px 26px color-mix(in srgb,var(--c) 18%,transparent);transform:translateY(-2px)}
.pl-col.drop{box-shadow:inset 0 0 0 2px var(--accent);background:color-mix(in srgb,var(--accent) 8%,var(--fill))}
.pl-col::before{content:"";position:absolute;left:14px;right:14px;top:0;height:3px;border-radius:0 0 3px 3px;background:var(--c)}
.pl-ch{display:flex;align-items:center;gap:9px;padding:16px 2px 10px}
.pl-ch .ico{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;background:var(--c);color:#fff;flex:none}
.pl-col[data-place="terminal"] .pl-ch .ico,.pl-here .pin.term{color:var(--pl-on)}
.pl-ch b{font:600 14px/1.15 var(--display);display:block}
.pl-ch small{font:11.5px/1.2 var(--mono);color:var(--ink3);display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pl-youare{position:absolute;right:10px;top:15px;font:700 10px/1 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--c);display:flex;align-items:center;gap:4px}
.pl-sec{font:600 11px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);margin:12px 2px 7px;display:flex;justify-content:space-between}
.pl-tags{display:flex;flex-wrap:wrap;gap:5px}
.pl-tag{display:inline-flex;flex-direction:column;gap:2px;padding:6px 9px;border-radius:10px;background:var(--win-bg);box-shadow:0 0 0 1px var(--line),0 1px 2px rgba(0,0,0,.05);cursor:grab;max-width:100%}
.pl-col.here .pl-tag{background:var(--fill)}
.pl-tag b{font:600 13px/1.1 var(--sans);display:flex;align-items:center;gap:5px;white-space:nowrap}
.pl-tag b .ic{color:var(--ink3)}
.pl-tag small{font:11px/1.1 var(--mono);color:var(--ink3);text-decoration:line-through;text-decoration-color:color-mix(in srgb,var(--danger) 55%,transparent);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pl-tag.proj{padding:5px 8px;box-shadow:none;background:transparent;border:1px dashed var(--line);cursor:default}
.pl-tag.proj b{font-weight:500;color:var(--ink2)}
.pl-tag.new{animation:plnew 1.4s ease}
@keyframes plnew{0%{box-shadow:0 0 0 2px var(--accent);transform:scale(1.06)}100%{box-shadow:0 0 0 1px var(--line)}}
.pl-tag.dragging{opacity:.4}
.pl-more{font:12px var(--sans);color:var(--ink3);padding:5px 4px}
.pl-empty{font:12.5px/1.45 var(--sans);color:var(--ink3);padding:4px 2px}
.pl-last{margin-top:12px;border-top:1px solid var(--line);padding-top:9px;font:12.5px/1.45 var(--sans);color:var(--ink2)}
.pl-last .seg.auto,.pl-last .seg.fixed{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 70%,transparent)}
.pl-last .seg.proj{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--c) 70%,transparent)}
.pl-foot{font:13px/1.5 var(--sans);color:var(--ink3);display:flex;gap:8px;align-items:center}
.pl-toggle{display:inline-flex;align-items:center;gap:7px;font:12px var(--sans);color:var(--ink2);cursor:pointer}
.pl-toggle i{width:26px;height:15px;border-radius:8px;background:var(--ok);position:relative}
.pl-toggle i::after{content:"";position:absolute;right:2px;top:2px;width:11px;height:11px;border-radius:50%;background:#fff}
.pl-toggle.off i{background:var(--fill2)}.pl-toggle.off i::after{right:auto;left:2px}

/* overlay */
.pl-pill{position:absolute;left:50%;bottom:40px;transform:translateX(-50%) scale(.85);opacity:0;height:40px;padding:0 6px 0 7px;border-radius:20px;display:flex;align-items:center;gap:10px;font:500 13px/1 var(--sans);transition:opacity .18s,transform .22s cubic-bezier(.2,1.2,.4,1);pointer-events:none;white-space:nowrap}
.pl-pill.show{opacity:1;transform:translateX(-50%) scale(1)}
.pl-pill .place{display:inline-flex;align-items:center;gap:7px;height:28px;padding:0 11px 0 5px;border-radius:14px;background:var(--pc,#3a3a3f);color:#fff;font:600 12.5px/1 var(--sans)}
.pl-pill .place.term,.pl-scope.term{color:var(--pl-on)}
.pl-pill .place .ic{width:20px;height:20px;border-radius:7px;background:rgba(255,255,255,.18);padding:3px}
.pl-pill .bars{display:flex;gap:3px;align-items:center;height:16px}
.pl-pill .bars span{width:3px;height:4px;border-radius:2px;background:var(--danger)}
.pl-pill .msg{padding-right:10px;color:var(--glass-ink)}
.pl-pill .msg b{font-weight:600}
.pl-pill .dim{color:var(--glass-ink2);padding-right:10px}
.pl-hints{position:absolute;left:50%;bottom:34px;transform:translateX(-50%);max-width:640px;min-width:420px;border-radius:22px;padding:12px 18px 12px;display:flex;flex-direction:column;gap:10px;animation:plin .22s cubic-bezier(.2,1.2,.4,1)}
@keyframes plin{from{opacity:0;transform:translateX(-50%) translateY(10px) scale(.96)}}
.pl-hh{display:flex;align-items:center;gap:8px;font:600 12px/1 var(--sans);color:var(--glass-ink2)}
.pl-hh .dot{width:8px;height:8px;border-radius:3px;background:var(--pc)}
.pl-hh .r{margin-left:auto;font-weight:500}
.pl-line{font:500 18px/2.3 var(--sans);color:var(--glass-ink)}
.pl-w{position:relative;display:inline-block;border-radius:6px;padding:0 2px;margin:0 -2px}
.pl-w kbd{position:absolute;left:50%;top:-6px;transform:translateX(-50%);min-width:17px;height:17px;padding:0 4px;border-radius:5px;background:var(--accent);color:var(--accent-ink);font:700 11px/17px var(--mono);text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.2)}
.pl-hints.picked .pl-w kbd{display:none}
.pl-w.sel{background:color-mix(in srgb,var(--accent) 22%,transparent);box-shadow:0 0 0 1.5px var(--accent)}
.pl-hints.picked .pl-w:not(.sel){opacity:.45}
.pl-cands{display:flex;gap:6px;flex-wrap:wrap}
.pl-cands button{height:34px;border-radius:17px;border:0;padding:0 12px 0 6px;display:inline-flex;align-items:center;gap:8px;font:600 14px/1 var(--sans);cursor:pointer;background:color-mix(in srgb,var(--glass-ink) 8%,transparent);color:var(--glass-ink)}
.pl-cands button:first-child{background:var(--accent);color:var(--accent-ink)}
.pl-cands kbd{min-width:22px;height:22px;border-radius:11px;display:grid;place-items:center;background:rgba(255,255,255,.25);font:700 11.5px var(--mono)}
.pl-cands button:not(:first-child) kbd{background:color-mix(in srgb,var(--glass-ink) 10%,transparent)}
.pl-cands small{font:500 11px var(--mono);opacity:.7}
.pl-type{display:flex;align-items:center;gap:8px;height:38px;border-radius:12px;padding:0 12px;background:color-mix(in srgb,var(--glass-ink) 7%,transparent)}
.pl-type input{flex:1;border:0;background:transparent;outline:none;font:600 17px var(--display);color:var(--glass-ink)}
.pl-hf{display:flex;gap:12px;align-items:center;font:12px/1 var(--sans);color:var(--glass-ink2)}
.pl-hf .kc{font-size:10.5px;height:17px}
.pl-hf button{margin-left:auto;border:0;background:transparent;color:var(--accent-text);font:600 12.5px var(--sans);cursor:pointer}
.pl-res{display:flex;align-items:center;gap:10px;font:600 15px/1 var(--sans)}
.pl-res .ok{color:var(--ok)}
.pl-res .h{font:500 13px var(--mono);color:var(--glass-ink2);text-decoration:line-through;text-decoration-color:var(--danger)}
.pl-res b{color:var(--accent-text)}
.pl-scope{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 9px 0 5px;border-radius:12px;background:var(--pc);color:#fff;font:600 11.5px/1 var(--sans)}
`;

const PLACES = [
  { id: 'terminal', name: 'Terminal', sub: '~/projects/tesseract', icon: 'terminal', c: 'var(--pl-term)', words: 48 },
  { id: 'notes', name: 'Notes', sub: 'iCloud', icon: 'notes', c: '#d9a200', words: 3 },
  { id: 'safari', name: 'Safari', sub: 'any site', icon: 'compass', c: '#2A78D6', words: 4 },
  { id: 'everywhere', name: 'Everywhere', sub: 'every app', icon: 'globe', c: '#8a6ad8', words: 2 },
];
const PROJECT = [
  { w: 'Tesseract', src: 'README' }, { w: 'DFlash2', src: 'branch' }, { w: 'CLAUDE.md', src: 'file' },
  { w: 'WhisperKit', src: 'Package.swift' }, { w: 'Qwen3-TTS', src: 'model' }, { w: 'worktree', src: 'git' }, { w: 'MLX', src: 'Package.swift' },
];
const placeOf = (id) => PLACES.find((p) => p.id === id);
const LETTERS = 'asdfghjklqwrtyuiopzxcvbnm';

V.mount = (ctx) => {
  S = { ctx, project: true, hint: null, pillTimer: 0, lastBy: {}, newTag: null };
  const seed = (heard, meant, scope, ago) => { const r = P.learn({ heard, meant, scope, source: 'fix' }); r.at = new Date(Date.now() - ago * 3600e3); r.applied = Math.round(ago / 9) + 1; };
  seed('Eleven Labs', 'ElevenLabs', 'everywhere', 50);
  seed('test flight', 'TestFlight', 'everywhere', 30);
  seed('QWEN 3 TTS', 'Qwen3-TTS', 'safari', 20);
  seed('whisper kit', 'WhisperKit', 'safari', 70);
  seed('bread flower', 'bread flour', 'notes', 90);
  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>`;
  ctx.overlay.innerHTML = `<div class="pl-pill glass" id="pl-pill"></div>`;
  S.unFrame = P.onFrame((lv) => {
    ctx.overlay.querySelectorAll('.pl-pill .bars span').forEach((b, i) => { b.style.height = `${4 + lv * 12 * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7 + performance.now() / 170)))}px`; });
  });
  render();
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.pillTimer); S = null; };

/* A rule applies in its own place, or everywhere. */
V.findRule = (heard, take) => P.st.rules.find((r) => r.on && P.eqi(r.heard, heard) && (r.scope === 'everywhere' || r.scope === take?.app));
/* Project words bias the recognizer: in Terminal · tesseract they come out right in the first place. */
V.transform = (take) => {
  take.segs.forEach((s, i) => {
    if (s.h == null || s.status !== 'wrong') return;
    if (take.app === 'terminal' && S.project && PROJECT.some((p) => P.eqi(p.w, s.m))) { s.text = P.fitCase(s.m, i === 0); s.status = 'right'; s.proj = true; }
  });
  take.segs.forEach((s) => {
    if (s.h == null || s.status === 'auto') return;
    const elsewhere = P.st.rules.find((r) => r.on && P.eqi(r.heard, s.h) && r.scope !== take.app && r.scope !== 'everywhere');
    if (elsewhere && s.text === s.h) s.leftAlone = elsewhere;
  });
};

/* ---------- page ---------- */
const render = () => {
  const here = placeOf(P.st.target);
  const n = here.id === 'terminal' ? here.words + P.st.rules.filter((r) => r.on && (r.scope === 'terminal' || r.scope === 'everywhere')).length - 2 : P.st.rules.filter((r) => r.on && (r.scope === here.id || r.scope === 'everywhere')).length;
  S.ctx.page.innerHTML = `<div class="pl" style="--here-c:${here.c}">
    <div class="pl-here"><span class="pin${here.id === 'terminal' ? ' term' : ''}">${P.icon(here.icon, 18, 1.9)}</span>
      <div class="txt"><span>You’re dictating into</span> ${here.name}${here.id === 'terminal' ? ` <code>${here.sub}</code>` : ''}</div>
      <div class="n"><i></i>${n} words listening here</div></div>
    <div class="pl-board">${PLACES.map(colHTML).join('')}</div>
    <div class="pl-foot">${P.icon('pin', 14)}A word you fix is learned for the place you fixed it in. Drag it to another place, or to Everywhere.</div>
  </div>`;
  wireDnD();
  S.newTag = null;
  S.ctx.page.querySelector('.pl-toggle')?.addEventListener('click', () => { S.project = !S.project; render(); });
};
const colHTML = (pl) => {
  const here = P.st.target === pl.id;
  const rules = P.st.rules.filter((r) => r.on && r.scope === pl.id);
  const tag = (r) => `<span class="pl-tag${r === S.newTag ? ' new' : ''}" draggable="true" data-r="${r.id}"><b>${P.icon(pl.id === 'everywhere' ? 'globe' : 'check', 11, 2.2)}${P.esc(r.meant)}</b><small>${P.esc(r.heard)}</small></span>`;
  const last = P.st.takes.find((t) => t.app === pl.id);
  const lastHTML = last ? `<div class="pl-last">${last.segs.map((s, i) => s.h == null ? P.esc(s.text) : P.segHTML(last, s, i).replace('class="seg', `class="seg${s.proj ? ' proj' : s.status === 'auto' || s.status === 'fixed' ? ' auto' : ''}`)).join('')}</div>` : '';
  return `<section class="pl-col${here ? ' here' : ''}" data-place="${pl.id}" style="--c:${pl.c}">
    ${here ? `<span class="pl-youare">${P.icon('pin', 11, 2.4)}Here</span>` : ''}
    <div class="pl-ch"><span class="ico">${P.icon(pl.icon, 15, 2)}</span><span style="min-width:0"><b>${pl.name}</b><small>${pl.sub}</small></span></div>
    <div class="pl-sec"><span>${pl.id === 'everywhere' ? 'In every app' : 'Your fixes'}</span></div>
    <div class="pl-tags">${rules.map(tag).join('') || `<span class="pl-empty">${pl.id === 'everywhere' ? 'Drag a word here to use it in every app.' : 'None yet.'}</span>`}</div>
    ${pl.id === 'terminal' ? `<div class="pl-sec"><span>From the project</span><span class="pl-toggle${S.project ? '' : ' off'}"><i></i></span></div>
      <div class="pl-tags">${PROJECT.slice(0, 6).map((p) => `<span class="pl-tag proj" title="From ${P.esc(p.src)}"><b>${P.icon(p.src === 'branch' ? 'hash' : p.src === 'git' ? 'folder' : 'doc', 11, 2)}${P.esc(p.w)}</b></span>`).join('')}<span class="pl-more">+${pl.words - 6} from file names, the README and branches</span></div>` : ''}
    ${lastHTML}
  </section>`;
};
const wireDnD = () => {
  const page = S.ctx.page;
  page.querySelectorAll('.pl-tag[draggable]').forEach((t) => {
    t.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', t.dataset.r); t.classList.add('dragging'); });
    t.addEventListener('dragend', () => t.classList.remove('dragging'));
    t.addEventListener('dblclick', () => moveRule(+t.dataset.r, 'everywhere'));
  });
  page.querySelectorAll('.pl-col').forEach((c) => {
    c.addEventListener('dragover', (e) => { e.preventDefault(); c.classList.add('drop'); });
    c.addEventListener('dragleave', () => c.classList.remove('drop'));
    c.addEventListener('drop', (e) => { e.preventDefault(); c.classList.remove('drop'); moveRule(+e.dataTransfer.getData('text/plain'), c.dataset.place); });
  });
};
const moveRule = (id, scope) => {
  const r = P.st.rules.find((x) => x.id === id); if (!r || r.scope === scope) return;
  r.scope = scope; S.newTag = r; P.step('move'); render();
};

/* ---------- overlay: which place is listening, and hint letters for fixing ---------- */
const pill = () => S.ctx.overlay.querySelector('#pl-pill');
const placeChip = (id) => { const pl = placeOf(id); return `<span class="place${id === 'terminal' ? ' term' : ''}" style="--pc:${pl.c}">${P.icon(pl.icon, 14, 2)}${pl.name}${id === 'terminal' ? ' · tesseract' : ''}</span>`; };

V.onEvent = (type, d) => {
  if (!S) return;
  const p = pill();
  if (type === 'focus') render();
  if (type === 'rec-start') {
    closeHints(); clearTimeout(S.pillTimer);
    p.style.setProperty('--pc', placeOf(d.take.app).c);
    p.innerHTML = `${placeChip(d.take.app)}<span class="bars">${'<span></span>'.repeat(6)}</span><span class="dim">${d.take.app === 'terminal' ? '48 words' : 'listening'}</span>`;
    p.classList.add('show');
  }
  if (type === 'processing') p.querySelector('.bars')?.replaceWith(P.el(`<span class="dim">…</span>`));
  if (type === 'inserted') {
    const t = d.take; P.step('dictate');
    const proj = t.segs.filter((s) => s.proj), auto = t.segs.filter((s) => s.status === 'auto'), alone = t.segs.filter((s) => s.leftAlone);
    let msg = '';
    if (alone.length) { msg = `Left “${P.esc(alone[0].text)}” as you said it. <b>${P.esc(alone[0].leftAlone.meant)}</b> is a ${placeOf(alone[0].leftAlone.scope).name}${alone[0].leftAlone.scope === 'terminal' ? ' · tesseract' : ''} word.`; P.step('scoped'); }
    else if (proj.length || auto.length) msg = `Wrote ${[...proj, ...auto].map((s) => `<b>${P.esc(s.text)}</b>`).join(', ')} from ${t.app === 'terminal' ? 'tesseract' : 'your words'}`;
    else msg = `<span style="color:var(--glass-ink2)">Wrong word? <span class="kc">⌃⌥Space</span></span>`;
    p.innerHTML = `${placeChip(t.app)}<span class="msg">${msg}</span>`;
    S.pillTimer = setTimeout(() => p.classList.remove('show'), alone.length ? 4200 : 3000);
    render();
  }
};

const closeHints = () => { S.hint = null; S.ctx.overlay.querySelector('.pl-hints')?.remove(); };
V.onFixKey = () => {
  if (S.hint) { closeHints(); return; }
  const take = P.lastLive() || P.st.takes.find((t) => t.inserted && !t.seed);
  if (!take) return;
  pill().classList.remove('show');
  S.hint = { take, toks: P.tokens(take), sel: -1, typing: false, done: null };
  drawHints();
};
const candidates = (take, tok) => {
  const place = take.app;
  const pool = [...new Set([
    ...(tok.term && !P.eqi(take.segs[tok.seg].m, tok.text) ? [take.segs[tok.seg].m] : []),
    ...(place === 'terminal' ? PROJECT.map((p) => p.w) : []),
    ...P.st.rules.filter((r) => r.on && (r.scope === place || r.scope === 'everywhere')).map((r) => r.meant),
    ...P.WORDS.map((w) => w.w),
  ])].filter((w) => !P.eqi(w, tok.text));
  const first = tok.text[0].toLowerCase();
  const head = pool.slice(0, tok.term && !P.eqi(take.segs[tok.seg].m, tok.text) ? 1 : 0);
  const rest = pool.slice(head.length).filter((w) => w[0].toLowerCase() === first || (first === 'c' && w.startsWith('CL')));
  return [...head, ...rest].slice(0, 3);
};
const drawHints = () => {
  const H = S.hint; const take = H.take; const pl = placeOf(take.app);
  let el = S.ctx.overlay.querySelector('.pl-hints');
  if (!el) { el = P.el(`<div class="pl-hints glass"></div>`); S.ctx.overlay.appendChild(el); }
  el.style.setProperty('--pc', pl.c);
  el.classList.toggle('picked', H.sel >= 0);
  if (H.done) {
    const r = H.done.rule;
    el.innerHTML = `<div class="pl-res"><span class="ok">${P.icon('check', 17, 2.4)}</span><span class="h">${P.esc(H.done.heard)}</span>${P.icon('arrow', 14, 2)}<b>${P.esc(r.meant)}</b>
      <span class="pl-scope${r.scope === 'terminal' ? ' term' : ''}" style="--pc:${placeOf(r.scope).c}">${P.icon(placeOf(r.scope).icon, 12, 2.2)}${r.scope === 'everywhere' ? 'Everywhere' : 'Only in ' + placeOf(r.scope).name + (r.scope === 'terminal' ? ' · tesseract' : '')}</span></div>
      <div class="pl-hf">${r.scope === 'everywhere' ? '<span>Used in every app now.</span>' : '<span><span class="kc">E</span> use it everywhere</span>'}<span><span class="kc">esc</span> close</span><button id="pl-undo">Undo</button></div>`;
    el.querySelector('#pl-undo').onclick = () => { P.unlearn(r); P.fixSeg(take, H.done.seg, H.done.heard, 'wrong'); closeHints(); render(); };
    return;
  }
  const line = P.joinTokens(H.toks, (t, i) => `<span class="pl-w${i === H.sel ? ' sel' : ''}" data-i="${i}">${H.sel < 0 && i < LETTERS.length ? `<kbd>${LETTERS[i]}</kbd>` : ''}${P.esc(t.text)}</span>`);
  let lower = `<div class="pl-hf"><span>Press the letter over the wrong word</span><span><span class="kc">esc</span> close</span></div>`;
  if (H.sel >= 0) {
    const c = candidates(take, H.toks[H.sel]); H.cands = c;
    lower = H.typing
      ? `<div class="pl-type">${P.icon('keyboard', 16)}<input id="pl-ti" placeholder="Type the right word" autocomplete="off" spellcheck="false"></div><div class="pl-hf"><span><span class="kc">↩</span> fix</span><span><span class="kc">esc</span> back</span></div>`
      : `<div class="pl-cands">${c.map((w, k) => `<button data-c="${k}"><kbd>${k + 1}</kbd>${P.esc(w)}</button>`).join('')}<button data-c="type"><kbd>${c.length + 1}</kbd>Type it…</button></div>
         <div class="pl-hf"><span>Words ${pl.id === 'terminal' ? 'from tesseract' : 'you use in ' + pl.name} that sound close</span><span><span class="kc">esc</span> back</span></div>`;
  }
  el.innerHTML = `<div class="pl-hh"><span class="dot"></span>Fix a word in ${pl.name}${pl.id === 'terminal' ? ' · tesseract' : ''}<span class="r">${P.ago(take.at)}</span></div><div class="pl-line">${line}</div>${lower}`;
  el.onclick = (e) => {
    const w = e.target.closest('.pl-w'); if (w && H.sel < 0) { H.sel = +w.dataset.i; drawHints(); return; }
    const b = e.target.closest('[data-c]'); if (b) pick(b.dataset.c === 'type' ? 'type' : +b.dataset.c);
  };
  const ti = el.querySelector('#pl-ti');
  if (ti) {
    ti.onkeydown = (e) => {
      if (['Enter', 'Escape'].includes(e.key)) { e.preventDefault(); e.stopPropagation(); }
      if (e.key === 'Escape') { H.typing = false; drawHints(); }
      if (e.key === 'Enter' && ti.value.trim()) apply(ti.value.trim());
    };
    setTimeout(() => ti.focus(), 20);
  }
};
const pick = (k) => {
  const H = S.hint;
  if (k === 'type' || k === H.cands.length) { H.typing = true; drawHints(); return; }
  if (H.cands[k]) apply(H.cands[k]);
};
const apply = (word) => {
  const H = S.hint; const take = H.take; const tok = H.toks[H.sel];
  const seg = take.segs[tok.seg];
  const heard = tok.term ? seg.h : tok.text.replace(/[.,!?]$/, '');
  const rule = P.learn({ heard, meant: word, scope: take.app, source: 'fix' });
  if (tok.term) P.fixSeg(take, tok.seg, word);
  else { seg.text = seg.text.replace(tok.text, word); P.renderTargets(); }
  S.newTag = rule;
  H.done = { rule, heard, seg: tok.seg };
  P.step('fix');
  drawHints(); render();
  clearTimeout(S.hintTimer);
  S.hintTimer = setTimeout(() => { if (S?.hint === H) closeHints(); }, 6000);
};

V.onKey = (e, k) => {
  if (!S || !S.hint) return false;
  if (k.inField) return false;
  const H = S.hint;
  if (e.key === 'Escape') { if (H.typing) H.typing = false; else if (H.sel >= 0 && !H.done) H.sel = -1; else { closeHints(); return true; } drawHints(); return true; }
  if (H.done) {
    if (e.key.toLowerCase() === 'e' && H.done.rule.scope !== 'everywhere') { H.done.rule.scope = 'everywhere'; S.newTag = H.done.rule; P.step('move'); drawHints(); render(); return true; }
    if (k.fixKey) { closeHints(); return true; }
    return false;
  }
  if (H.sel < 0) {
    const i = LETTERS.indexOf(e.key.toLowerCase());
    if (i >= 0 && i < H.toks.length) { H.sel = i; drawHints(); return true; }
    return !k.talkKey;
  }
  if (/^[1-9]$/.test(e.key)) { pick(+e.key - 1); return true; }
  return !k.talkKey;
};

P.register(V);
})();
