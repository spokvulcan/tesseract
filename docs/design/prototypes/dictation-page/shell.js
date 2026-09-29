/* PROTOTYPE, THROWAWAY. Dictation page redesign, round 2 (HTML).
   The shared shell: a fake Mac (Terminal, Notes, Tesseract's Dictation window, the overlay layer),
   a scripted dictation engine that mishears the owner's real terms the way Whisper does
   (docs/research/2026-09-28-dictation-errors-and-learning.md), and the variant switcher.
   Variants register with Proto.register({...}) and own the page, the overlay and the fix flow. */
(() => {
'use strict';
const P = (window.Proto = {});
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
P.$ = $; P.$$ = $$;
P.esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
P.el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
P.wait = (ms) => new Promise((r) => setTimeout(r, ms));
P.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const eqi = (a, b) => String(a).toLowerCase() === String(b).toLowerCase();
P.eqi = eqi;
let uid = 0;
P.uid = () => ++uid;

/* ---------- icons (SF Symbol stand-ins) ---------- */
const ICONS = {
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  wave: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  play: '<path d="M8 5.5v13l10-6.5z" fill="currentColor" stroke="none"/>',
  search: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
  terminal: '<rect x="3" y="4.5" width="18" height="15" rx="3"/><path d="M7 9.5l3 2.5-3 2.5M12.5 15h4"/>',
  notes: '<rect x="4.5" y="3.5" width="15" height="17" rx="2.5"/><path d="M8 8.5h8M8 12h8M8 15.5h5"/>',
  compass: '<circle cx="12" cy="12" r="8.5"/><path d="M15.5 8.5l-2 5-5 2 2-5z"/>',
  folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  doc: '<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4"/>',
  brain: '<path d="M9 4.5a3 3 0 0 0-3 3 3 3 0 0 0-1.5 5.3A3 3 0 0 0 7 18a2.5 2.5 0 0 0 5 .5V6.5a2 2 0 0 0-3-2zM15 4.5a3 3 0 0 1 3 3 3 3 0 0 1 1.5 5.3A3 3 0 0 1 17 18a2.5 2.5 0 0 1-5 .5"/>',
  bubble: '<path d="M4.5 6.5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H10l-4.5 4v-4a3 3 0 0 1-1-2.2z"/>',
  person: '<circle cx="12" cy="8" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/>',
  pin: '<path d="M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0 1 13 0C18.5 14.8 12 21 12 21z"/><circle cx="12" cy="10" r="2.3"/>',
  globe: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.5 2.6 2.5 14.4 0 17M12 3.5c-2.5 2.6-2.5 14.4 0 17"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6 10h.01M9.5 10h.01M13 10h.01M16.5 10h.01M7 14h10"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  chevron: '<path d="M9 5l7 7-7 7"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  ear: '<path d="M7 9a5 5 0 0 1 10 0c0 3-3 3.5-3 6.5a2.5 2.5 0 0 1-5 0"/><path d="M10 9.5a2 2 0 0 1 4 0"/>',
  bolt: '<path d="M13 3L5 13.5h6L10 21l8-10.5h-6z"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
  trash: '<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  chart: '<path d="M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-3"/>',
  hash: '<path d="M9 4L7 20M17 4l-2 16M4.5 9h16M3.5 15h16"/>',
};
P.icon = (n, s = 16, w = 1.7) => `<svg class="ic" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n] || ''}</svg>`;
P.tesseractGlyph = (s = 15) => `<svg width="${s}" height="${s}" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="2" y="2" width="11" height="11" rx="1.5"/><rect x="7" y="7" width="11" height="11" rx="1.5"/><path d="M2.6 2.6L7.4 7.4M12.6 2.6l4.8 4.8M2.6 12.6l4.8 4.8M12.6 12.6l4.8 4.8"/></svg>`;

/* ---------- the owner's words, and what Whisper hears (from the 2026-09-28 replay) ---------- */
const T = (h, m) => ({ h, m });
P.T = T;
P.WORDS = [
  { w: 'Claude', heard: 'cloud', kind: 'Name', from: 'memory', why: 'you talk to it in Terminal every day', uses: 212 },
  { w: 'Tesseract', heard: 'SRACT', alts: ['SRAX', 'TSRAC'], kind: 'Project', from: 'project', why: 'your project in ~/projects/tesseract', uses: 148 },
  { w: 'CLAUDE.md', heard: 'cloud.md', kind: 'File', from: 'project', why: 'a file in tesseract', uses: 31 },
  { w: 'DFlash2', heard: 'D flash two', kind: 'Branch', from: 'project', why: 'a branch in tesseract', uses: 44 },
  { w: 'worktree', heard: 'work tree', kind: 'Term', from: 'memory', why: 'a git term you use', uses: 27 },
  { w: 'Qwen3-TTS', heard: 'QWEN 3 TTS', kind: 'Model', from: 'project', why: 'the voice model in tesseract', uses: 19 },
  { w: 'WhisperKit', heard: 'whisper kit', kind: 'Library', from: 'project', why: 'a package in tesseract', uses: 23 },
  { w: 'TestFlight', heard: 'test flight', kind: 'Product', from: 'memory', why: 'you ship builds there', uses: 9 },
  { w: 'ElevenLabs', heard: 'Eleven Labs', kind: 'Company', from: 'memory', why: 'you compared its voices', uses: 6 },
  { w: 'Wispr Flow', heard: 'whisper flow', kind: 'Product', from: 'memory', why: 'you researched it last week', uses: 5 },
  { w: 'MLX', heard: 'MLX', right: true, kind: 'Library', from: 'project', why: 'a package in tesseract', uses: 37 },
  { w: 'vendored', heard: 'vendored', right: true, kind: 'Term', from: 'memory', why: 'a word you use about packages', uses: 8 },
];
P.word = (w) => P.WORDS.find((x) => eqi(x.w, w));

/* What you say next, per app. Strings are heard right; T(heard, meant) is a term Whisper may miss. */
P.SCRIPTS = {
  terminal: [
    ['Ask ', T('cloud', 'Claude'), ' why the ', T('SRACT', 'Tesseract'), ' server drops the first request'],
    ['Then put the fix in ', T('cloud.md', 'CLAUDE.md'), ' and rebase the ', T('D flash two', 'DFlash2'), ' ', T('work tree', 'worktree')],
    [T('Cloud', 'Claude'), ', run the ', T('SRAX', 'Tesseract'), ' tests again before you open ', T('APR', 'a PR')],
    ['Good. Tell ', T('cloud', 'Claude'), ' to ship the ', T('D flash two', 'DFlash2'), ' build to ', T('test flight', 'TestFlight')],
  ],
  notes: [
    ['Low ', T('cloud', 'cloud'), ' this morning, so the drone test moves to Friday'],
    ['Ask ', T('whisper flow', 'Wispr Flow'), ' and ', T('Eleven Labs', 'ElevenLabs'), ' for team pricing'],
  ],
};

P.APPS = {
  terminal: { name: 'Terminal', place: 'Terminal · tesseract', short: 'tesseract', icon: 'terminal' },
  notes: { name: 'Notes', place: 'Notes', short: 'Notes', icon: 'notes' },
  safari: { name: 'Safari', place: 'Safari', short: 'Safari', icon: 'compass' },
};
P.placeOf = (app) => P.APPS[app]?.place || app;

const SEED_TAKES = [ // newest first, minutes ago
  { ago: 34, app: 'terminal', line: ['Why does the ', T('SRAX', 'Tesseract'), ' menu bar icon flicker after sleep'] },
  { ago: 58, app: 'notes', line: ['Idea: read chapter three aloud with the new voice before bed'] },
  { ago: 96, app: 'terminal', line: ['Rebase the ', T('D flash two', 'DFlash2'), ' branch onto main and rerun the benchmarks'] },
  { ago: 131, app: 'safari', line: [T('whisper flow', 'Wispr Flow'), ' pricing for teams'] },
  { ago: 1210, app: 'terminal', line: ['Tell ', T('cloud', 'Claude'), ' to add the release steps to ', T('cloud.md', 'CLAUDE.md')] },
  { ago: 1262, app: 'notes', line: ['Pick up coffee filters and call the dentist about Thursday'] },
  { ago: 1330, app: 'terminal', line: ['Can you check why the ', T('SRACT', 'Tesseract'), ' build fails on the release config'] },
];
const SEED_MEMORIES = [
  ['Owner is building ', T('SRACT', 'Tesseract'), ', an offline assistant for macOS'],
  ['Owner asks ', T('cloud', 'Claude'), ' for a code review before merging'],
  ['Benchmarks for the ', T('D flash two', 'DFlash2'), ' drafter live in the research folder'],
];

/* ---------- takes ---------- */
const segsFrom = (line) => line.map((s) => (typeof s === 'string' ? { text: s } : { h: s.h, m: s.m, text: s.h, status: eqi(s.h, s.m) ? 'right' : 'wrong' }));
P.segsFrom = segsFrom;
P.isMistake = (s) => s.m != null && !eqi(s.text, s.m);
P.makeTake = (app, line, at = new Date()) => {
  const t = { id: P.uid(), at, app, place: P.placeOf(app), segs: segsFrom(line) };
  P.st && (P.st.byId[t.id] = t);
  return t;
};
P.take = (id) => P.st.byId[id];
P.takeText = (t) => t.segs.map((s) => s.text).join('');
P.mistakes = (t) => t.segs.map((s, i) => [s, i]).filter(([s]) => P.isMistake(s));
const fitCase = (word, first) => (first && /^[a-z]/.test(word) ? word[0].toUpperCase() + word.slice(1) : word);
P.fitCase = fitCase;

/* Tokens: words for live partials and word pickers. A term is one token even when it is several words. */
P.tokens = (t) => {
  const out = []; let space = false;
  t.segs.forEach((s, i) => {
    if (s.h != null) { out.push({ seg: i, term: true, heard: s.h, text: s.text, glue: out.length > 0 && !space }); space = false; return; }
    const re = /(\s+)|(\S+)/g; let m;
    while ((m = re.exec(s.text))) {
      if (m[1]) space = true;
      else { out.push({ seg: i, term: false, text: m[2], glue: out.length > 0 && !space }); space = false; }
    }
  });
  return out;
};
P.joinTokens = (toks, map = (t) => P.esc(t.text)) => toks.map((t, i) => (i && !t.glue ? ' ' : '') + map(t, i)).join('');

/* Rules: what one fix teaches ("heard → meant"). Variants may scope them (Places). */
P.learn = ({ heard, meant, scope = 'everywhere', source = 'fix', note = '' }) => {
  const old = P.st.rules.find((r) => eqi(r.heard, heard) && r.scope === scope && r.on);
  if (old) { old.meant = meant; return old; }
  const r = { id: P.uid(), heard, meant, scope, source, note, at: new Date(), applied: 0, on: true };
  P.st.rules.unshift(r);
  P.emit('learn', { rule: r });
  return r;
};
P.unlearn = (r) => { r.on = false; P.emit('unlearn', { rule: r }); };
P.findRule = (heard, take) => (P.active?.findRule ? P.active.findRule(heard, take) : P.st.rules.find((r) => r.on && eqi(r.heard, heard)));
P.applyRules = (take) => {
  take.segs.forEach((s, i) => {
    if (s.h == null) return;
    const r = P.findRule(s.h, take);
    if (!r) return;
    s.text = fitCase(r.meant, i === 0);
    s.status = eqi(r.meant, s.m) ? 'auto' : 'overfix';
    s.rule = r.id; r.applied++;
  });
};

/* ---------- state ---------- */
P.reset = () => {
  uid = 0;
  const st = (P.st = {
    rules: [], takes: [], byId: {}, memories: [],
    target: 'terminal', front: 'terminal', lineIdx: { terminal: 0, notes: 0 },
    term: { log: [], input: [] }, notes: { paras: [] },
    phase: 'idle', talk: null, capture: null, steps: {}, zoomed: false,
  });
  st.takes = SEED_TAKES.map((s) => { const t = P.makeTake(s.app, s.line, new Date(Date.now() - s.ago * 60000)); t.seed = true; return t; });
  st.memories = SEED_MEMORIES.map((line) => ({ id: P.uid(), segs: segsFrom(line) }));
  st.term.log = st.takes.filter((t) => t.app === 'terminal').slice(0, 2).reverse().map((t) => P.takeHTML(t));
  st.notes.paras = st.takes.filter((t) => t.app === 'notes').slice(0, 2).reverse().map((t) => [{ take: t.id }]);
  $('#win-tess').classList.remove('zoomed');
};

/* ---------- events ---------- */
const listeners = {};
P.on = (type, fn) => ((listeners[type] ||= []).push(fn), () => (listeners[type] = listeners[type].filter((f) => f !== fn)));
P.emit = (type, data = {}) => {
  (listeners[type] || []).forEach((f) => f(data));
  try { P.active?.onEvent?.(type, data); } catch (err) { console.error(err); }
};

/* ---------- the mic level (fake, but it moves like speech) ---------- */
let lvOn = false, lv = 0, lvTarget = 0, lvNext = 0;
const frameSubs = new Set();
P.onFrame = (fn) => (frameSubs.add(fn), () => frameSubs.delete(fn));
P.level = () => lv;
const frame = (t) => {
  if (lvOn && t > lvNext) { lvTarget = 0.28 + Math.random() * 0.72; lvNext = t + 60 + Math.random() * 110; }
  if (!lvOn) lvTarget = 0;
  lv += (lvTarget - lv) * 0.2;
  frameSubs.forEach((f) => { try { f(lv, t); } catch (e) { console.error(e); } });
  requestAnimationFrame(frame);
};

/* ---------- dictation ---------- */
P.peekLine = (app = P.st.target) => { const s = P.SCRIPTS[app]; return s[P.st.lineIdx[app] % s.length]; };
P.saidHTML = (line) => line.map((s) => (typeof s === 'string' ? P.esc(s) : eqi(s.h, s.m) ? P.esc(s.m) : `<b>${P.esc(s.m)}</b>`)).join('');
P.saidText = (line) => line.map((s) => (typeof s === 'string' ? s : s.m)).join('');

P.startTalk = () => {
  const st = P.st;
  if (st.phase !== 'idle') return;
  setTalkUI(true);
  if (st.capture) { // a variant is listening for a single word (Roll Call)
    st.phase = 'listening'; st.listenT0 = performance.now(); lvOn = true;
    st.capture.onStart?.(); P.emit('listen-start');
    return;
  }
  const app = st.target;
  const take = P.makeTake(app, P.peekLine(app));
  take.toks = P.tokens(take);
  st.talk = { take, shown: 0, held: true };
  st.phase = 'recording'; lvOn = true;
  $('#mb-tess').classList.add('live');
  P.emit('rec-start', { take });
  st.talk.timer = setTimeout(tick, 380);
};
const tick = () => {
  const t = P.st.talk; if (!t) return;
  if (t.shown < t.take.toks.length) {
    t.shown++;
    P.emit('partial', { take: t.take, shown: t.shown });
    t.timer = setTimeout(tick, t.held ? 230 + Math.random() * 150 : 60);
  } else if (!t.held) finishTalk();
  else t.done = true;
};
P.stopTalk = () => {
  const st = P.st;
  setTalkUI(false);
  if (st.phase === 'listening') {
    st.phase = 'idle'; lvOn = false;
    const ms = performance.now() - st.listenT0;
    P.emit('listen-end', { ms }); st.capture?.onEnd?.(ms);
    return;
  }
  const t = st.talk; if (!t || !t.held) return;
  t.held = false;
  if (t.done) finishTalk(); else { clearTimeout(t.timer); tick(); }
};
const finishTalk = async () => {
  const st = P.st, t = st.talk; if (!t || t.finishing) return;
  t.finishing = true; lvOn = false;
  $('#mb-tess').classList.remove('live');
  st.phase = 'processing';
  P.emit('processing', { take: t.take });
  await P.wait(360);
  if (P.st !== st) return; // reset while transcribing
  const take = t.take;
  P.applyRules(take);
  P.active?.transform?.(take);
  st.talk = null; st.lineIdx[take.app]++;
  st.takes.unshift(take); st.last = take;
  st.phase = 'idle';
  const held = P.active?.beforeInsert?.(take) === false;
  P.emit('final', { take, held });
  if (!held) P.insert(take);
  renderHarness();
};
P.discard = (take) => { // a held take that never lands
  P.st.takes = P.st.takes.filter((t) => t !== take);
  P.emit('discard', { take });
};

/* ---------- where the words land ---------- */
P.insert = (take) => {
  if (take.app === 'terminal') {
    if (P.st.term.input.length) P.sendTerminal();
    P.st.term.input.push({ take: take.id });
  } else {
    P.st.notes.paras.push([{ take: take.id }]);
  }
  take.inserted = true; take.live = true;
  renderTargets();
  P.emit('inserted', { take });
};
P.sendTerminal = () => {
  const inp = P.st.term.input;
  if (!inp.length) return;
  inp.forEach((p) => { if (p.take) P.take(p.take).live = false; });
  P.st.term.log.push(partsHTML(inp));
  P.st.term.log = P.st.term.log.slice(-6);
  P.st.term.input = [];
  renderTargets();
  P.emit('sent', {});
};
P.lastLive = () => P.st.takes.find((t) => t.live);

/* Replace one term of a take, in the app if it is still there. Returns true when it changed in place. */
P.fixSeg = (take, i, text, status = 'fixed') => {
  const s = take.segs[i];
  s.prev = s.text; s.text = fitCase(text, i === 0); s.status = eqi(s.text, s.m) ? status : 'wrong'; s.flash = Date.now();
  renderTargets();
  P.emit('seg-fixed', { take, i, inPlace: !!take.live });
  return !!take.live;
};
P.findSeg = (take, heard) => take.segs.findIndex((s) => s.h != null && eqi(s.h, heard) && s.text === s.h);

/* Typing into the front app */
const inputParts = (app) => (app === 'terminal' ? P.st.term.input : P.st.notes.paras[P.st.notes.paras.length - 1] || (P.st.notes.paras.push([]), P.st.notes.paras[0]));
P.typedTail = (app = P.st.front) => {
  const parts = inputParts(app); let s = '';
  for (let i = parts.length - 1; i >= 0 && parts[i].typed != null; i--) s = parts[i].typed + s;
  return s;
};
P.trimTyped = (app, n) => {
  const parts = inputParts(app);
  while (n > 0 && parts.length && parts[parts.length - 1].typed != null) {
    const p = parts[parts.length - 1];
    const cut = Math.min(n, p.typed.length);
    p.typed = p.typed.slice(0, p.typed.length - cut); n -= cut;
    if (!p.typed) parts.pop();
  }
  renderTargets();
};
P.typeInto = (app, key) => {
  if (app === 'terminal' && key === 'Enter') { P.sendTerminal(); return; }
  if (app === 'notes' && key === 'Enter') { P.st.notes.paras.push([]); renderTargets(); return; }
  const parts = inputParts(app);
  const last = parts[parts.length - 1];
  if (key === 'Backspace') {
    if (!last) return;
    if (last.typed != null) { last.typed = last.typed.slice(0, -1); if (!last.typed) parts.pop(); }
    else { const t = P.take(last.take); t.live = false; parts[parts.length - 1] = { typed: P.takeText(t).slice(0, -1) }; }
  } else if (last && last.typed != null) last.typed += key;
  else parts.push({ typed: key });
  renderTargets();
  P.emit('typed', { app, key, tail: P.typedTail(app) });
};

/* ---------- rendering the target apps ---------- */
P.segHTML = (take, s, i) => {
  const mis = P.isMistake(s);
  const flash = s.flash && Date.now() - s.flash < 1500;
  return `<span class="seg${mis ? ' mis' : ''}${flash ? ' flash' : ''}" data-take="${take.id}" data-seg="${i}"${mis ? ` title="You said ${P.esc(s.m)}"` : ''}>${P.esc(s.text)}</span>`;
};
P.takeHTML = (take) => take.segs.map((s, i) => (s.h != null ? P.segHTML(take, s, i) : P.esc(s.text))).join('');
const partsHTML = (parts) => parts.map((p) => (p.take ? P.takeHTML(P.take(p.take)) : P.esc(p.typed))).join('');
const renderTargets = () => {
  const st = P.st;
  $('#term-log').innerHTML = st.term.log.map((html) => `<div class="ln"><i>›</i>${html}</div>`).join('');
  $('#term-box').innerHTML = `<span class="pr">›</span>${partsHTML(st.term.input)}<span class="caret"></span>`;
  const paras = st.notes.paras.map((parts) => `<p>${partsHTML(parts) || '&nbsp;'}</p>`);
  $('#notes-body').innerHTML = `<h4>Field log</h4><div class="nd">Today, ${P.clock(new Date(), false)}</div>${paras.join('')}`.replace(/<\/p>$/, '<span class="notes-caret"></span></p>');
  P.emit('targets-rendered');
};
P.renderTargets = renderTargets;

/* ---------- windows ---------- */
P.focusWin = (id) => {
  const st = P.st;
  if (st.front === id) return;
  st.front = id;
  if (id === 'terminal' || id === 'notes') st.target = id;
  $$('.win').forEach((w) => w.classList.toggle('front', w.dataset.win === id));
  $('#mb-app').textContent = id === 'tess' ? 'Tesseract Agent' : P.APPS[id].name;
  renderHarness();
  P.emit('focus', { id });
};
P.menuBadge = (n) => {
  const b = $('#mb-tess');
  b.innerHTML = P.tesseractGlyph(15) + (n ? `<span class="badge">${n}</span>` : '');
};

/* ---------- time ---------- */
P.clock = (d = new Date(), withDay = true) => {
  const hm = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  if (!withDay) return hm;
  return `${d.toLocaleDateString('en-GB', { weekday: 'short' })} ${d.getDate()} ${d.toLocaleDateString('en-GB', { month: 'short' })}  ${hm}`;
};
P.ago = (d) => {
  const min = Math.round((Date.now() - d.getTime()) / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min} min ago`;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const hm = P.clock(d, false);
  return d >= today ? hm : `Yesterday ${hm}`;
};

/* ---------- harness ---------- */
P.variants = [];
P.register = (v) => { P.variants.push(v); P.variants.sort((a, b) => a.num - b.num); };
P.step = (id) => { if (!P.st.steps[id]) { P.st.steps[id] = true; renderHarness(); } };
const renderHarness = () => {
  const v = P.active; if (!v) return;
  const i = P.variants.indexOf(v);
  $('#h-num').textContent = `${i + 1}/${P.variants.length}`;
  $('#h-name').textContent = v.name;
  $('#h-thesis').textContent = v.thesis;
  let nowSet = false;
  $('#h-steps').innerHTML = v.steps.map((s) => {
    const done = P.st.steps[s.id];
    const now = !done && !nowSet && (nowSet = true);
    return `<li class="${done ? 'done' : now ? 'now' : ''}"><span>${s.text}</span></li>`;
  }).join('');
  $('#h-say-app').textContent = `into ${P.APPS[P.st.target].name}`;
  $('#h-say-line').innerHTML = P.st.capture && P.st.front === 'tess' ? (P.st.capture.hint || 'the word on the card') : P.saidHTML(P.peekLine());
};
P.renderHarness = renderHarness;
const setTalkUI = (on) => $('#h-talk').classList.toggle('on', on);

const renderSwitcher = () => {
  const sw = $('#sw');
  sw.innerHTML = `<button class="arr" data-go="-1" aria-label="Previous variant">‹</button>` +
    P.variants.map((v) => `<button class="v${v === P.active ? ' on' : ''}" data-id="${v.id}">${v.num} <b>${v.name}</b></button>`).join('') +
    `<button class="arr" data-go="1" aria-label="Next variant">›</button>`;
};
P.go = (delta) => {
  const n = P.variants.length, i = P.variants.indexOf(P.active);
  P.show(P.variants[(i + delta + n) % n].id);
};
P.show = (id) => {
  const v = P.variants.find((x) => x.id === id);
  if (v && v !== P.active) P.mount(v);
  try { history.replaceState(null, '', '#' + id); } catch (e) { /* the viewer's frame may refuse; the switch already happened */ }
};

P.mount = (v) => {
  try { P.active?.unmount?.(); } catch (e) { console.error(e); }
  clearTimeout(P.st?.talk?.timer);
  lvOn = false; setTalkUI(false);
  P.active = v;
  try { localStorage.setItem('dictation-proto-variant', v.id); } catch (e) { /* private window */ }
  P.reset();
  $('#variant-css').textContent = v.css || '';
  $('#tess-content').innerHTML = ''; $('#tess-tools').innerHTML = ''; $('#overlay-layer').innerHTML = '';
  $('#screen').dataset.variant = v.id;
  $$('.win').forEach((w) => w.classList.toggle('front', w.dataset.win === 'terminal'));
  $('#mb-app').textContent = 'Terminal';
  P.menuBadge(0);
  v.mount({ page: $('#tess-content'), tools: $('#tess-tools'), overlay: $('#overlay-layer'), win: $('#win-tess') });
  renderTargets(); renderHarness(); renderSwitcher();
};

/* ---------- input ---------- */
const isField = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
const onKeyDown = (e) => {
  const inField = isField(document.activeElement) || isField(e.target);
  const talkKey = (!inField && (e.code === 'Backquote' || e.key === '`') && !e.metaKey && !e.ctrlKey) || (e.code === 'Space' && e.altKey && !e.ctrlKey && !e.metaKey);
  const fixKey = (!inField && e.key === 'Tab' && !e.shiftKey && !e.altKey && !e.metaKey) || (e.code === 'Space' && e.altKey && e.ctrlKey);
  if (talkKey && P.st.phase !== 'idle' && e.repeat) { e.preventDefault(); return; }
  if (P.active?.onKey?.(e, { inField, talkKey, fixKey })) { e.preventDefault(); return; }
  if (talkKey) { e.preventDefault(); if (!e.repeat) P.startTalk(); return; }
  if (fixKey) { e.preventDefault(); P.active?.onFixKey?.(); return; }
  if (inField || e.metaKey || e.ctrlKey) return;
  const front = P.st.front;
  if ((front === 'terminal' || front === 'notes') && (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Enter')) {
    e.preventDefault(); P.typeInto(front, e.key); return;
  }
  if (e.key === 'ArrowLeft') { e.preventDefault(); P.go(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); P.go(1); }
};
const onKeyUp = (e) => {
  if (e.code === 'Backquote' || e.key === '`' || e.code === 'Space' || e.key === 'Alt') {
    if (P.st.phase === 'recording' || P.st.phase === 'listening') P.stopTalk();
  }
  P.active?.onKeyUp?.(e);
};

const fit = () => {
  const r = $('#stage').getBoundingClientRect();
  const s = Math.max(0.2, Math.min((r.width - 8) / 1280, (r.height - 8) / 800));
  $('#screen').style.setProperty('--s', s.toFixed(4));
  P.scale = s;
};

P.boot = () => {
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', () => { if (P.st?.phase === 'recording' || P.st?.phase === 'listening') P.stopTalk(); });
  $$('.win').forEach((w) => w.addEventListener('pointerdown', () => P.focusWin(w.dataset.win)));
  $('#tess-zoom').addEventListener('click', (e) => {
    e.stopPropagation();
    P.st.zoomed = $('#win-tess').classList.toggle('zoomed');
    P.focusWin('tess'); P.emit('zoom', { zoomed: P.st.zoomed });
  });
  const talk = $('#h-talk');
  talk.addEventListener('pointerdown', (e) => { e.preventDefault(); talk.setPointerCapture?.(e.pointerId); P.startTalk(); });
  talk.addEventListener('pointerup', () => P.stopTalk());
  talk.addEventListener('pointercancel', () => P.stopTalk());
  $('#h-fix').addEventListener('click', () => P.active?.onFixKey?.());
  $('#h-reset').addEventListener('click', () => P.mount(P.active));
  $('#h-truth').addEventListener('change', (e) => $('#screen').classList.toggle('truth', e.target.checked));
  $('#sw').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.go) P.go(+b.dataset.go); else P.show(b.dataset.id);
  });
  window.addEventListener('hashchange', () => {
    const v = P.variants.find((x) => x.id === location.hash.slice(1));
    if (v && v !== P.active) P.mount(v);
  });
  new ResizeObserver(fit).observe($('#stage'));
  fit();
  const tickClock = () => ($('#mb-clock').textContent = P.clock());
  tickClock(); setInterval(tickClock, 15000);
  requestAnimationFrame(frame);
  let want = location.hash.slice(1) || new URLSearchParams(location.search).get('variant');
  if (!want) { try { want = localStorage.getItem('dictation-proto-variant'); } catch (e) { /* ignore */ } }
  P.mount(P.variants.find((v) => v.id === want || String(v.num) === want) || P.variants[0]);
};
})();
