/* PROTOTYPE, THROWAWAY. Variant 3, Ask Me.
   It notices; you answer. When a word it wrote sounds like one of your words (a project, a name from
   memory) or isn't a word at all, it asks one short question instead of guessing. Answer in the overlay
   with one key (⌃⌥Space yes, esc no), or later on the page, which is a conversation with your dictation.
   It never changes a word you didn't confirm. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'ask-me', num: 3, name: 'Ask Me',
  thesis: 'It notices; you answer. One question at a time, one key to say yes. It never changes a word you didn’t confirm.',
  steps: [
    { id: 'asked', text: 'Hold <b>`</b> and dictate. It asks about a word it isn’t sure of' },
    { id: 'yes', text: 'Say yes with <b>Tab</b> (⌃⌥Space). The word is fixed where it landed' },
    { id: 'page', text: 'Click the Tesseract window and answer the rest: <b>Y</b>, <b>N</b>, or <b>E</b> to type it' },
    { id: 'again', text: 'Dictate again. What you answered comes out right' },
  ],
};

V.css = `
.am{max-width:640px;margin:0 auto;padding:6px 30px 150px;display:flex;flex-direction:column;gap:18px;font-family:var(--rounded)}
.am-day{align-self:center;font:600 11.5px/1 var(--rounded);letter-spacing:.04em;color:var(--ink3);padding:6px 0 0}
.am-msg{display:flex;flex-direction:column;gap:8px;max-width:560px}
.am-meta{display:flex;align-items:center;gap:7px;font:500 12px/1 var(--rounded);color:var(--ink3)}
.am-meta .who{color:var(--ink2);font-weight:700}
.am-meta svg{color:var(--accent-text)}
.am-text{font:500 15.5px/1.5 var(--rounded);color:var(--ink)}
.am-quote{font:14.5px/1.5 var(--sans);color:var(--ink2);padding:2px 0 2px 12px;border-left:2px solid var(--line)}
.am-quote mark{background:none;color:var(--ink);font:500 13.5px/1 var(--mono);padding:1px 3px;border-radius:4px;box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 70%,transparent)}
.am-ask-q{font:700 22px/1.25 var(--rounded);letter-spacing:-.01em;color:var(--ink);margin:2px 0 0}
.am-ask-q b{color:var(--accent-text);font-weight:800}
.am-why{display:flex;gap:7px;align-items:center;font:500 13px/1.4 var(--rounded);color:var(--ink3)}
.am-why svg{flex:none}
.am-replies{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.am-replies button{height:34px;padding:0 13px;border-radius:17px;border:1px solid var(--line);background:var(--win-bg);color:var(--ink);font:600 13.5px/1 var(--rounded);display:inline-flex;align-items:center;gap:8px;cursor:pointer}
.am-replies button:hover{border-color:var(--ink3)}
.am-replies button.yes{background:var(--accent);border-color:transparent;color:var(--accent-ink)}
.am-replies button.yes .kc{border-color:rgba(0,0,0,.12);background:rgba(255,255,255,.22);color:inherit}
.am-replies button.ghost{border-color:transparent;color:var(--ink2);background:transparent}
.am-replies .kc{font:600 10.5px/1 var(--rounded);min-width:17px;height:17px}
.am-msg:not(.active) .am-replies .kc{display:none}
.am-msg.active .am-ask-q::after{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);margin-left:10px;vertical-align:4px;animation:ampulse 1.6s ease-in-out infinite}
@keyframes ampulse{50%{opacity:.25}}
.am-other{display:flex;gap:8px;margin-top:2px}
.am-other input{flex:1;height:36px;border-radius:18px;border:1px solid var(--accent);background:var(--win-bg);color:var(--ink);padding:0 14px;font:600 15px var(--rounded);outline:none;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent)}
.am-play{display:inline-flex;align-items:center;gap:6px}
.am-play .wv{display:inline-flex;gap:2px;align-items:center;height:14px}
.am-play .wv i{width:2.5px;border-radius:2px;background:currentColor;height:4px;transition:height .12s}
.am-play.on .wv i{animation:amwv .5s ease-in-out infinite alternate}
.am-play .wv i:nth-child(2){animation-delay:.1s}.am-play .wv i:nth-child(3){animation-delay:.2s}.am-play .wv i:nth-child(4){animation-delay:.05s}.am-play .wv i:nth-child(5){animation-delay:.15s}
@keyframes amwv{to{height:13px}}
.am-me{align-self:flex-end;max-width:70%;background:color-mix(in srgb,var(--accent) 17%,var(--win-bg));color:var(--ink);font:600 15px/1.4 var(--rounded);padding:9px 15px;border-radius:19px 19px 5px 19px}
.am-me.fresh{animation:amin .3s cubic-bezier(.2,1.2,.4,1)}
@keyframes amin{from{opacity:0;transform:translateY(8px) scale(.96)}}
.am-note{display:flex;gap:9px;align-items:baseline;font:500 14px/1.45 var(--rounded);color:var(--ink2);max-width:560px}
.am-note.fresh{animation:amin .35s ease .15s both}
.am-note .ok{color:var(--ok);flex:none;transform:translateY(2px)}
.am-note b{color:var(--ink);font-weight:700}
.am-note button{border:0;background:none;padding:0;margin-left:4px;color:var(--accent-text);font:700 13.5px var(--rounded);cursor:pointer}
.am-note .h{font:500 13px var(--mono);text-decoration:line-through;text-decoration-color:var(--danger)}
.am-composer{position:sticky;bottom:14px;margin-top:10px;display:flex;align-items:center;gap:10px;height:46px;padding:0 8px 0 16px;border-radius:23px}
.am-composer input{flex:1;border:0;background:transparent;outline:none;font:500 15px var(--rounded);color:var(--glass-ink)}
.am-composer input::placeholder{color:var(--glass-ink2)}
.am-composer .send{width:32px;height:32px;border-radius:16px;border:0;background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;cursor:pointer}
.am-words{position:absolute;right:0;top:38px;width:300px;border-radius:16px;padding:10px 12px;z-index:10;display:flex;flex-direction:column;gap:2px;font-family:var(--rounded)}
.am-words h4{margin:2px 4px 6px;font:700 12px/1 var(--rounded);color:var(--glass-ink2)}
.am-words div{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:baseline;padding:6px 4px;border-top:1px solid var(--line);font:600 14px/1.3 var(--rounded)}
.am-words small{grid-column:1;font:12px var(--mono);color:var(--glass-ink2);font-weight:400}
.am-words button{grid-row:1/3;grid-column:2;border:0;background:none;color:var(--accent-text);font:700 12px var(--rounded);cursor:pointer}
.am-count{display:inline-grid;place-items:center;min-width:18px;height:18px;border-radius:9px;background:var(--accent);color:var(--accent-ink);font:800 11px/1 var(--rounded);padding:0 5px}

/* overlay */
.am-pill{position:absolute;left:50%;bottom:40px;transform:translateX(-50%) scale(.85);opacity:0;height:38px;padding:0 16px;border-radius:19px;display:flex;align-items:center;gap:10px;font:600 13px/1 var(--rounded);transition:opacity .18s,transform .22s cubic-bezier(.2,1.2,.4,1);pointer-events:none;white-space:nowrap}
.am-pill.show{opacity:1;transform:translateX(-50%) scale(1)}
.am-pill .bars{display:flex;gap:3px;align-items:center;height:18px}
.am-pill .bars span{width:3px;height:4px;border-radius:2px;background:var(--danger)}
.am-ask{position:absolute;left:50%;bottom:36px;transform:translateX(-50%);display:flex;align-items:center;gap:12px;height:50px;padding:0 8px 0 10px;border-radius:25px;font:600 15px/1 var(--rounded);white-space:nowrap;animation:amask .35s cubic-bezier(.2,1.3,.4,1);transition:opacity .3s,transform .45s cubic-bezier(.5,0,.7,.4)}
@keyframes amask{from{opacity:0;transform:translateX(-50%) translateY(14px) scale(.9)}}
.am-ask.fly{opacity:0;transform:translate(420px,-720px) scale(.2)}
.am-ask .ring{width:30px;height:30px;flex:none;display:grid;place-items:center;color:var(--accent-text)}
.am-ask .ring svg{position:absolute}
.am-ask .h{font:500 14px/1 var(--mono);color:var(--glass-ink2);text-decoration:line-through;text-decoration-color:var(--danger)}
.am-ask b{font-weight:800;color:var(--accent-text)}
.am-ask .btns{display:flex;gap:6px;margin-left:4px}
.am-ask .btns button{height:34px;border-radius:17px;border:0;padding:0 12px;font:700 13px/1 var(--rounded);display:inline-flex;gap:7px;align-items:center;cursor:pointer;background:color-mix(in srgb,var(--glass-ink) 8%,transparent);color:var(--glass-ink)}
.am-ask .btns button.yes{background:var(--accent);color:var(--accent-ink)}
.am-ask .btns .kc{font:600 10.5px/1 var(--rounded);border-color:rgba(0,0,0,.1);background:rgba(255,255,255,.2);color:inherit}
.am-ask.done{padding:0 18px 0 14px}
.am-ask.done .ok{color:var(--ok)}
`;

/* ---------- why it asks (what it knows about the word) ---------- */
const EXTRA = { 'a PR': { why: 'isn’t a word, and you say “a PR” often in Terminal', icon: 'terminal' } };
const reasonFor = (heard, meant) => {
  const w = P.word(meant);
  const nonWord = /^[A-Z]{3,}$/.test(heard) || /\.md$/.test(heard);
  if (EXTRA[meant]) return { text: `“${heard}” ${EXTRA[meant].why}.`, icon: EXTRA[meant].icon };
  if (!w) return { text: `“${heard}” sounds like ${meant}.`, icon: 'ear' };
  const src = w.from === 'project' ? 'folder' : 'brain';
  return { text: `${nonWord ? `“${heard}” isn’t a word. ` : ''}It sounds like ${meant}, ${w.why}.`, icon: src };
};

/* ---------- thread ---------- */
V.mount = (ctx) => {
  S = { ctx, items: [], seq: 0, overlayQ: null, askTimer: 0, pillTimer: 0, playing: null };
  const DAY = 86400e3;
  // Yesterday's answered question, and one from this morning still open.
  const learnedEL = P.learn({ heard: 'Eleven Labs', meant: 'ElevenLabs', source: 'question' });
  S.items.push({ kind: 'day', label: 'Monday' });
  S.items.push({ kind: 'intro', at: new Date(Date.now() - 2.1 * DAY) });
  S.items.push({ kind: 'day', label: 'Yesterday' });
  const yq = q({ quote: ['Compare the ', P.T('Eleven Labs', 'ElevenLabs'), ' voices with ours on the long chapter'], heard: 'Eleven Labs', meant: 'ElevenLabs', app: 'safari', at: new Date(Date.now() - DAY - 3 * 3600e3) });
  yq.state = 'yes'; yq.rule = learnedEL; yq.note = `Learned. From now on I’ll write <b>ElevenLabs</b>.`;
  S.items.push({ kind: 'day', label: 'Today' });
  const dfl = P.st.takes.find((t) => t.segs.some((s) => s.h === 'D flash two'));
  q({ take: dfl, seg: dfl.segs.findIndex((s) => s.h === 'D flash two'), heard: 'D flash two', meant: 'DFlash2', app: 'terminal', at: dfl.at });
  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>
    <div style="position:relative"><button class="gbtn" id="am-wbtn">${P.icon('sparkle', 14)}What I know</button></div>`;
  ctx.tools.querySelector('#am-wbtn').onclick = (e) => { e.stopPropagation(); toggleWords(); };
  ctx.overlay.innerHTML = `<div class="am-pill glass" id="am-pill"></div>`;
  S.unFrame = P.onFrame(meter);
  render(true);
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.askTimer); clearTimeout(S?.pillTimer); S = null; };

const q = (o) => {
  const item = { kind: 'q', id: ++S.seq, state: 'open', ...o };
  item.reason = reasonFor(o.heard, o.meant);
  S.items.push(item);
  return item;
};
const open = () => S.items.filter((x) => x.kind === 'q' && x.state === 'open');
const active = () => { const o = open(); return o[o.length - 1]; };
const badge = () => { P.menuBadge(open().length); const c = S.ctx.tools.querySelector('#am-wbtn'); if (c) c.innerHTML = `${P.icon('sparkle', 14)}What I know`; };

const quoteHTML = (it) => {
  const segs = it.take ? it.take.segs.map((s, i) => ({ ...s, i })) : P.segsFrom(it.quote).map((s, i) => ({ ...s, i }));
  const hit = it.take ? it.seg : segs.findIndex((s) => s.h === it.heard);
  return segs.map((s) => (s.i === hit ? `<mark>${P.esc(it.state === 'open' || it.state === 'no' ? it.heard : (it.answer || it.meant))}</mark>` : P.esc(s.text))).join('');
};
const itemHTML = (it) => {
  if (it.kind === 'day') return `<div class="am-day">${it.label}</div>`;
  if (it.kind === 'intro') return `<div class="am-msg"><div class="am-meta">${P.tesseractGlyph(13)}<span class="who">Dictation</span><span>${P.clock(it.at, false)}</span></div>
    <div class="am-text">When something you say sounds like one of your words, I’ll ask here instead of guessing. One answer teaches me, and I never change a word you didn’t confirm.</div></div>`;
  if (it.kind === 'you') return `<div class="am-me${it.fresh ? ' fresh' : ''}">${P.esc(it.text)}</div>`;
  if (it.kind === 'note') return `<div class="am-note${it.fresh ? ' fresh' : ''}"><span class="ok">${P.icon(it.icon || 'check', 15, 2.4)}</span><span>${it.html}</span></div>`;
  const isActive = it === active();
  const time = P.ago(it.at).replace('Yesterday ', '');
  const head = `<div class="am-meta">${P.tesseractGlyph(13)}<span class="who">Dictation</span><span>${time}</span><span>· ${P.APPS[it.app].name}</span></div>
    <div class="am-quote">“${quoteHTML(it)}”</div>
    <div class="am-ask-q">Did you mean <b>${P.esc(it.meant)}</b>?</div>`;
  if (it.state === 'open') return `<div class="am-msg${isActive ? ' active' : ''}" data-q="${it.id}">${head}
    <div class="am-why">${P.icon(it.reason.icon, 14)}<span>${P.esc(it.reason.text)}</span></div>
    ${it.typing ? `<div class="am-other"><input id="am-other-${it.id}" placeholder="Type how it’s spelled" autocomplete="off" spellcheck="false"></div>` : `<div class="am-replies">
      <button class="yes" data-a="yes">Yes, ${P.esc(it.meant)} <span class="kc">Y</span></button>
      <button data-a="no">No, keep “${P.esc(it.heard)}” <span class="kc">N</span></button>
      <button data-a="other">Something else <span class="kc">E</span></button>
      <button class="ghost am-play${S.playing === it.id ? ' on' : ''}" data-a="play"><span class="wv"><i></i><i></i><i></i><i></i><i></i></span>Hear it</button>
    </div>`}</div>`;
  const reply = it.state === 'yes' ? `Yes, ${it.meant}` : it.state === 'no' ? `No, keep “${it.heard}”` : `It’s ${it.answer}`;
  return `<div class="am-msg" data-q="${it.id}">${head}</div><div class="am-me${it.fresh ? ' fresh' : ''}">${P.esc(reply)}</div>${it.note ? `<div class="am-note${it.fresh ? ' fresh' : ''}"><span class="ok">${P.icon(it.state === 'no' ? 'x' : 'check', 15, 2.4)}</span><span>${it.note}${it.rule && it.rule.on ? ` <button data-undo="${it.id}">Undo</button>` : ''}</span></div>` : ''}`;
};
const render = (toBottom) => {
  const page = S.ctx.page;
  const keep = page.scrollTop;
  page.innerHTML = `<div class="am">${S.items.map(itemHTML).join('')}
    <div class="am-composer glass"><input id="am-say" placeholder="Tell me a word, like “it’s spelled TestFlight”" autocomplete="off"><button class="send" aria-label="Send">${P.icon('arrow', 16, 2.2)}</button></div></div>`;
  page.onclick = onPageClick;
  const inp = page.querySelector('#am-say');
  inp.onkeydown = (e) => { if (e.key === 'Enter') { e.stopPropagation(); tell(inp.value); } if (e.key === 'Escape') inp.blur(); };
  page.querySelector('.am-composer .send').onclick = () => tell(inp.value);
  const other = page.querySelector('.am-other input');
  if (other) {
    other.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === 'Escape') e.stopPropagation();
      const it = S.items.find((x) => x.typing);
      if (e.key === 'Escape') { it.typing = false; render(); }
      if (e.key === 'Enter' && other.value.trim()) answer(it, 'other', other.value.trim());
    };
    setTimeout(() => other.focus(), 20);
  }
  page.scrollTop = toBottom ? page.scrollHeight : keep;
  S.items.forEach((x) => { x.fresh = false; });
  badge();
};
const onPageClick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.undo) { undo(S.items.find((x) => x.id === +b.dataset.undo)); return; }
  if (b.dataset.undoRule) {
    const r = P.st.rules.find((x) => x.id === +b.dataset.undoRule); if (r) P.unlearn(r);
    const note = S.items.find((x) => x.kind === 'note' && x.html.includes(`data-undo-rule="${b.dataset.undoRule}"`));
    if (note) note.html = 'Undone. I’ll ask when I hear it again.';
    render(); return;
  }
  const card = b.closest('[data-q]'); if (!card || !b.dataset.a) return;
  const it = S.items.find((x) => x.id === +card.dataset.q);
  if (b.dataset.a === 'play') { play(it); return; }
  if (b.dataset.a === 'other') { it.typing = true; render(); return; }
  answer(it, b.dataset.a);
};
const play = (it) => { S.playing = it.id; render(); setTimeout(() => { if (S && S.playing === it.id) { S.playing = null; render(); } }, 1400); };

/* ---------- answering ---------- */
const answer = (it, how, text) => {
  if (!it || it.state !== 'open') return;
  it.typing = false; it.fresh = true;
  it.state = how; it.answer = text || it.meant;
  if (how === 'no') {
    it.note = `OK. I’ll leave “${P.esc(it.heard)}” alone in ${P.APPS[it.app].name}.`;
  } else {
    const meant = how === 'other' ? text : it.meant;
    it.rule = P.learn({ heard: it.heard, meant, source: 'question' });
    let inPlace = false;
    if (it.take && it.take.segs[it.seg].text === it.heard) inPlace = P.fixSeg(it.take, it.seg, meant);
    // Older takes with the same heard form are fixed in the history too.
    let older = 0;
    P.st.takes.forEach((t) => t.segs.forEach((s, i) => { if (t !== it.take && s.h != null && P.eqi(s.h, it.heard) && s.text === s.h) { s.text = P.fitCase(meant, i === 0); s.status = P.eqi(s.text, s.m) ? 'fixed' : 'wrong'; older++; } }));
    // Other open questions about the same sound are answered by this one.
    open().forEach((o) => { if (o !== it && P.eqi(o.heard, it.heard)) { o.state = how; o.answer = meant; o.note = 'Same answer as above.'; if (o.take && o.take.segs[o.seg].text === o.heard) P.fixSeg(o.take, o.seg, meant); } });
    const where = inPlace ? `Fixed it in ${P.APPS[it.app].name}` : it.take?.inserted && !it.take.seed ? 'Fixed it in your history (that one was already sent)' : 'Noted';
    it.note = `${where}${older ? ` and in ${older} older ${older === 1 ? 'take' : 'takes'}` : ''}. From now on I’ll write <b>${P.esc(meant)}</b>.`;
    P.renderTargets();
  }
  if (P.st.front === 'tess') P.step('page');
  if (S.overlayQ === it) { if (how === 'yes') P.step('yes'); closeAsk(how); }
  render(true);
};
const undo = (it) => {
  if (!it?.rule) return;
  P.unlearn(it.rule);
  if (it.take) { const s = it.take.segs[it.seg]; if (it.take.live) P.fixSeg(it.take, it.seg, it.heard, 'wrong'); else { s.text = it.heard; s.status = 'wrong'; } }
  it.note = `Undone. I’ll ask again next time I hear “${P.esc(it.heard)}”.`;
  it.rule = null;
  render();
};
const tell = (raw) => {
  const text = (raw || '').trim(); if (!text) return;
  const word = text.replace(/^(it[’']?s|the word is|spell it|spelled|it is)\s+/i, '').replace(/^spelled\s+/i, '').replace(/[.!]$/, '').trim();
  const w = P.word(word);
  S.items.push({ kind: 'you', text, fresh: true });
  if (w && !w.right) {
    const r = P.learn({ heard: w.heard, meant: w.w, source: 'told' });
    S.items.push({ kind: 'note', fresh: true, html: `Got it. When I hear “${P.esc(w.heard)}” I’ll write <b>${P.esc(w.w)}</b>. <button data-undo-rule="${r.id}">Undo</button>` });
  } else if (w) S.items.push({ kind: 'note', fresh: true, icon: 'ear', html: `I already hear <b>${P.esc(w.w)}</b> right. Nothing to change.` });
  else S.items.push({ kind: 'note', fresh: true, icon: 'ear', html: `I’ll listen for <b>${P.esc(word)}</b>. If something close comes up, I’ll ask.` });
  render(true);
};

/* ---------- overlay ---------- */
const pill = () => S.ctx.overlay.querySelector('#am-pill');
const meter = (lv) => {
  if (!S) return;
  pill()?.querySelectorAll('.bars span').forEach((b, i) => { b.style.height = `${4 + lv * 14 * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7 + performance.now() / 170)))}px`; });
};
const ASK_MS = 7000;
const showAsk = (it) => {
  S.overlayQ = it;
  S.ctx.overlay.querySelector('.am-ask')?.remove();
  const el = P.el(`<div class="am-ask glass">
    <span class="ring"><svg width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="12" fill="none" stroke="color-mix(in srgb,var(--glass-ink) 12%,transparent)" stroke-width="2.5"/><circle id="am-arc" cx="15" cy="15" r="12" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="75.4" stroke-dashoffset="0" transform="rotate(-90 15 15)" style="transition:stroke-dashoffset ${ASK_MS}ms linear"/></svg>${P.icon('ear', 13, 2)}</span>
    <span><span class="h">${P.esc(it.heard)}</span></span><span>${P.icon('arrow', 15, 2.2)}</span><span><b>${P.esc(it.meant)}</b>?</span>
    <span class="btns"><button class="yes" data-a="yes">Yes <span class="kc">⌃⌥Space</span></button><button data-a="no">No <span class="kc">esc</span></button></span>
  </div>`);
  S.ctx.overlay.appendChild(el);
  el.onclick = (e) => { const b = e.target.closest('button'); if (b) answer(it, b.dataset.a); };
  requestAnimationFrame(() => requestAnimationFrame(() => { const arc = el.querySelector('#am-arc'); if (arc) arc.style.strokeDashoffset = '75.4'; }));
  clearTimeout(S.askTimer);
  S.askTimer = setTimeout(() => { if (S?.overlayQ === it) closeAsk('later'); }, ASK_MS);
};
const closeAsk = (how) => {
  const el = S.ctx.overlay.querySelector('.am-ask'); S.overlayQ = null; clearTimeout(S.askTimer);
  if (!el) return;
  if (how === 'later') { el.classList.add('fly'); setTimeout(() => el.remove(), 500); badge(); return; }
  el.classList.add('done');
  el.innerHTML = how === 'no' ? `<span>${P.icon('x', 15, 2.2)}</span><span>Left as you said it</span>` : `<span class="ok">${P.icon('check', 16, 2.4)}</span><span>Fixed. I’ll remember.</span>`;
  setTimeout(() => el.remove(), 1500);
};

V.onEvent = (type, d) => {
  if (!S) return;
  const p = pill();
  if (type === 'rec-start') { clearTimeout(S.pillTimer); if (S.overlayQ) closeAsk('later'); p.innerHTML = `<span class="bars">${'<span></span>'.repeat(7)}</span>`; p.classList.add('show'); }
  if (type === 'processing') p.innerHTML = `<span class="rc-dots" style="display:flex;gap:4px">${'<span style="width:5px;height:5px;border-radius:50%;background:var(--glass-ink2)"></span>'.repeat(3)}</span>`;
  if (type === 'inserted') {
    const t = d.take;
    const auto = t.segs.filter((s) => s.status === 'auto');
    if (auto.length) P.step('again');
    // What it would ask: mistakes it can explain. Non-words first.
    const cands = P.mistakes(t).map(([s, i]) => ({ s, i })).sort((a, b) => (/^[A-Z]{3,}$/.test(b.s.h) ? 1 : 0) - (/^[A-Z]{3,}$/.test(a.s.h) ? 1 : 0));
    const qs = cands.map(({ s, i }) => q({ take: t, seg: i, heard: s.h, meant: s.m, app: t.app, at: t.at }));
    if (qs.length) { P.step('asked'); p.classList.remove('show'); showAsk(qs[0]); }
    else if (auto.length) { p.innerHTML = `<span style="color:var(--ok)">${P.icon('check', 15, 2.4)}</span><span>Wrote ${auto.map((s) => `<b>${P.esc(s.text)}</b>`).join(', ')}, as you told me</span>`; S.pillTimer = setTimeout(() => p.classList.remove('show'), 2400); }
    else p.classList.remove('show');
    render(true);
  }
  if (type === 'focus') render();
};

V.onFixKey = () => { if (S.overlayQ) { answer(S.overlayQ, 'yes'); P.step('yes'); } };

V.onKey = (e, k) => {
  if (!S) return false;
  if (S.overlayQ && e.key === 'Escape') { answer(S.overlayQ, 'no'); return true; }
  if (k.inField || e.metaKey || e.ctrlKey) return false;
  if (P.st.front === 'tess') {
    const it = active(); if (!it) return false;
    const key = e.key.toLowerCase();
    if (key === 'y') { answer(it, 'yes'); return true; }
    if (key === 'n') { answer(it, 'no'); return true; }
    if (key === 'e') { it.typing = true; render(); return true; }
  }
  return false;
};

const toggleWords = () => {
  const host = S.ctx.tools.querySelector('#am-wbtn').parentElement;
  const old = host.querySelector('.am-words'); if (old) { old.remove(); return; }
  const rules = P.st.rules.filter((r) => r.on);
  const pop = P.el(`<div class="am-words glass"><h4>${rules.length} words you taught me</h4>${rules.map((r) => `<div><span>${P.esc(r.meant)}</span><small>${P.esc(r.heard)}</small><button data-r="${r.id}">Forget</button></div>`).join('') || '<div><small>Nothing yet.</small></div>'}</div>`);
  host.appendChild(pop);
  pop.onclick = (e) => { const b = e.target.closest('button'); if (!b) return; const r = P.st.rules.find((x) => x.id === +b.dataset.r); if (r) P.unlearn(r); pop.remove(); toggleWords(); };
};

P.register(V);
})();
