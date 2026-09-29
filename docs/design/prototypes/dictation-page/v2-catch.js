/* PROTOTYPE, THROWAWAY. Variant 2, Catch.
   See it before it lands. The overlay is a live lens: words appear as you speak, and words it has learned
   flip to your spelling in place. Tap ⇧ while talking and the take waits in the lens instead of pasting:
   ← → to pick a word, type the right one, ↩ to paste. Every fix you make there is caught next time.
   The page is the catch record: proof, per word, that the same mistake stopped coming back. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'catch', num: 2, name: 'Catch',
  thesis: 'See it before it lands. Learned words flip as you speak; tap ⇧ and the take waits for a fix before it pastes.',
  steps: [
    { id: 'flip', text: 'Hold <b>`</b> and dictate. Watch SRACT flip to Tesseract while you talk' },
    { id: 'hold', text: 'Talk again and tap <b>⇧</b> before you let go. The take waits instead of pasting' },
    { id: 'fixed', text: 'Pick a wrong word with <b>← →</b>, type the right one, <b>↩</b> to paste' },
    { id: 'after', text: 'Missed one after it pasted? <b>Tab</b> reopens the last take' },
  ],
};

V.css = `
.ca{max-width:730px;margin:0 auto;padding:8px 34px 130px;display:flex;flex-direction:column;gap:30px}
.ca-hero{display:grid;grid-template-columns:minmax(0,1fr) 236px;gap:28px;align-items:end}
.ca-sent{font:600 27px/1.22 var(--display);letter-spacing:-.022em;text-wrap:balance;margin:0}
.ca-sent .n{color:var(--accent-text);font-variant-numeric:tabular-nums}
.ca-sent .sub{display:block;font:500 14.5px/1.5 var(--sans);color:var(--ink2);letter-spacing:0;margin-top:10px;text-wrap:pretty}
.ca-sent .sub b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.ca-days{display:grid;grid-template-columns:repeat(7,1fr);gap:7px;align-items:end}
.ca-col{display:flex;flex-direction:column;align-items:stretch;gap:5px}
.ca-bar{height:92px;display:flex;flex-direction:column;justify-content:flex-end;gap:2px}
.ca-bar i{display:block;border-radius:3px;min-height:0;transition:height .5s cubic-bezier(.2,.9,.3,1)}
.ca-bar .c{background:var(--accent)}
.ca-bar .f{background:color-mix(in srgb,var(--ink) 34%,transparent)}
.ca-col label{font:500 11px/1 var(--sans);color:var(--ink3);text-align:center}
.ca-col.today label{color:var(--ink)}
.ca-legend{grid-column:1/-1;display:flex;gap:14px;font:11.5px/1 var(--sans);color:var(--ink2);margin-top:4px}
.ca-legend span{display:inline-flex;align-items:center;gap:6px}
.ca-legend i{width:9px;height:9px;border-radius:2px;display:inline-block}
.ca h3{margin:0 0 12px;font:600 13px/1 var(--sans);color:var(--ink2);display:flex;justify-content:space-between;align-items:baseline}
.ca h3 small{font:12px var(--sans);color:var(--ink3)}
.ca-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ca-tile{border-radius:14px;background:var(--fill);padding:13px 16px 12px;display:grid;gap:6px;box-shadow:inset 0 0 0 1px var(--line)}
.ca-tile.new{animation:catile 1.2s ease}
@keyframes catile{from{box-shadow:inset 0 0 0 2px var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--fill))}}
.ca-tile .top{display:flex;align-items:baseline;gap:9px}
.ca-tile .w{font:600 17px/1.2 var(--display);letter-spacing:-.01em}
.ca-tile .heard{font:12.5px/1 var(--mono);color:var(--ink3);text-decoration:line-through;text-decoration-color:color-mix(in srgb,var(--danger) 55%,transparent)}
.ca-tile .when{margin-left:auto;font:11.5px/1 var(--sans);color:var(--ink3)}
.ca-tile svg{display:block;width:100%;height:48px;overflow:visible}
.ca-tile .stat{font:12.5px/1.35 var(--sans);color:var(--ink2)}
.ca-tile .stat b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.ca-tile .stat .acc{color:var(--accent-text)}
.ca-list{list-style:none;margin:0;padding:0}
.ca-list li{display:grid;grid-template-columns:82px 1fr auto;gap:14px;padding:10px 0;border-bottom:1px solid var(--line);font:15px/1.45 var(--sans)}
.ca-list .when{font:13px/1.45 var(--sans);color:var(--ink3);font-variant-numeric:tabular-nums}
.ca-list .tag{font:11.5px/1.6 var(--sans);color:var(--ink3);white-space:nowrap}
.ca-list .tag.checked{color:var(--accent-text)}
.ca-list .seg.auto,.ca-list .seg.fixed{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 75%,transparent)}
.ca-menu{position:relative}
.ca-pop{position:absolute;right:0;top:36px;width:250px;border-radius:14px;padding:6px;display:flex;flex-direction:column;z-index:10}
.ca-pop button{text-align:left;border:0;background:transparent;border-radius:8px;padding:8px 10px;font:13px/1.35 var(--sans);color:var(--glass-ink);cursor:pointer;display:grid;grid-template-columns:16px 1fr;gap:6px}
.ca-pop button:hover{background:color-mix(in srgb,var(--accent) 16%,transparent)}
.ca-pop small{grid-column:2;color:var(--glass-ink2);font-size:11.5px}

/* the lens */
.ca-lens{position:absolute;left:50%;bottom:34px;transform:translateX(-50%);min-width:330px;max-width:680px;border-radius:24px;padding:11px 18px 13px;display:flex;flex-direction:column;gap:7px;transition:box-shadow .2s,border-color .2s,opacity .25s,transform .3s cubic-bezier(.2,1.1,.4,1);animation:calin .22s cubic-bezier(.2,1.2,.4,1)}
@keyframes calin{from{opacity:0;transform:translateX(-50%) translateY(12px) scale(.94)}}
.ca-lens.out{opacity:0;transform:translateX(-50%) translateY(8px) scale(.96);pointer-events:none}
.ca-lens[data-mode="held"],.ca-lens[data-mode="review"],.ca-lens[data-mode="after"]{box-shadow:var(--glass-shadow),0 0 0 2px var(--accent)}
.ca-top{display:flex;align-items:center;gap:10px;font:500 12px/1 var(--sans);color:var(--glass-ink2);white-space:nowrap}
.ca-top .lvl{display:flex;gap:2.5px;align-items:center;height:14px}
.ca-top .lvl span{width:2.5px;height:3px;border-radius:2px;background:var(--danger)}
.ca-top .hint{margin-left:auto;display:inline-flex;align-items:center;gap:6px}
.ca-top .held{margin-left:auto;color:var(--accent-text);font-weight:600;display:inline-flex;gap:6px;align-items:center}
.ca-top .caught{color:var(--accent-text);font-weight:600}
.ca-words{font:500 19px/1.5 var(--sans);color:var(--glass-ink);letter-spacing:-.005em;min-height:28px}
.ca-words .tok{display:inline-block;border-radius:6px;padding:0 2px;margin:0 -2px;position:relative;animation:catok .22s ease both}
.ca-words.settled .tok{animation:none}
@keyframes catok{from{opacity:0;transform:translateY(5px);filter:blur(3px)}}
.ca-words .tok.sel{background:color-mix(in srgb,var(--accent) 22%,transparent);box-shadow:0 0 0 1.5px var(--accent)}
.ca-words .tok.changed{color:var(--accent-text)}
.ca-words .tok .was{position:absolute;left:50%;bottom:100%;transform:translateX(-50%);font:500 10.5px/1 var(--mono);color:var(--glass-ink2);white-space:nowrap;text-decoration:line-through;text-decoration-color:var(--danger);padding-bottom:3px}
.ca-words .tok.edit{background:var(--raise);box-shadow:0 0 0 1.5px var(--accent),0 2px 8px rgba(0,0,0,.12);color:var(--ink)}
.ca-words .tok.edit .gh{color:var(--ink3)}
.ca-words .tok.edit .cr{display:inline-block;width:1.5px;height:1em;vertical-align:-2px;background:var(--accent);animation:blink 1s steps(1) infinite}
.ca-words .flip{display:inline-grid;perspective:260px;vertical-align:bottom}
.ca-words .flip > span{grid-area:1/1;backface-visibility:hidden;text-align:center}
.ca-words .flip .o{animation:caold .5s cubic-bezier(.5,0,.6,1) .28s both;color:var(--glass-ink2)}
.ca-words .flip .n{animation:canew .5s cubic-bezier(.2,.8,.3,1.1) .5s both;color:var(--accent-text)}
.ca-words.settled .flip .o{animation:none;opacity:0}
.ca-words.settled .flip .n{animation:none}
@keyframes caold{to{transform:rotateX(90deg);opacity:0}}
@keyframes canew{from{transform:rotateX(-90deg);opacity:0}to{transform:none;opacity:1}}
.ca-words .flip::after{content:"";grid-area:1/1;align-self:end;justify-self:center;width:4px;height:4px;border-radius:50%;background:var(--accent);transform:translateY(5px);opacity:0;animation:cadot .3s ease .9s both}
@keyframes cadot{to{opacity:1}}
.ca-foot{display:flex;gap:14px;flex-wrap:wrap;font:12px/1 var(--sans);color:var(--glass-ink2)}
.ca-foot span{display:inline-flex;gap:5px;align-items:center}
.ca-foot .kc{font-size:10.5px;min-width:18px;height:17px}
.ca-landed{display:flex;align-items:center;gap:10px;font:500 13.5px/1 var(--sans);color:var(--glass-ink);white-space:nowrap}
.ca-landed .ok{color:var(--ok)}
.ca-landed .h{font:500 12.5px/1 var(--mono);text-decoration:line-through;text-decoration-color:var(--danger);color:var(--glass-ink2)}
.ca-landed b{color:var(--accent-text);font-weight:600}
.ca-landed button{margin-left:6px;border:0;background:transparent;color:var(--accent-text);font:600 12.5px var(--sans);cursor:pointer}
.ca-dots{display:inline-flex;gap:3px}.ca-dots i{width:4px;height:4px;border-radius:50%;background:var(--glass-ink2);animation:rcdot 1s infinite}
.ca-dots i:nth-child(2){animation-delay:.15s}.ca-dots i:nth-child(3){animation-delay:.3s}
@keyframes rcdot{50%{opacity:.25}}
`;

/* ---------- seeded record: four words taught earlier this week ---------- */
const DAY = 86400e3;
const SEED = [
  { heard: 'SRACT', meant: 'Tesseract', learned: 6, before: [1, 2, 0, 3, 1, 2, 1, 0], after: [2, 1, 3, 2, 1, 2, 0] },
  { heard: 'D flash two', meant: 'DFlash2', learned: 4, before: [0, 1, 1, 0, 2, 1, 0, 1, 1, 0], after: [1, 2, 1, 1, 0] },
  { heard: 'Eleven Labs', meant: 'ElevenLabs', learned: 3, before: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], after: [0, 1, 0, 0] },
  { heard: 'test flight', meant: 'TestFlight', learned: 2, before: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], after: [1, 0, 0] },
];
const BEFORE_GUESS = { cloud: 23, 'cloud.md': 6, 'work tree': 5, apr: 4, srax: 3 };
const MODES = [
  { id: 'shift', label: 'When I tap ⇧', note: 'Paste right away unless you tap ⇧ while talking.' },
  { id: 'always', label: 'Always', note: 'Every take waits in the lens for ↩.' },
  { id: 'never', label: 'Never', note: 'Always paste. Tab still reopens the last take.' },
];

V.mount = (ctx) => {
  S = { ctx, mode: 'shift', lens: null, review: null, holdReq: false, caughtLive: 0, fixedLive: 0, stats: {}, newRule: null, landedTimer: 0 };
  SEED.forEach((s) => {
    const r = P.learn({ heard: s.heard, meant: s.meant, source: 'fix' });
    r.at = new Date(Date.now() - s.learned * DAY);
    S.stats[r.id] = { before: s.before, after: s.after.slice(), learned: s.learned };
  });
  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>
    <div class="ca-menu"><button class="gbtn" id="ca-mode">${P.icon('eye', 14)}Check before pasting: <b id="ca-mode-l" style="font-weight:600">When I tap ⇧</b>${P.icon('down', 12)}</button></div>`;
  ctx.tools.querySelector('#ca-mode').onclick = (e) => { e.stopPropagation(); togglePop(); };
  S.unFrame = P.onFrame(meter);
  render();
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.landedTimer); S = null; };

const togglePop = () => {
  const host = S.ctx.tools.querySelector('.ca-menu');
  const old = host.querySelector('.ca-pop'); if (old) { old.remove(); return; }
  const pop = P.el(`<div class="ca-pop glass">${MODES.map((m) => `<button data-m="${m.id}"><span>${m.id === S.mode ? P.icon('check', 14, 2.2) : ''}</span><span>${P.esc(m.label)}</span><small>${P.esc(m.note)}</small></button>`).join('')}</div>`);
  host.appendChild(pop);
  pop.onclick = (e) => {
    const b = e.target.closest('button'); if (!b) return;
    S.mode = b.dataset.m; S.ctx.tools.querySelector('#ca-mode-l').textContent = MODES.find((m) => m.id === S.mode).label; pop.remove();
  };
};

/* ---------- page: the catch record ---------- */
const weekCounts = () => {
  const caught = [0, 0, 0, 0, 0, 0, 0], fixed = [0, 0, 0, 0, 0, 0, 0];
  P.st.rules.forEach((r) => {
    const s = S.stats[r.id]; if (!s || !r.on) return;
    s.after.forEach((n, k) => { const ago = s.learned - k; if (ago >= 0 && ago < 7) caught[6 - ago] += n; });
    if (s.learned < 7) fixed[6 - s.learned] += 1;
  });
  return { caught, fixed };
};
const render = () => {
  const { caught, fixed } = weekCounts();
  const totalC = caught.reduce((a, b) => a + b, 0), totalF = fixed.reduce((a, b) => a + b, 0);
  const max = Math.max(4, ...caught.map((c, i) => c + fixed[i]));
  const days = [...Array(7)].map((_, i) => new Date(Date.now() - (6 - i) * DAY).toLocaleDateString('en-GB', { weekday: 'narrow' }));
  const rules = P.st.rules.filter((r) => r.on && S.stats[r.id]);
  const today = P.st.takes.filter((t) => Date.now() - t.at < 3 * 3600e3);
  S.ctx.page.innerHTML = `<div class="ca">
    <section class="ca-hero">
      <p class="ca-sent">This week I caught <span class="n">${totalC}</span> mistakes before they reached an app.
        <span class="sub">You taught me <b>${totalF}</b> ${totalF === 1 ? 'word' : 'words'}, one fix each. None of them has slipped through since.</span></p>
      <div class="ca-days" role="img" aria-label="Mistakes caught and words you fixed, per day">
        ${caught.map((c, i) => `<div class="ca-col${i === 6 ? ' today' : ''}"><div class="ca-bar"><i class="c" style="height:${(c / max) * 92}px"></i><i class="f" style="height:${(fixed[i] / max) * 92}px"></i></div><label>${days[i]}</label></div>`).join('')}
        <div class="ca-legend"><span><i style="background:var(--accent)"></i>Caught</span><span><i style="background:color-mix(in srgb,var(--ink) 34%,transparent)"></i>You fixed</span></div>
      </div>
    </section>
    <section>
      <h3>What it catches <small>${rules.length} words · one fix each</small></h3>
      <div class="ca-grid">${rules.map(tileHTML).join('')}</div>
    </section>
    <section>
      <h3>Today</h3>
      <ol class="ca-list">${today.map(rowHTML).join('') || '<li><span></span><span style="color:var(--ink3)">Nothing dictated yet today.</span></li>'}</ol>
    </section>
  </div>`;
  S.newRule = null;
};
const tileHTML = (r) => {
  const s = S.stats[r.id];
  const slipped = s.before.reduce((a, b) => a + b, 0), got = s.after.reduce((a, b) => a + b, 0);
  const all = [...s.before, ...s.after].slice(-14);
  const learnAt = 14 - s.after.length; // column index where learning happened
  const w = 300, colW = w / 14;
  const marks = all.map((n, k) => [...Array(Math.min(n, 4))].map((_, j) => {
    const cx = k * colW + colW / 2, cy = 40 - j * 9;
    return k >= learnAt ? `<circle cx="${cx}" cy="${cy}" r="3.3" fill="var(--accent)"/>` : `<path d="M${cx - 2.6} ${cy - 2.6}l5.2 5.2M${cx + 2.6} ${cy - 2.6}l-5.2 5.2" stroke="var(--ink3)" stroke-width="1.5" stroke-linecap="round"/>`;
  }).join('')).join('');
  const lx = learnAt * colW;
  const when = s.learned === 0 ? 'taught just now' : `taught ${new Date(Date.now() - s.learned * DAY).toLocaleDateString('en-GB', { weekday: 'short' })}`;
  return `<div class="ca-tile${r === S.newRule ? ' new' : ''}">
    <div class="top"><span class="w">${P.esc(r.meant)}</span><span class="heard">${P.esc(r.heard)}</span><span class="when">${when}</span></div>
    <svg viewBox="0 0 ${w} 48" preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" y1="45.5" x2="${w}" y2="45.5" stroke="var(--line)" />
      <line x1="${lx}" y1="2" x2="${lx}" y2="46" stroke="var(--accent)" stroke-dasharray="2 3" />
      ${marks}
    </svg>
    <div class="stat"><b>${slipped}</b> slipped through before you taught it · <b class="acc">${got}</b> caught since</div>
  </div>`;
};
const rowHTML = (t) => {
  const tag = t.checked ? `<span class="tag checked">checked in the lens</span>` : t.segs.some((s) => s.status === 'auto') ? `<span class="tag">${t.segs.filter((s) => s.status === 'auto').length} caught</span>` : '<span></span>';
  const text = t.segs.map((s, i) => {
    if (s.h == null) return P.esc(s.text);
    const cls = s.status === 'auto' ? ' auto' : s.status === 'fixed' ? ' fixed' : '';
    return P.segHTML(t, s, i).replace('class="seg', `class="seg${cls}`).replace('<span ', `<span ${cls ? `title="Heard ${P.esc(s.h)}" ` : ''}`);
  }).join('');
  return `<li><span class="when">${P.ago(t.at)}</span><span>${text}</span>${tag}</li>`;
};

/* ---------- the lens ---------- */
const lensEl = () => S.ctx.overlay.querySelector('.ca-lens');
const lensTop = (mode) => {
  const app = P.APPS[P.st.target].name;
  if (mode === 'live') return `<span class="lvl">${'<span></span>'.repeat(6)}</span><span>Listening · ${app}</span><span class="caught" id="ca-c"></span>${S.holdReq || S.mode === 'always' ? `<span class="held">${P.icon('eye', 13, 2)}Waits for ↩</span>` : S.mode === 'shift' ? `<span class="hint"><span class="kc">⇧</span> check before pasting</span>` : ''}`;
  if (mode === 'processing') return `<span class="ca-dots"><i></i><i></i><i></i></span><span>Writing it down</span>`;
  if (mode === 'review') return `<span>${P.icon('eye', 13, 2)}</span><span>Not pasted yet · ${app}</span><span class="held">Check it, then ↩</span>`;
  if (mode === 'after') return `<span>${P.icon('undo', 13, 2)}</span><span>Your last take · ${app}</span><span class="held">Fix it in place</span>`;
  return '';
};
const openLens = (mode) => {
  clearTimeout(S.landedTimer);
  let l = lensEl();
  if (!l) { l = P.el(`<div class="ca-lens glass"><div class="ca-top"></div><div class="ca-words"></div><div class="ca-foot" hidden></div></div>`); S.ctx.overlay.appendChild(l); }
  l.classList.remove('out');
  l.dataset.mode = mode;
  l.onpointerdown = () => { // clicking the lens while talking holds the take too
    if (P.st.phase === 'recording' && !S.holdReq) { S.holdReq = true; l.dataset.mode = 'held'; l.querySelector('.ca-top').innerHTML = lensTop('live'); }
  };
  l.querySelector('.ca-top').innerHTML = lensTop(mode);
  return l;
};
const closeLens = (ms = 0) => {
  clearTimeout(S.landedTimer);
  S.landedTimer = setTimeout(() => { const l = lensEl(); if (!l) return; l.classList.add('out'); setTimeout(() => l.classList.contains('out') && l.remove(), 300); }, ms);
};
const meter = (lv) => {
  if (!S) return;
  const bars = lensEl()?.querySelectorAll('.lvl span');
  bars?.forEach((b, i) => { b.style.height = `${3 + lv * 11 * (0.5 + 0.5 * Math.abs(Math.sin(i * 1.9 + performance.now() / 160)))}px`; });
};

const tokHTML = (tok, live) => {
  const r = tok.term && live ? P.findRule(tok.heard) : null;
  if (r) return `<span class="tok flip"><span class="o">${P.esc(tok.heard)}</span><span class="n">${P.esc(P.fitCase(r.meant, tok.seg === 0))}</span></span>`;
  return `<span class="tok">${P.esc(tok.text)}</span>`;
};
const updateCaught = (take, shown) => {
  const n = take.toks.slice(0, shown).filter((t) => t.term && P.findRule(t.heard)).length;
  const c = lensEl()?.querySelector('#ca-c');
  if (c) c.textContent = n ? `${n} caught` : '';
  if (n) P.step('flip');
};

V.onEvent = (type, d) => {
  if (!S) return;
  if (type === 'rec-start') {
    S.holdReq = false; S.review = null;
    const l = openLens('live');
    l.querySelector('.ca-words').className = 'ca-words'; l.querySelector('.ca-words').innerHTML = '';
    l.querySelector('.ca-foot').hidden = true;
  }
  if (type === 'partial') {
    const l = lensEl(); if (!l) return;
    const tok = d.take.toks[d.shown - 1];
    const w = l.querySelector('.ca-words');
    w.insertAdjacentHTML('beforeend', (d.shown > 1 && !tok.glue ? ' ' : '') + tokHTML(tok, true));
    updateCaught(d.take, d.shown);
  }
  if (type === 'processing') { const l = lensEl(); if (l) l.querySelector('.ca-top').innerHTML = lensTop('processing'); }
  if (type === 'final') {
    const t = d.take;
    const auto = t.segs.filter((s) => s.status === 'auto').length;
    S.caughtLive += auto;
    P.st.rules.forEach((r) => { const s = S.stats[r.id]; if (!s) return; const n = t.segs.filter((x) => x.rule === r.id).length; if (n) s.after[s.after.length - 1] += n; });
    if (d.held) { P.step('hold'); startReview(t, 'review'); }
  }
  if (type === 'inserted') {
    const t = d.take;
    if (S.review || S.quiet) { S.quiet = false; return; }
    const auto = t.segs.filter((s) => s.status === 'auto').length;
    const l = openLens('landed');
    l.querySelector('.ca-top').innerHTML = '';
    l.querySelector('.ca-foot').hidden = true;
    l.querySelector('.ca-words').className = 'ca-words settled';
    l.querySelector('.ca-words').innerHTML = `<div class="ca-landed"><span class="ok">${P.icon('check', 16, 2.4)}</span>Pasted${auto ? ` · <b>${auto} caught</b>` : ''}<span style="color:var(--glass-ink2);margin-left:8px">Missed one? <span class="kc">Tab</span></span></div>`;
    closeLens(2600);
    render();
  }
};

V.beforeInsert = () => !(S.holdReq || S.mode === 'always');

/* ---------- review: pick a word, type, paste ---------- */
const startReview = (take, mode) => {
  const toks = P.tokens(take);
  let sel = toks.findIndex((t) => t.term && P.isMistake(take.segs[t.seg]));
  if (sel < 0) sel = toks.length - 1;
  S.review = { take, mode, toks, sel, edit: null, changes: {} };
  const l = openLens(mode);
  const foot = l.querySelector('.ca-foot');
  foot.hidden = false;
  foot.innerHTML = `<span><span class="kc">←</span><span class="kc">→</span> pick a word</span><span>type to fix</span><span><span class="kc">↩</span> ${mode === 'review' ? 'paste' : 'replace in ' + P.APPS[take.app].name}</span><span><span class="kc">esc</span> ${mode === 'review' ? 'discard' : 'close'}</span>`;
  drawWords();
};
const complete = (typed) => { // the whole word it would complete to, exact case first
  if (!typed) return null;
  const pool = [...new Set([...P.st.rules.map((r) => r.meant), ...P.WORDS.map((w) => w.w)])];
  const exact = pool.find((w) => P.eqi(w, typed));
  if (exact) return exact.length === typed.length ? exact : null;
  return pool.find((w) => w.startsWith(typed) && w.length > typed.length)
    || pool.find((w) => w.toLowerCase().startsWith(typed.toLowerCase()) && w.length > typed.length) || null;
};
const suggestion = (typed) => { const hit = complete(typed); return hit && hit.length > typed.length ? hit.slice(typed.length) : ''; };
const drawWords = () => {
  const R = S.review; const l = lensEl(); if (!R || !l) return;
  const w = l.querySelector('.ca-words');
  w.className = 'ca-words settled';
  w.innerHTML = P.joinTokens(R.toks, (t, i) => {
    const changed = R.changes[i];
    const shown = changed ?? t.text;
    if (R.edit && i === R.sel) return `<span class="tok edit" data-i="${i}">${P.esc(R.edit.text)}<span class="cr"></span><span class="gh">${P.esc(suggestion(R.edit.text))}</span></span>`;
    const cls = ['tok', i === R.sel ? 'sel' : '', changed ? 'changed' : '', !changed && t.term && R.take.segs[t.seg].status === 'auto' ? 'changed' : ''].join(' ');
    const was = changed ? `<span class="was">${P.esc(t.text)}</span>` : '';
    return `<span class="${cls}" data-i="${i}">${was}${P.esc(shown)}</span>`;
  });
  w.onclick = (e) => { const tk = e.target.closest('.tok'); if (!tk) return; R.sel = +tk.dataset.i; R.edit = null; drawWords(); };
};
const commitEdit = () => {
  const R = S.review; if (!R?.edit) return;
  const txt = (complete(R.edit.text) || R.edit.text).trim();
  if (txt && txt !== R.toks[R.sel].text) R.changes[R.sel] = txt;
  R.edit = null; drawWords();
};
const applyChanges = () => {
  const R = S.review; const take = R.take; const learned = [];
  Object.entries(R.changes).forEach(([i, txt]) => {
    const tok = R.toks[+i]; const seg = take.segs[tok.seg];
    if (tok.term) {
      learned.push(P.learn({ heard: seg.h, meant: txt, source: 'fix' }));
      if (R.mode === 'after') P.fixSeg(take, tok.seg, txt);
      else { seg.text = P.fitCase(txt, tok.seg === 0); seg.status = P.eqi(seg.text, seg.m) ? 'fixed' : 'wrong'; }
    } else {
      learned.push(P.learn({ heard: tok.text.replace(/[.,!?]$/, ''), meant: txt.replace(/[.,!?]$/, ''), source: 'fix' }));
      seg.text = seg.text.replace(tok.text, txt);
      P.renderTargets();
    }
  });
  learned.forEach((r) => { if (!S.stats[r.id]) S.stats[r.id] = { before: spread(BEFORE_GUESS[r.heard.toLowerCase()] ?? 2), after: [0], learned: 0 }; });
  S.newRule = learned[0] || null;
  return learned;
};
const spread = (n) => { const a = new Array(13).fill(0); for (let k = 0; k < n; k++) a[(k * 5 + 3) % 13]++; return a; };
const finishReview = () => {
  const R = S.review; if (!R) return;
  commitEdit();
  const learned = applyChanges();
  const take = R.take; take.checked = true;
  S.review = null;
  if (R.mode === 'review') { S.quiet = true; P.insert(take); }
  if (learned.length) P.step(R.mode === 'after' ? 'after' : 'fixed');
  const l = openLens('landed');
  l.querySelector('.ca-top').innerHTML = '';
  l.querySelector('.ca-foot').hidden = true;
  const w = l.querySelector('.ca-words'); w.className = 'ca-words settled';
  const r = learned[0];
  w.innerHTML = `<div class="ca-landed"><span class="ok">${P.icon('check', 16, 2.4)}</span>${R.mode === 'review' ? 'Pasted' : take.live ? 'Fixed in ' + P.APPS[take.app].name : 'Learned (already sent)'}${r ? ` · next time <span class="h">${P.esc(r.heard)}</span> becomes <b>${P.esc(r.meant)}</b>${learned.length > 1 ? ` +${learned.length - 1}` : ''}<button id="ca-undo">Undo</button>` : ''}</div>`;
  const u = w.querySelector('#ca-undo'); if (u) u.onclick = () => { learned.forEach(P.unlearn); closeLens(0); render(); };
  closeLens(4200);
  render();
};
const cancelReview = () => {
  const R = S.review; if (!R) return;
  S.review = null;
  if (R.mode === 'review') {
    P.discard(R.take);
    const l = openLens('landed'); l.querySelector('.ca-top').innerHTML = ''; l.querySelector('.ca-foot').hidden = true;
    const w = l.querySelector('.ca-words'); w.className = 'ca-words settled';
    w.innerHTML = `<div class="ca-landed">${P.icon('x', 15, 2.2)}Discarded. Nothing was pasted.</div>`;
    closeLens(1600);
  } else closeLens(0);
};

V.onFixKey = () => {
  if (S.review) return;
  const t = P.lastLive() || P.st.takes.find((x) => x.inserted && !x.seed);
  if (!t) return;
  startReview(t, 'after');
};

V.onKey = (e, k) => {
  if (!S) return false;
  if (P.st.phase === 'recording' && e.key === 'Shift' && S.mode === 'shift') {
    S.holdReq = true;
    const l = lensEl(); if (l) { l.dataset.mode = 'held'; l.querySelector('.ca-top').innerHTML = lensTop('live'); }
    return true;
  }
  const R = S.review; if (!R) return false;
  if (k.inField) return false;
  if (e.metaKey || e.ctrlKey) return false;
  if (k.talkKey) return true;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { commitEdit(); R.sel = Math.max(0, Math.min(R.toks.length - 1, R.sel + (e.key === 'ArrowLeft' ? -1 : 1))); drawWords(); return true; }
  if (e.key === 'Escape') { if (R.edit) { R.edit = null; drawWords(); } else cancelReview(); return true; }
  if (e.key === 'Enter') { if (R.edit) commitEdit(); else finishReview(); return true; }
  if (e.key === 'Tab') { if (R.edit) { R.edit.text = complete(R.edit.text) || R.edit.text; drawWords(); } return true; }
  if (e.key === 'Backspace') { if (R.edit) { R.edit.text = R.edit.text.slice(0, -1); if (!R.edit.text) R.edit = null; drawWords(); } else if (R.changes[R.sel]) { delete R.changes[R.sel]; drawWords(); } return true; }
  if (e.key.length === 1) { if (!R.edit) R.edit = { text: '' }; R.edit.text += e.key; drawWords(); return true; }
  return true;
};

P.register(V);
})();
