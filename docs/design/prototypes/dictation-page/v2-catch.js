/* PROTOTYPE, THROWAWAY. Variant 2, Catch, refined after the owner picked it (2026-10-03).
   See it before it lands. Words stream into the lens while you talk: the newest are still settling
   (grey), learned words flip to your spelling, and when you let go the final pass lands and may
   change a word or two. Tap ⇧ while talking and the take waits in the lens. To fix a word, type the
   word you meant: the lens finds the word that sounds like it, and ↩ fixes, pastes and learns.
   Missed one after it pasted? Tab (⌃⌥Space) reopens the last take for the same fix. Fixing a
   learned word back teaches an exception for that app. The page is the catch record. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'catch', num: 2, name: 'Catch',
  thesis: 'See it before it lands. Words stream in as you talk and learned ones flip. To fix one, type the word you meant: it finds the one that sounds like it.',
  finalMs: 750,
  steps: [
    { id: 'flip', text: 'Hold <b>`</b> and talk. SRACT turns into Tesseract as you speak; grey words are still settling' },
    { id: 'hold', text: 'Talk again and tap <b>⇧</b> before you let go. The take waits in the lens' },
    { id: 'fixed', text: 'Type <b>claude</b>, then <b>↩</b>. It finds Cloud, fixes it, pastes and learns' },
    { id: 'after', text: 'One slipped through? <b>Tab</b>, type the word, <b>↩</b>. Fixed in Terminal' },
    { id: 'except', text: 'Click Notes and dictate. Claude flips in where you meant cloud: <b>Tab</b>, type <b>cloud</b>, <b>↩</b>' },
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
.ca h3{margin:0 0 12px;font:600 13px/1 var(--sans);color:var(--ink2);display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.ca h3 small{font:12px var(--sans);color:var(--ink3)}
.ca-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ca-tile{position:relative;border-radius:14px;background:var(--fill);padding:13px 16px 12px;display:grid;gap:6px;box-shadow:inset 0 0 0 1px var(--line)}
.ca-tile.new{animation:catile 1.4s ease}
@keyframes catile{from{box-shadow:inset 0 0 0 2px var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--fill))}}
.ca-tile .top{display:flex;align-items:baseline;gap:9px;min-width:0}
.ca-tile .w{font:600 17px/1.2 var(--display);letter-spacing:-.01em;white-space:nowrap}
.ca-tile .heard{font:12.5px/1.2 var(--mono);color:var(--ink3);text-decoration:line-through;text-decoration-color:color-mix(in srgb,var(--danger) 55%,transparent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.ca-tile .when{margin-left:auto;font:11.5px/1 var(--sans);color:var(--ink3);white-space:nowrap}
.ca-tile svg{display:block;width:100%;height:48px;overflow:visible}
.ca-tile .stat{font:12.5px/1.35 var(--sans);color:var(--ink2);padding-right:46px}
.ca-tile .stat b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.ca-tile .stat .acc{color:var(--accent-text)}
.ca-tile .ex{color:var(--ink)}
.ca-tile .forget{position:absolute;right:10px;bottom:9px;border:0;background:transparent;color:var(--ink3);font:500 11.5px/1 var(--sans);cursor:pointer;opacity:0;transition:opacity .15s;padding:3px}
.ca-tile:hover .forget,.ca-tile .forget:focus-visible{opacity:1}
.ca-tile .forget:hover{color:var(--danger)}
.ca-gone{display:flex;align-items:center;gap:10px;margin:-2px 0 12px;font:13px/1.3 var(--sans);color:var(--ink2)}
.ca-gone button{border:0;background:transparent;color:var(--accent-text);font:600 13px var(--sans);cursor:pointer;padding:0}
.ca-list{list-style:none;margin:0 -10px;padding:0}
.ca-list li{display:grid;grid-template-columns:82px 1fr auto;gap:14px;padding:10px;border-radius:10px;font:15px/1.45 var(--sans);cursor:pointer;position:relative}
.ca-list li + li::before{content:"";position:absolute;left:10px;right:10px;top:0;border-top:1px solid var(--line)}
.ca-list li:hover{background:var(--fill)}
.ca-list li:hover::before,.ca-list li:hover + li::before{opacity:0}
.ca-list .when{font:13px/1.45 var(--sans);color:var(--ink3);font-variant-numeric:tabular-nums}
.ca-list .tag{font:11.5px/1.6 var(--sans);color:var(--ink3);white-space:nowrap}
.ca-list .tag.checked{color:var(--accent-text)}
.ca-list .tag.kept{color:var(--ink)}
.ca-list .seg.auto,.ca-list .seg.fixed{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 75%,transparent)}
.ca-list li.empty{cursor:default}.ca-list li.empty:hover{background:transparent}
.ca-menu{position:relative}
.ca-pop{position:absolute;right:0;top:36px;width:250px;border-radius:14px;padding:6px;display:flex;flex-direction:column;z-index:10}
.ca-pop button{text-align:left;border:0;background:transparent;border-radius:8px;padding:8px 10px;font:13px/1.35 var(--sans);color:var(--glass-ink);cursor:pointer;display:grid;grid-template-columns:16px 1fr;gap:6px}
.ca-pop button:hover{background:color-mix(in srgb,var(--accent) 16%,transparent)}
.ca-pop small{grid-column:2;color:var(--glass-ink2);font-size:11.5px}

/* the lens */
.ca-lens{position:absolute;left:50%;bottom:34px;transform:translateX(-50%);width:max-content;min-width:360px;max-width:640px;border-radius:26px;padding:12px 18px 14px;display:flex;flex-direction:column;gap:8px;transition:box-shadow .2s,opacity .25s,transform .3s cubic-bezier(.2,1.1,.4,1);animation:calin .22s cubic-bezier(.2,1.2,.4,1)}
@keyframes calin{from{opacity:0;transform:translateX(-50%) translateY(12px) scale(.94)}}
.ca-lens.out{opacity:0;transform:translateX(-50%) translateY(8px) scale(.96);pointer-events:none}
.ca-lens[data-mode="held"],.ca-lens[data-mode="review"],.ca-lens[data-mode="after"]{box-shadow:var(--glass-shadow),0 0 0 2px var(--accent)}
.ca-top{display:flex;align-items:center;gap:10px;font:500 12px/1 var(--sans);color:var(--glass-ink2);white-space:nowrap;min-height:16px}
.ca-top .lvl{display:flex;gap:2.5px;align-items:center;height:14px}
.ca-top .lvl span{width:2.5px;height:3px;border-radius:2px;background:var(--danger)}
.ca-top .hint{margin-left:auto;display:inline-flex;align-items:center;gap:6px}
.ca-top .held{margin-left:auto;color:var(--accent-text);font-weight:600;display:inline-flex;gap:6px;align-items:center}
.ca-top .caught{color:var(--accent-text);font-weight:600}
.ca-top .ok{color:var(--ok);display:inline-flex}
.ca-top .note{margin-left:auto;color:var(--glass-ink);display:inline-flex;gap:6px;align-items:baseline}
.ca-top .note s{font:500 11.5px/1 var(--mono);color:var(--glass-ink2);text-decoration-color:var(--danger)}
.ca-top .note b{color:var(--accent-text);font-weight:600}
.ca-top button{border:0;background:transparent;color:var(--accent-text);font:600 12px var(--sans);cursor:pointer;padding:0 0 0 4px}
.ca-words{font:500 18px/1.5 var(--sans);color:var(--glass-ink);letter-spacing:-.008em;min-height:27px;max-width:604px;text-wrap:pretty}
.ca-words .tok{display:inline-block;border-radius:6px;padding:0 2px;margin:0 -2px;position:relative;transition:opacity .4s ease}
.ca-words.live .tok{animation:catok .24s ease both}
@keyframes catok{from{opacity:0;transform:translateY(4px);filter:blur(3px)}}
.ca-words .tok.vol{opacity:.42}
.ca-lens[data-mode="review"] .ca-words .tok,.ca-lens[data-mode="after"] .ca-words .tok{cursor:pointer}
.ca-words .tok.auto,.ca-words .tok.fixed{color:var(--accent-text)}
.ca-words .tok.fixed{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 70%,transparent);border-radius:2px}
.ca-words .tok.sel{background:color-mix(in srgb,var(--accent) 20%,transparent);box-shadow:0 0 0 1.5px var(--accent)}
.ca-words .tok.edit{background:var(--raise);box-shadow:0 0 0 1.5px var(--accent),0 2px 8px rgba(0,0,0,.12);color:var(--ink)}
.ca-words .gh,.ca-typed .gh{color:var(--ink3)}
.cr{display:inline-block;width:1.5px;height:1em;vertical-align:-2px;background:var(--accent);animation:blink 1s steps(1) infinite}
.ca-words .flip{display:inline-grid;perspective:260px;vertical-align:bottom}
.ca-words .flip > span{grid-area:1/1;backface-visibility:hidden;text-align:center}
.ca-words .flip .o{animation:caold .5s cubic-bezier(.5,0,.6,1) .28s both;color:var(--glass-ink2)}
.ca-words .flip .n{animation:canew .5s cubic-bezier(.2,.8,.3,1.1) .5s both;color:var(--accent-text)}
@keyframes caold{to{transform:rotateX(90deg);opacity:0}}
@keyframes canew{from{transform:rotateX(-90deg);opacity:0}to{transform:none;opacity:1}}
.ca-words .settle{display:inline-grid;vertical-align:bottom}
.ca-words .settle > span{grid-area:1/1;white-space:nowrap}
.ca-words .settle .o{animation:casout .4s ease .05s both}
.ca-words .settle .n{animation:casin .45s ease .2s both}
@keyframes casout{to{opacity:0;transform:translateY(-5px);filter:blur(2px)}}
@keyframes casin{from{opacity:0;transform:translateY(5px);filter:blur(2px)}}
.ca-foot{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font:12px/1 var(--sans);color:var(--glass-ink2)}
.ca-foot span{display:inline-flex;gap:5px;align-items:center}
.ca-foot s{font:500 11.5px/1 var(--mono);text-decoration-color:var(--danger)}
.ca-foot .kc{font-size:10.5px;min-width:18px;height:17px}
.ca-typed{font:500 14px/1 var(--sans);color:var(--glass-ink);padding:4px 7px;border-radius:7px;background:var(--raise);box-shadow:0 0 0 1.5px var(--accent)}
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
const BEFORE_GUESS = { cloud: 23, 'cloud.md': 6, 'work tree': 5, apr: 4, srax: 3, tsrac: 2, 'whisper flow': 3 };
const MODES = [
  { id: 'shift', label: 'When I tap ⇧', note: 'Paste right away unless you tap ⇧ while talking.' },
  { id: 'always', label: 'Always', note: 'Every take waits in the lens for ↩.' },
  { id: 'never', label: 'Never', note: 'Always paste. Tab still reopens the last take.' },
];
/* Ordinary words never become rules: fixing one fixes that take only. */
const COMMON = new Set('a an and are as at be but by can do for from had has have he her his i if in into is it its just me my no not of on or our out so than that the their them then there they this to too up us was we were what when which who why will with would you your'.split(' '));

/* ---------- sounds like (the real app would use a proper phonetic key) ---------- */
const NUM = { zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9' };
const soundKey = (s) => {
  let t = String(s).toLowerCase().replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine)\b/g, (m) => NUM[m]).replace(/[^a-z0-9]/g, '');
  t = t.replace(/ph/g, 'f').replace(/wh/g, 'w').replace(/ck/g, 'k').replace(/c(?=[eiy])/g, 's').replace(/[cq]/g, 'k').replace(/x/g, 'ks').replace(/z/g, 's');
  return t ? (t[0] + t.slice(1).replace(/[aeiouy]/g, '')).replace(/(.)\1+/g, '$1') : '';
};
const lev = (a, b) => {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = d[j];
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return d[b.length];
};
const sounds = (a, b) => { const x = soundKey(a), y = soundKey(b); return x && y ? 1 - lev(x, y) / Math.max(x.length, y.length) : 0; };
const bare = (s) => s.replace(/^[^\w]+/, '').replace(/[^\w]+$/, '');

V.mount = (ctx) => {
  S = { ctx, mode: 'shift', review: null, holdReq: false, stats: {}, fixedDays: [0, 0, 0, 0, 0, 0, 0], fresh: null, gone: null, landedTimer: 0, quiet: false };
  SEED.forEach((s) => {
    const r = P.learn({ heard: s.heard, meant: s.meant, source: 'fix' });
    r.at = new Date(Date.now() - s.learned * DAY); r.except = [];
    S.stats[s.meant] = { before: s.before, after: s.after.slice(), learned: s.learned };
    S.fixedDays[6 - s.learned]++;
  });
  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>
    <div class="ca-menu"><button class="gbtn" id="ca-mode">${P.icon('eye', 14)}Check before pasting: <b id="ca-mode-l" style="font-weight:600">When I tap ⇧</b>${P.icon('down', 12)}</button></div>`;
  ctx.tools.querySelector('#ca-mode').onclick = (e) => { e.stopPropagation(); togglePop(); };
  ctx.page.onclick = onPageClick;
  S.unFrame = P.onFrame(meter);
  render();
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.landedTimer); if (S) S.ctx.page.onclick = null; S = null; };

/* Rules skip the apps where you fixed them back. */
V.findRule = (heard, take) => P.st.rules.find((r) => r.on && P.eqi(r.heard, heard) && !(take && r.except?.includes(take.app)));

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
const words = () => { // one entry per word you meant, with every way it was heard
  const by = new Map();
  P.st.rules.filter((r) => r.on && S.stats[r.meant]).forEach((r) => {
    const w = by.get(r.meant) || { meant: r.meant, heards: [], except: new Set(), rules: [], st: S.stats[r.meant] };
    if (!w.heards.some((h) => P.eqi(h, r.heard))) w.heards.push(r.heard);
    (r.except || []).forEach((a) => w.except.add(a));
    w.rules.push(r); by.set(r.meant, w);
  });
  return [...by.values()].sort((a, b) => a.st.learned - b.st.learned);
};
const weekCaught = (list) => {
  const caught = [0, 0, 0, 0, 0, 0, 0];
  list.forEach((w) => w.st.after.forEach((n, k) => { const ago = w.st.learned - k; if (ago >= 0 && ago < 7) caught[6 - ago] += n; }));
  return caught;
};
const render = () => {
  const list = words();
  const caught = weekCaught(list), fixed = S.fixedDays;
  const totalC = caught.reduce((a, b) => a + b, 0), totalF = fixed.reduce((a, b) => a + b, 0);
  const max = Math.max(4, ...caught.map((c, i) => c + fixed[i]));
  const days = [...Array(7)].map((_, i) => new Date(Date.now() - (6 - i) * DAY).toLocaleDateString('en-GB', { weekday: 'narrow' }));
  const today = P.st.takes.filter((t) => Date.now() - t.at < 3 * 3600e3);
  S.ctx.page.innerHTML = `<div class="ca">
    <section class="ca-hero">
      <p class="ca-sent">This week I caught <span class="n">${totalC}</span> mistakes before they reached an app.
        <span class="sub">You taught me <b>${list.length}</b> ${list.length === 1 ? 'word' : 'words'} with <b>${totalF}</b> ${totalF === 1 ? 'fix' : 'fixes'}. None of them has slipped through since.</span></p>
      <div class="ca-days" role="img" aria-label="Mistakes caught and fixes you made, per day">
        ${caught.map((c, i) => `<div class="ca-col${i === 6 ? ' today' : ''}"><div class="ca-bar"><i class="c" style="height:${(c / max) * 92}px"></i><i class="f" style="height:${(fixed[i] / max) * 92}px"></i></div><label>${days[i]}</label></div>`).join('')}
        <div class="ca-legend"><span><i style="background:var(--accent)"></i>Caught</span><span><i style="background:color-mix(in srgb,var(--ink) 34%,transparent)"></i>You fixed</span></div>
      </div>
    </section>
    <section>
      <h3>What it catches <small>${list.length} words · one fix each</small></h3>
      ${S.gone ? `<div class="ca-gone">Forgot ${P.esc(S.gone.meant)}. ${P.esc(S.gone.heards.join(', '))} will come through as heard. <button data-act="unforget">Undo</button></div>` : ''}
      <div class="ca-grid">${list.map(tileHTML).join('')}</div>
    </section>
    <section>
      <h3>Today <small>Click a take to fix a word in it</small></h3>
      <ol class="ca-list">${today.map(rowHTML).join('') || '<li class="empty"><span></span><span style="color:var(--ink3)">Nothing dictated yet today.</span></li>'}</ol>
    </section>
  </div>`;
  S.fresh = null;
};
const tileHTML = (w) => {
  const s = w.st;
  const slipped = s.before.reduce((a, b) => a + b, 0), got = s.after.reduce((a, b) => a + b, 0);
  const all = [...s.before, ...s.after].slice(-14);
  const learnAt = 14 - Math.min(14, s.after.length);
  const wd = 300, colW = wd / 14;
  const marks = all.map((n, k) => [...Array(Math.min(n, 4))].map((_, j) => {
    const cx = k * colW + colW / 2, cy = 40 - j * 9;
    return k >= learnAt ? `<circle cx="${cx}" cy="${cy}" r="3.3" fill="var(--accent)"/>` : `<path d="M${cx - 2.6} ${cy - 2.6}l5.2 5.2M${cx + 2.6} ${cy - 2.6}l-5.2 5.2" stroke="var(--ink3)" stroke-width="1.5" stroke-linecap="round"/>`;
  }).join('')).join('');
  const lx = learnAt * colW;
  const when = s.learned === 0 ? 'taught just now' : `taught ${new Date(Date.now() - s.learned * DAY).toLocaleDateString('en-GB', { weekday: 'short' })}`;
  const ex = [...w.except].map((a) => P.APPS[a].name);
  return `<div class="ca-tile${w.meant === S.fresh ? ' new' : ''}">
    <div class="top"><span class="w">${P.esc(w.meant)}</span><span class="heard" title="Heard as ${P.esc(w.heards.join(', '))}">${P.esc(w.heards.join(', '))}</span><span class="when">${when}</span></div>
    <svg viewBox="0 0 ${wd} 48" preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" y1="45.5" x2="${wd}" y2="45.5" stroke="var(--line)" />
      <line x1="${lx}" y1="2" x2="${lx}" y2="46" stroke="var(--accent)" stroke-dasharray="2 3" />
      ${marks}
    </svg>
    <div class="stat"><b>${slipped}</b> got through before · <b class="acc">${got}</b> caught since${ex.length ? ` · <span class="ex">left alone in ${P.esc(ex.join(', '))}</span>` : ''}</div>
    <button class="forget" data-act="forget" data-meant="${P.esc(w.meant)}">Forget</button>
  </div>`;
};
const rowHTML = (t) => {
  const caught = caughtIn(t);
  const tag = t.kept && !t.inserted ? '<span class="tag kept">not pasted</span>'
    : t.checked ? '<span class="tag checked">checked</span>'
    : caught ? `<span class="tag">${caught} caught</span>` : '<span></span>';
  const text = t.segs.map((s, i) => {
    if (s.h == null) return P.esc(s.text);
    const cls = s.rule && (s.status === 'auto' || s.status === 'overfix') ? ' auto' : s.status === 'fixed' ? ' fixed' : '';
    return P.segHTML(t, s, i).replace('class="seg', `class="seg${cls}`).replace('<span ', `<span ${cls ? `title="Heard ${P.esc(s.h)}" ` : ''}`);
  }).join('');
  return `<li data-take="${t.id}"><span class="when">${P.ago(t.at)}</span><span>${text}</span>${tag}</li>`;
};
const onPageClick = (e) => {
  const act = e.target.closest('[data-act]');
  if (act?.dataset.act === 'forget') {
    const w = words().find((x) => x.meant === act.dataset.meant); if (!w) return;
    w.rules.forEach(P.unlearn);
    S.gone = { meant: w.meant, heards: w.heards, rules: w.rules };
    render(); return;
  }
  if (act?.dataset.act === 'unforget') {
    S.gone?.rules.forEach((r) => { r.on = true; });
    S.fresh = S.gone?.meant; S.gone = null; render(); return;
  }
  const li = e.target.closest('li[data-take]');
  if (!li || S.review?.mode === 'review') return; // a take waiting to paste keeps the lens
  const t = P.take(+li.dataset.take);
  if (t) startReview(t, t.kept && !t.inserted ? 'review' : 'after');
};

/* ---------- the lens ---------- */
const lensEl = () => S.ctx.overlay.querySelector('.ca-lens');
const appName = (take) => P.APPS[take?.app || P.st.target].name;
const holdHint = () => (S.holdReq || S.mode === 'always'
  ? `<span class="held">${P.icon('eye', 13, 2)}Waits for ↩</span>`
  : S.mode === 'shift' ? '<span class="hint"><span class="kc">⇧</span> check before pasting</span>' : '');
const setTop = (html) => { const l = lensEl(); if (l) l.querySelector('.ca-top').innerHTML = html; };
const openLens = (mode) => {
  clearTimeout(S.landedTimer);
  let l = lensEl();
  if (!l) { l = P.el('<div class="ca-lens glass"><div class="ca-top"></div><div class="ca-words"></div><div class="ca-foot" hidden></div></div>'); S.ctx.overlay.appendChild(l); }
  l.classList.remove('out');
  l.dataset.mode = mode;
  l.onpointerdown = () => { if (P.st.phase === 'recording') hold(); };
  return l;
};
const closeLens = (ms = 0) => {
  clearTimeout(S.landedTimer);
  S.landedTimer = setTimeout(() => { const l = lensEl(); if (!l) return; l.classList.add('out'); setTimeout(() => l.classList.contains('out') && l.remove(), 300); }, ms);
};
const meter = (lv) => {
  if (!S) return;
  lensEl()?.querySelectorAll('.lvl span').forEach((b, i) => { b.style.height = `${3 + lv * 11 * (0.5 + 0.5 * Math.abs(Math.sin(i * 1.9 + performance.now() / 160)))}px`; });
};
const hold = () => {
  if (S.holdReq || S.mode === 'never') return;
  S.holdReq = true;
  const l = lensEl(); if (!l) return;
  l.dataset.mode = 'held';
  const h = l.querySelector('.ca-top .hint, .ca-top .held');
  if (h) h.outerHTML = holdHint();
};
const liveTop = (take) => `<span class="lvl">${'<span></span>'.repeat(6)}</span><span>Listening · ${appName(take)}</span><span class="caught" id="ca-c"></span>${holdHint()}`;

/* While you talk: the preview's words, flips for learned ones. */
const liveTokHTML = (tok, take) => {
  if (tok.term) {
    const r = P.findRule(tok.heard, take);
    if (r) return `<span class="tok flip"><span class="o">${P.esc(tok.heard)}</span><span class="n">${P.esc(P.fitCase(r.meant, tok.seg === 0))}</span></span>`;
    return `<span class="tok">${P.esc(tok.heard)}</span>`;
  }
  return `<span class="tok">${P.esc(tok.pv ?? tok.text)}</span>`;
};
/* After the final pass: what will paste, with the preview's misses settling into place. */
const settledHTML = (take, animate) => P.joinTokens(P.tokens(take), (t) => {
  const seg = take.segs[t.seg];
  if (t.term && seg.rule && (seg.status === 'auto' || seg.status === 'overfix')) return `<span class="tok auto" title="Heard ${P.esc(seg.h)}">${P.esc(t.text)}</span>`;
  if (animate && t.pv != null && t.pv !== t.text) return `<span class="tok settle"><span class="o">${P.esc(t.pv)}</span><span class="n">${P.esc(t.text)}</span></span>`;
  return `<span class="tok">${P.esc(t.text)}</span>`;
});
const caughtIn = (take) => take.segs.filter((s) => s.rule && (s.status === 'auto' || s.status === 'overfix')).length;

V.onEvent = (type, d) => {
  if (!S) return;
  if (type === 'rec-start') {
    S.holdReq = false;
    if (S.review) { S.review = null; }
    const l = openLens('live');
    l.querySelector('.ca-top').innerHTML = liveTop(d.take);
    const w = l.querySelector('.ca-words'); w.className = 'ca-words live'; w.innerHTML = '';
    l.querySelector('.ca-foot').hidden = true;
  }
  if (type === 'partial') {
    const l = lensEl(); if (!l) return;
    const tok = d.take.toks[d.shown - 1];
    const w = l.querySelector('.ca-words');
    w.insertAdjacentHTML('beforeend', (d.shown > 1 && !tok.glue ? ' ' : '') + liveTokHTML(tok, d.take));
    const toks = [...w.children];
    toks.forEach((t, k) => t.classList.toggle('vol', k >= toks.length - 2 && !t.classList.contains('flip')));
    const n = d.take.toks.slice(0, d.shown).filter((t) => t.term && P.findRule(t.heard, d.take)).length;
    const c = l.querySelector('#ca-c'); if (c) c.textContent = n ? `${n} caught` : '';
    if (n) P.step('flip');
  }
  if (type === 'processing') {
    const l = lensEl(); if (!l) return;
    l.querySelectorAll('.tok.vol').forEach((t) => t.classList.remove('vol'));
    const n = l.querySelector('#ca-c')?.textContent || '';
    setTop(`<span class="ca-dots"><i></i><i></i><i></i></span><span>Finishing</span><span class="caught">${P.esc(n)}</span>${S.holdReq || S.mode === 'always' ? holdHint() : ''}`);
  }
  if (type === 'final') {
    const t = d.take;
    t.segs.forEach((s) => { if (!s.rule) return; const r = P.st.rules.find((x) => x.id === s.rule); const st = r && S.stats[r.meant]; if (st) st.after[st.after.length - 1]++; });
    const l = lensEl();
    if (l) { const w = l.querySelector('.ca-words'); w.className = 'ca-words'; w.innerHTML = settledHTML(t, true); }
    if (d.held) { P.step('hold'); startReview(t, 'review', { settled: true }); }
  }
  if (type === 'inserted') {
    if (S.quiet) { S.quiet = false; return; }
    if (S.review) return;
    const t = d.take, n = caughtIn(t);
    openLens('landed');
    setTop(`<span class="ok">${P.icon('check', 15, 2.4)}</span><span style="color:var(--glass-ink)">Pasted into ${appName(t)}</span>${n ? `<span class="caught">${n} caught</span>` : ''}<span class="hint">Missed one? <span class="kc">Tab</span></span>`);
    closeLens(2800);
    render();
  }
};

V.beforeInsert = () => !(S.holdReq || S.mode === 'always');

/* ---------- fixing: type the word you meant, it finds the one that sounds like it ---------- */
const complete = (typed) => { // the whole word it would complete to, exact case first
  if (!typed) return null;
  const pool = [...new Set([...P.st.rules.filter((r) => r.on).map((r) => r.meant), ...P.WORDS.map((w) => w.w)])];
  const exact = pool.find((w) => P.eqi(w, typed));
  if (exact) return exact.length === typed.length ? exact : null;
  return pool.find((w) => w.startsWith(typed) && w.length > typed.length)
    || pool.find((w) => w.toLowerCase().startsWith(typed.toLowerCase()) && w.length > typed.length) || null;
};
const suggestion = (typed) => { const hit = complete(typed); return hit && hit.length > typed.length ? hit.slice(typed.length) : ''; };
const wordOf = (typed) => (complete(typed) || typed).trim();
const isWord = (t) => /[a-z0-9]/i.test(t.text);
const bestTarget = (R) => {
  const word = wordOf(R.typed);
  let best = null, bestScore = 0;
  R.toks.forEach((t, i) => {
    if (!isWord(t) || P.eqi(bare(t.text), word)) return;
    const s = sounds(t.text, word);
    if (s < 0.5) return;
    const score = s + 0.05 * (1 - lev(t.text.toLowerCase(), word.toLowerCase()) / Math.max(t.text.length, word.length));
    if (score > bestScore) { best = i; bestScore = score; }
  });
  return best;
};

const startReview = (take, mode, opt = {}) => {
  S.review = { take, mode, toks: P.tokens(take), target: null, manual: false, typed: '', fixes: [], fixed: [], note: '', settle: !!opt.settled };
  const l = openLens(mode);
  l.querySelector('.ca-foot').hidden = false;
  drawReview();
};
const reviewTop = (R) => {
  const app = appName(R.take);
  const where = R.mode === 'review' ? `Not pasted yet · ${app}` : R.take.live ? `Last take · still in ${app}` : `${P.ago(R.take.at)} · sent to ${app}`;
  return `<span>${P.icon(R.mode === 'review' ? 'eye' : 'undo', 13, 2)}</span><span>${where}</span>${R.note ? `<span class="note">${R.note}</span>` : R.mode === 'review' ? '<span class="held">Waiting for you</span>' : ''}`;
};
const doneLabel = (R) => (R.mode === 'review' ? 'fix and paste' : R.take.live ? `fix in ${appName(R.take)}` : 'fix and learn');
const footHTML = (R) => {
  if (!R.typed) {
    return `<span>Type the word you meant</span><span><span class="kc">←</span><span class="kc">→</span> pick</span><span><span class="kc">↩</span> ${R.mode === 'review' ? 'paste' : 'done'}</span><span><span class="kc">esc</span> ${R.mode === 'review' ? 'keep for later' : 'close'}</span>`;
  }
  if (R.target == null) {
    return `<span class="ca-typed">${P.esc(R.typed)}<span class="cr"></span><span class="gh">${P.esc(suggestion(R.typed))}</span></span><span>Nothing here sounds like it</span><span><span class="kc">←</span><span class="kc">→</span> pick a word</span>`;
  }
  return `<span>Replaces <s>${P.esc(R.toks[R.target].text)}</s></span><span><span class="kc">⇥</span> fix another</span><span><span class="kc">↩</span> ${doneLabel(R)}</span><span><span class="kc">esc</span> clear</span>`;
};
const drawReview = () => {
  const R = S.review; const l = lensEl(); if (!R || !l) return;
  l.querySelector('.ca-top').innerHTML = reviewTop(R);
  const w = l.querySelector('.ca-words');
  w.className = 'ca-words';
  w.innerHTML = P.joinTokens(R.toks, (t, i) => {
    const seg = R.take.segs[t.seg];
    if (i === R.target && R.typed) {
      return `<span class="tok edit" data-i="${i}">${P.esc(R.typed)}<span class="cr"></span><span class="gh">${P.esc(suggestion(R.typed))}</span></span>`;
    }
    if (R.settle && t.pv != null && t.pv !== t.text) return `<span class="tok settle" data-i="${i}"><span class="o">${P.esc(t.pv)}</span><span class="n">${P.esc(t.text)}</span></span>`;
    const fixed = R.fixed.some((f) => f.seg === t.seg && (t.term || P.eqi(bare(t.text), f.word)));
    const auto = t.term && seg.rule && !fixed && (seg.status === 'auto' || seg.status === 'overfix');
    const cls = ['tok', i === R.target ? 'sel' : '', fixed ? 'fixed' : '', auto ? 'auto' : ''].filter(Boolean).join(' ');
    return `<span class="${cls}" data-i="${i}"${auto ? ` title="Heard ${P.esc(seg.h)}"` : ''}>${P.esc(t.text)}</span>`;
  });
  R.settle = false;
  w.onclick = (e) => { const tk = e.target.closest('.tok'); if (!tk) return; R.target = +tk.dataset.i; R.manual = true; drawReview(); };
  l.querySelector('.ca-foot').innerHTML = footHTML(R);
};

const setSeg = (R, k, text) => {
  const s = R.take.segs[k];
  if (R.mode === 'after') P.fixSeg(R.take, k, text);
  else { s.prev = s.text; s.text = P.fitCase(text, k === 0); s.status = P.eqi(s.text, s.m) ? 'fixed' : 'wrong'; }
};
/* One fix: change the word, and learn from it unless it was an ordinary word. */
const commitFix = () => {
  const R = S.review; if (!R?.typed) return false;
  const word = wordOf(R.typed);
  if (R.target == null || !word) return false;
  const tok = R.toks[R.target], take = R.take, seg = take.segs[tok.seg], app = take.app;
  const fix = { seg: tok.seg, word, prev: seg.text, heard: null, here: seg.h, rule: null, except: null, once: false };
  if (tok.term) {
    const applied = seg.rule ? P.st.rules.find((r) => r.id === seg.rule) : null;
    if (applied) { // a learned word that was wrong here: leave the heard form alone in this app
      applied.except = [...new Set([...(applied.except || []), app])];
      fix.except = applied; fix.once = !P.eqi(word, seg.h);
      const st = S.stats[applied.meant]; if (st) st.after[st.after.length - 1] = Math.max(0, st.after[st.after.length - 1] - 1);
      seg.rule = null;
      setSeg(R, tok.seg, word);
    } else {
      const r = P.learn({ heard: seg.h, meant: word, source: 'fix' });
      r.except = (r.except || []).filter((a) => a !== app);
      fix.rule = r; fix.heard = seg.h;
      setSeg(R, tok.seg, word);
      take.segs.forEach((x, k) => { // the same mishearing elsewhere in this take
        if (k !== tok.seg && x.h != null && P.eqi(x.h, seg.h) && x.text === x.h) { setSeg(R, k, word); R.fixed.push({ seg: k, word }); }
      });
    }
  } else {
    const heard = bare(tok.text);
    if (sounds(heard, word) >= 0.5 && !COMMON.has(heard.toLowerCase())) {
      const r = P.learn({ heard, meant: word, source: 'fix' }); r.except = r.except || [];
      fix.rule = r; fix.heard = heard;
    } else fix.once = true;
    seg.text = seg.text.replace(tok.text, tok.text.replace(heard, word));
    if (seg.pv != null) seg.pv = seg.text;
    P.renderTargets();
  }
  R.fixes.push(fix); R.fixed.push({ seg: tok.seg, word });
  R.note = noteFor(fix, app);
  R.typed = ''; R.target = null; R.manual = false;
  R.toks = P.tokens(take);
  return true;
};
const noteFor = (f, app) => {
  if (f.rule) return `Learned <s>${P.esc(f.heard)}</s> → <b>${P.esc(f.rule.meant)}</b>`;
  if (f.except) return f.once ? `${P.esc(f.except.meant)} left alone in ${P.esc(P.APPS[app].name)}` : `In ${P.esc(P.APPS[app].name)}, <b>${P.esc(f.here)}</b> stays ${P.esc(f.here)}`;
  return 'Fixed here only';
};

const finishReview = () => {
  const R = S.review; if (!R) return;
  if (R.typed && !commitFix()) return;
  S.review = null;
  if (R.mode === 'after' && !R.fixes.length) { closeLens(0); return; }
  const take = R.take; take.checked = true;
  R.fixes.forEach((f) => {
    if (f.rule && !S.stats[f.rule.meant]) S.stats[f.rule.meant] = { before: spread(BEFORE_GUESS[f.heard.toLowerCase()] ?? 2), after: [0], learned: 0 };
    if (f.rule || f.except) S.fixedDays[6]++;
  });
  S.fresh = R.fixes.find((f) => f.rule)?.rule.meant ?? R.fixes.find((f) => f.except)?.except.meant ?? null;
  if (R.mode === 'review') { take.kept = false; S.quiet = true; P.insert(take); }
  if (R.fixes.length && R.mode === 'review') P.step('fixed');
  if (R.fixes.some((f) => f.rule) && R.mode === 'after') P.step('after');
  if (R.fixes.some((f) => f.except)) P.step('except');
  const l = openLens('landed');
  l.querySelector('.ca-foot').hidden = true;
  const w = l.querySelector('.ca-words'); w.className = 'ca-words'; w.innerHTML = settledHTML(take, false);
  const head = R.mode === 'review' ? `Pasted into ${appName(take)}` : take.live ? `Fixed in ${appName(take)}` : `${appName(take)} already sent it`;
  const first = R.fixes.find((f) => f.rule || f.except);
  const more = R.fixes.filter((f) => f.rule || f.except).length - 1;
  setTop(`<span class="ok">${P.icon('check', 15, 2.4)}</span><span style="color:var(--glass-ink)">${head}</span>${first ? `<span class="note">${noteFor(first, take.app)}${more > 0 ? ` +${more}` : ''}<button id="ca-undo">Undo</button></span>` : ''}`);
  const u = l.querySelector('#ca-undo');
  if (u) u.onclick = () => undo(R);
  closeLens(first ? 4600 : 1800);
  render();
};
const undo = (R) => {
  R.fixes.slice().reverse().forEach((f) => {
    if (f.rule) P.unlearn(f.rule);
    if (f.except) f.except.except = (f.except.except || []).filter((a) => a !== R.take.app);
    if (f.rule || f.except) S.fixedDays[6] = Math.max(0, S.fixedDays[6] - 1);
    if (R.mode === 'after' && R.take.live) P.fixSeg(R.take, f.seg, f.prev, 'wrong');
  });
  setTop(`<span>${P.icon('undo', 13, 2)}</span><span style="color:var(--glass-ink)">Undone. It won't learn that.</span>`);
  closeLens(1400);
  render();
};
const spread = (n) => { const a = new Array(13).fill(0); for (let k = 0; k < n; k++) a[(k * 5 + 3) % 13]++; return a; };
const closeReview = () => {
  const R = S.review; if (!R) return;
  S.review = null;
  if (R.mode === 'review') {
    R.take.kept = true;
    openLens('landed').querySelector('.ca-foot').hidden = true;
    setTop(`<span>${P.icon('clock', 13, 2)}</span><span style="color:var(--glass-ink)">Kept, not pasted.</span><span class="hint"><span class="kc">Tab</span> brings it back</span>`);
    closeLens(2200);
    render();
  } else closeLens(0);
};

V.onFixKey = () => {
  if (S.review) return;
  const t = P.st.takes.find((x) => !x.seed && (x.inserted || x.kept)) || P.st.takes[0];
  if (t) startReview(t, t.kept && !t.inserted ? 'review' : 'after');
};

V.onKey = (e, k) => {
  if (!S) return false;
  if (P.st.phase === 'recording' && e.key === 'Shift') { hold(); return true; }
  const R = S.review; if (!R) return false;
  if (k.inField || e.metaKey || e.ctrlKey) return false;
  const retarget = () => { if (!R.manual) R.target = R.typed ? bestTarget(R) : null; };
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    const words = R.toks.map((t, i) => (isWord(t) ? i : -1)).filter((i) => i >= 0);
    const at = words.indexOf(R.target);
    R.target = at < 0 ? (e.key === 'ArrowLeft' ? words[words.length - 1] : words[0]) : words[Math.max(0, Math.min(words.length - 1, at + (e.key === 'ArrowLeft' ? -1 : 1)))];
    R.manual = true; drawReview(); return true;
  }
  if (e.key === 'Escape') { if (R.typed) { R.typed = ''; retarget(); drawReview(); } else closeReview(); return true; }
  if (e.key === 'Enter') { finishReview(); return true; }
  if (e.key === 'Tab') { if (R.typed && commitFix()) drawReview(); return true; }
  if (e.key === 'Backspace') { if (R.typed) { R.typed = R.typed.slice(0, -1); retarget(); drawReview(); } return true; }
  if (k.talkKey && e.key !== ' ') return true;
  if (e.key.length === 1) { R.typed += e.key; retarget(); drawReview(); return true; }
  return true;
};

P.register(V);
})();
