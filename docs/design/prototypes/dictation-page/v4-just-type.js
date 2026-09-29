/* PROTOTYPE, THROWAWAY. Variant 4, Just Type It.
   Type the word you meant. It finds where it goes. Anywhere you type, ";;claude" then space fixes the word in
   your last take that sounds like it (no selecting, no window). ⌃⌥Space does the same from a small field.
   The page is one field that fixes a word everywhere: every take and memory it was misheard in, and from now on. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'just-type', num: 4, name: 'Just Type It',
  thesis: 'Type the word you meant, right where you are: “;;claude” then space. It finds what sounds like it and fixes it everywhere.',
  steps: [
    { id: 'dictate', text: 'Hold <b>`</b> and dictate into Terminal' },
    { id: 'inline', text: 'Type <b>;;claude</b> then space in Terminal. It finds “cloud” and fixes it' },
    { id: 'everywhere', text: 'In Tesseract, search <b>Tesseract</b> and fix every place it was misheard' },
    { id: 'again', text: 'Dictate again. Both come out right' },
  ],
};

V.css = `
.jt{max-width:720px;margin:0 auto;padding:8px 34px 130px;display:flex;flex-direction:column;gap:26px}
.jt-field{position:relative;display:flex;align-items:center;gap:12px;height:58px;padding:0 16px 0 18px;border-radius:17px;background:var(--fill);box-shadow:inset 0 0 0 1px var(--line);transition:box-shadow .2s,background .2s}
.jt-field:focus-within{background:var(--win-bg);box-shadow:inset 0 0 0 1.5px var(--accent),0 0 0 4px color-mix(in srgb,var(--accent) 18%,transparent)}
.jt-field .ic{color:var(--ink3)}
.jt-field input{flex:1;min-width:0;border:0;background:transparent;outline:none;font:600 24px/1 var(--display);letter-spacing:-.02em;color:var(--ink)}
.jt-field input::placeholder{color:var(--ink3);font-weight:500}
.jt-field .gh{position:absolute;left:52px;font:600 24px/1 var(--display);letter-spacing:-.02em;color:var(--ink3);pointer-events:none;white-space:pre}
.jt-field .kbd{display:flex;gap:4px;align-items:center;font:12px var(--sans);color:var(--ink3)}
.jt-under{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-top:-14px;flex-wrap:wrap}
.jt-try{display:flex;gap:6px;align-items:center;font:13px var(--sans);color:var(--ink3)}
.jt-try button{height:26px;padding:0 10px;border-radius:13px;border:1px solid var(--line);background:transparent;font:500 12.5px var(--sans);color:var(--ink2);cursor:pointer}
.jt-try button:hover{border-color:var(--ink3);color:var(--ink)}
.jt-tip{display:flex;align-items:center;gap:9px;font:13px/1.35 var(--sans);color:var(--ink2)}
.jt-tip code{font:500 13px/1 var(--mono);padding:4px 7px;border-radius:6px;background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent-text)}
.jt-body{display:flex;flex-direction:column;gap:28px}
.jt-body > .jt-sum + .jt-groups{margin-top:-10px}
.jt h3{margin:0 0 10px;font:600 13px/1 var(--sans);color:var(--ink2);display:flex;justify-content:space-between}
.jt h3 small{font:12px var(--sans);color:var(--ink3)}
.jt-sum{font:600 21px/1.3 var(--display);letter-spacing:-.015em;text-wrap:balance;margin:0}
.jt-sum em{font-style:normal;color:var(--accent-text)}
.jt-sum small{display:block;margin-top:6px;font:14px/1.45 var(--sans);letter-spacing:0;color:var(--ink2)}
.jt-groups{display:flex;flex-direction:column;gap:10px}
.jt-g{border-radius:14px;box-shadow:inset 0 0 0 1px var(--line);padding:12px 16px 10px}
.jt-g.maybe{background:transparent;opacity:.78}
.jt-g.on{background:var(--fill)}
.jt-gh{display:grid;grid-template-columns:22px auto 64px auto 1fr;gap:10px;align-items:center;cursor:pointer}
.jt-box{width:18px;height:18px;border-radius:5px;box-shadow:inset 0 0 0 1.5px var(--ink3);display:grid;place-items:center;color:var(--accent-ink)}
.jt-g.on .jt-box{background:var(--accent);box-shadow:none}
.jt-h{font:500 16px/1 var(--mono);color:var(--ink)}
.jt-g.done .jt-h{color:var(--ink3);text-decoration:line-through;text-decoration-color:var(--danger)}
.jt-link{width:64px;height:22px;overflow:visible}
.jt-link path{stroke:var(--accent);stroke-width:1.6;fill:none;stroke-dasharray:2 4;stroke-linecap:round}
.jt-g.maybe .jt-link path{stroke:var(--ink3)}
.jt-g.done .jt-link path{animation:jtdash 1s linear infinite}
@keyframes jtdash{to{stroke-dashoffset:-12}}
.jt-m{font:600 17px/1 var(--display);letter-spacing:-.01em;color:var(--accent-text)}
.jt-g.maybe .jt-m{color:var(--ink2)}
.jt-n{justify-self:end;font:12.5px var(--sans);color:var(--ink3);font-variant-numeric:tabular-nums;text-align:right}
.jt-snips{list-style:none;margin:8px 0 0 32px;padding:0;display:flex;flex-direction:column;gap:5px}
.jt-snips li{display:grid;grid-template-columns:18px 1fr auto;gap:8px;align-items:baseline;font:14px/1.45 var(--sans);color:var(--ink2)}
.jt-snips li .ic{color:var(--ink3);transform:translateY(3px)}
.jt-snips .t{font:12px var(--sans);color:var(--ink3);white-space:nowrap}
.jt-snips mark{background:none;color:var(--ink);font:500 13px var(--mono);box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 65%,transparent);padding:0 2px}
.jt-snips mark.fixed{font:600 14px var(--sans);color:var(--accent-text);box-shadow:none;animation:jtfix .8s ease}
@keyframes jtfix{from{background:color-mix(in srgb,var(--accent) 35%,transparent)}}
.jt-why{margin:6px 0 0 32px;font:12.5px var(--sans);color:var(--ink3)}
.jt-apply{display:flex;align-items:center;gap:14px;margin-top:4px}
.jt-apply button{height:38px;padding:0 18px;border-radius:19px;border:0;background:var(--accent);color:var(--accent-ink);font:600 14px var(--sans);cursor:pointer}
.jt-apply button[disabled]{opacity:.4;cursor:default}
.jt-apply span{font:13px var(--sans);color:var(--ink3)}
.jt-apply .undo{background:transparent;color:var(--accent-text);padding:0 4px}
.jt-index{display:grid;grid-template-columns:1fr 1fr;gap:0 28px}
.jt-index div{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--line);cursor:pointer}
.jt-index b{font:600 15px var(--display)}
.jt-index span{font:12.5px var(--mono);color:var(--ink3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.jt-index small{font:12px var(--sans);color:var(--ink3);font-variant-numeric:tabular-nums}
.jt-recent{list-style:none;margin:0;padding:0}
.jt-recent li{display:grid;grid-template-columns:88px 1fr;gap:14px;padding:9px 0;border-bottom:1px solid var(--line);font:15px/1.45 var(--sans)}
.jt-recent .when{font:13px var(--sans);color:var(--ink3);font-variant-numeric:tabular-nums}
.jt-recent .seg.fixed,.jt-recent .seg.auto{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 70%,transparent)}

/* overlay */
.jt-pill{position:absolute;left:50%;bottom:40px;transform:translateX(-50%) scale(.85);opacity:0;height:36px;padding:0 15px;border-radius:18px;display:flex;align-items:center;gap:9px;font:500 13px/1 var(--sans);transition:opacity .18s,transform .22s cubic-bezier(.2,1.2,.4,1);pointer-events:none;white-space:nowrap}
.jt-pill.show{opacity:1;transform:translateX(-50%) scale(1)}
.jt-pill .bars{display:flex;gap:3px;align-items:center;height:16px}
.jt-pill .bars span{width:3px;height:4px;border-radius:2px;background:var(--danger)}
.jt-card{position:absolute;left:50%;bottom:34px;transform:translateX(-50%);width:520px;border-radius:20px;padding:14px 18px 12px;display:flex;flex-direction:column;gap:10px;animation:jtin .22s cubic-bezier(.2,1.2,.4,1)}
@keyframes jtin{from{opacity:0;transform:translateX(-50%) translateY(10px) scale(.96)}}
.jt-card .line{font:500 16px/1.9 var(--sans);color:var(--glass-ink2)}
.jt-card .hit{position:relative;display:inline-block;color:var(--glass-ink);font:500 15px/1 var(--mono);padding:3px 4px;border-radius:6px;background:color-mix(in srgb,var(--accent) 20%,transparent);box-shadow:0 0 0 1.5px var(--accent)}
.jt-card .hit .gh{position:absolute;left:50%;bottom:calc(100% + 5px);transform:translateX(-50%);font:600 13px/1 var(--sans);color:var(--accent-text);white-space:nowrap}
.jt-card .foot{display:flex;gap:12px;align-items:center;font:12px/1 var(--sans);color:var(--glass-ink2)}
.jt-card .foot code{font:500 12px var(--mono);color:var(--accent-text)}
.jt-card .foot .kc{font-size:10.5px;height:17px}
.jt-card .miss{font:14px/1.4 var(--sans);color:var(--glass-ink)}
.jt-res{display:flex;align-items:center;gap:6px;font:600 20px/1 var(--display);letter-spacing:-.015em}
.jt-res .h{font:500 16px/1 var(--mono);color:var(--glass-ink2);text-decoration:line-through;text-decoration-color:var(--danger)}
.jt-res b{color:var(--accent-text);font-weight:700}
.jt-res svg{width:70px;height:26px}
.jt-res svg path{stroke:var(--accent);stroke-width:1.8;fill:none;stroke-dasharray:2 4;stroke-linecap:round;animation:jtdraw .7s ease both}
@keyframes jtdraw{from{stroke-dashoffset:40;opacity:0}}
.jt-card .sub{font:13.5px/1.45 var(--sans);color:var(--glass-ink2)}
.jt-card .acts{display:flex;gap:8px;align-items:center}
.jt-card .acts button{height:30px;border-radius:15px;border:0;padding:0 12px;font:600 12.5px/1 var(--sans);cursor:pointer;display:inline-flex;gap:7px;align-items:center;background:color-mix(in srgb,var(--glass-ink) 8%,transparent);color:var(--glass-ink)}
.jt-card .acts button.pri{background:var(--accent);color:var(--accent-ink)}
.jt-card .acts .kc{font-size:10.5px;height:17px;border-color:rgba(0,0,0,.1);background:rgba(255,255,255,.2);color:inherit}
.jt-card .fld{display:flex;align-items:center;gap:10px;height:44px;border-radius:12px;padding:0 12px;background:color-mix(in srgb,var(--glass-ink) 7%,transparent)}
.jt-card .fld span{font:600 13px var(--sans);color:var(--glass-ink2)}
.jt-card .fld input{flex:1;border:0;background:transparent;outline:none;font:600 19px var(--display);color:var(--glass-ink)}
`;

/* sound-alikes the prototype knows (the real thing would use a phonetic key) */
const EXTRA_FORMS = { Tesseract: ['struct'], Claude: ['cloud'] };
const canon = (typed) => {
  if (!typed) return null;
  const pool = [...new Set([...P.WORDS.map((w) => w.w), ...P.st.rules.map((r) => r.meant), 'a PR'])];
  return pool.find((w) => P.eqi(w, typed)) || pool.find((w) => w.startsWith(typed)) || pool.find((w) => w.toLowerCase().startsWith(typed.toLowerCase())) || null;
};
const formsOf = (W) => {
  const w = P.word(W); const f = new Set();
  if (w && !w.right) { f.add(w.heard); (w.alts || []).forEach((x) => f.add(x)); }
  (EXTRA_FORMS[W] || []).forEach((x) => f.add(x));
  P.st.rules.forEach((r) => { if (r.meant === W) f.add(r.heard); });
  P.st.takes.forEach((t) => t.segs.forEach((s) => { if (s.h != null && P.eqi(s.m, W) && !P.eqi(s.h, W)) f.add(s.h); }));
  return [...f];
};
const findIn = (take, W) => {
  if (!take || !W) return -1;
  let i = take.segs.findIndex((s) => s.h != null && P.eqi(s.m, W) && s.text === s.h && !P.eqi(s.h, W));
  if (i >= 0) return i;
  const forms = formsOf(W);
  return take.segs.findIndex((s) => s.h != null && s.text === s.h && forms.some((f) => P.eqi(f, s.h)) && !P.eqi(s.h, W));
};

V.mount = (ctx) => {
  S = { ctx, q: '', result: null, card: null, pillTimer: 0, checked: {} };
  // A few more takes and a memory, so a search has something to find.
  const add = (ago, app, line) => { const t = P.makeTake(app, line, new Date(Date.now() - ago * 60000)); t.seed = true; P.st.takes.push(t); };
  add(170, 'terminal', ['Make the settings a ', P.T('struct', 'struct'), ' instead of a class']);
  add(300, 'notes', ['Rain and low ', P.T('cloud', 'cloud'), ' all afternoon, so no run']);
  add(1500, 'terminal', ['Ask ', P.T('cloud', 'Claude'), ' to review the ', P.T('TSRAC', 'Tesseract'), ' cache change']);
  P.st.takes.sort((a, b) => b.at - a.at);
  P.learn({ heard: 'Eleven Labs', meant: 'ElevenLabs', source: 'typed' }).applied = 3;
  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>`;
  ctx.overlay.innerHTML = `<div class="jt-pill glass" id="jt-pill"></div>`;
  S.unFrame = P.onFrame((lv) => {
    ctx.overlay.querySelectorAll('.jt-pill .bars span').forEach((b, i) => { b.style.height = `${4 + lv * 12 * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7 + performance.now() / 170)))}px`; });
  });
  render();
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.pillTimer); clearTimeout(S?.cardTimer); S = null; };

/* ---------- gathering every place a word was misheard ---------- */
const gather = (W) => {
  const forms = formsOf(W); const groups = new Map();
  const put = (heard, real, entry) => {
    const key = heard.toLowerCase() + (real ? '' : '~');
    if (!groups.has(key)) groups.set(key, { key, heard, real, takes: [], mems: [] });
    groups.get(key)[entry.mem ? 'mems' : 'takes'].push(entry);
  };
  P.st.takes.forEach((t) => t.segs.forEach((s, i) => {
    if (s.h == null || s.text !== s.h || P.eqi(s.h, W) || !forms.some((f) => P.eqi(f, s.h))) return;
    put(s.h, P.eqi(s.m, W), { take: t, i });
  }));
  P.st.memories.forEach((m) => m.segs.forEach((s, i) => {
    if (s.h == null || s.text !== s.h || !forms.some((f) => P.eqi(f, s.h))) return;
    put(s.h, P.eqi(s.m, W), { mem: m, i });
  }));
  return [...groups.values()].sort((a, b) => (b.real - a.real) || (b.takes.length + b.mems.length) - (a.takes.length + a.mems.length));
};
const snip = (segs, i, W, fixed) => segs.map((s, k) => k === i ? `<mark${fixed ? ' class="fixed"' : ''}>${P.esc(fixed ? P.fitCase(W, k === 0) : s.h)}</mark>` : P.esc(s.text)).join('');
const ARC = `<svg class="jt-link" viewBox="0 0 64 22" aria-hidden="true"><path d="M3 15 C 20 1, 44 1, 61 15"/><path d="M56 10.5l5 4.5-6.2 1.6" style="stroke-dasharray:none"/></svg>`;

/* ---------- page ---------- */
const render = () => {
  const page = S.ctx.page;
  page.innerHTML = `<div class="jt">
    <div class="jt-field">${P.icon('search', 20, 2)}<span class="gh" id="jt-gh"></span><input id="jt-q" placeholder="Type a word you meant" value="${P.esc(S.q)}" autocomplete="off" spellcheck="false"><span class="kbd">${S.q ? '<span class="kc">↩</span> fix' : ''}</span></div>
    <div class="jt-under">
      <div class="jt-try">Try<button data-try="Tesseract">Tesseract</button><button data-try="Claude">Claude</button><button data-try="DFlash2">DFlash2</button></div>
      <div class="jt-tip">Anywhere you type <code>;;claude</code> then space fixes your last take.</div>
    </div>
    <div id="jt-body" class="jt-body"></div>
  </div>`;
  const inp = page.querySelector('#jt-q');
  inp.oninput = () => { S.q = inp.value; S.result = null; S.checked = {}; ghost(); body(); };
  inp.onkeydown = (e) => {
    if (e.key === 'Tab') { e.preventDefault(); e.stopPropagation(); const c = canon(inp.value); if (c) { inp.value = S.q = c; ghost(); body(); } }
    if (e.key === 'Enter') { e.stopPropagation(); const c = canon(inp.value); if (c && c !== inp.value) { inp.value = S.q = c; ghost(); body(); } else applyAll(); }
    if (e.key === 'Escape') { e.stopPropagation(); inp.value = S.q = ''; S.result = null; ghost(); body(); inp.blur(); }
  };
  page.querySelector('.jt-try').onclick = (e) => { const b = e.target.closest('[data-try]'); if (!b) return; S.q = b.dataset.try; S.result = null; S.checked = {}; render(); };
  ghost(); body();
};
const ghost = () => {
  const inp = S.ctx.page.querySelector('#jt-q'); const g = S.ctx.page.querySelector('#jt-gh');
  const c = canon(inp.value);
  g.textContent = c && c.toLowerCase().startsWith(inp.value.toLowerCase()) && c.length > inp.value.length ? inp.value + c.slice(inp.value.length) : '';
  S.ctx.page.querySelector('.jt-field .kbd').innerHTML = inp.value ? (g.textContent ? '<span class="kc">tab</span>' : '<span class="kc">↩</span> fix') : '';
};
const body = () => {
  const el = S.ctx.page.querySelector('#jt-body');
  const W = S.q && (P.WORDS.find((w) => P.eqi(w.w, S.q))?.w || canon(S.q) === S.q && S.q);
  if (!S.q) { el.innerHTML = idleHTML(); wireIdle(el); return; }
  if (!W) { el.innerHTML = `<p class="jt-sum" style="color:var(--ink3)">Keep typing, or press <span class="kc">tab</span> to finish the word.</p>`; return; }
  const groups = S.result?.W === W ? S.result.groups : gather(W);
  if (!S.result) groups.forEach((g) => { if (!(g.key in S.checked)) S.checked[g.key] = g.real; });
  const real = groups.filter((g) => g.real);
  const nT = groups.filter((g) => S.checked[g.key]).reduce((a, g) => a + g.takes.length, 0);
  const nM = groups.filter((g) => S.checked[g.key]).reduce((a, g) => a + g.mems.length, 0);
  const done = S.result?.W === W;
  const forms = real.map((g) => g.heard);
  const sum = done
    ? `Fixed. <em>${P.esc(W)}</em> everywhere it was misheard.<small>${S.result.nT} takes and ${S.result.nM} ${S.result.nM === 1 ? 'memory' : 'memories'} now say ${P.esc(W)}, and from now on I’ll write it when I hear ${S.result.forms.map((f) => `“${P.esc(f)}”`).join(' or ')}.</small>`
    : groups.length ? `I heard <em>${P.esc(W)}</em> ${real.length} ${real.length === 1 ? 'way' : 'different ways'}.<small>${real.reduce((a, g) => a + g.takes.length, 0)} takes and ${real.reduce((a, g) => a + g.mems.length, 0)} of the agent’s memories still have the wrong word. Uncheck anything that was right.</small>`
      : `Nothing to fix for <em>${P.esc(W)}</em>.<small>Every take and memory already spells it right${P.st.rules.some((r) => r.on && r.meant === W) ? ', and new takes will too' : ''}.</small>`;
  el.innerHTML = `<p class="jt-sum">${sum}</p>
    <div class="jt-groups">${groups.map((g) => groupHTML(g, W, done)).join('')}</div>
    ${groups.length ? `<div class="jt-apply">${done ? `<span>${P.icon('check', 14, 2.4)}</span><span>Done. It’s in your words below.</span><button class="undo" id="jt-undo">Undo</button>` : `<button id="jt-go"${nT + nM ? '' : ' disabled'}>Fix ${nT} ${nT === 1 ? 'take' : 'takes'}${nM ? ` and ${nM} ${nM === 1 ? 'memory' : 'memories'}` : ''}</button><span>and write ${P.esc(W)} from now on when I hear ${forms.slice(0, 3).map((f) => `“${P.esc(f)}”`).join(', ')}</span>`}</div>` : ''}`;
  el.onclick = (e) => {
    const gh = e.target.closest('.jt-gh'); if (gh && !done) { const k = gh.dataset.k; S.checked[k] = !S.checked[k]; body(); return; }
    if (e.target.closest('#jt-go')) applyAll();
    if (e.target.closest('#jt-undo')) undoAll();
  };
};
const groupHTML = (g, W, done) => {
  const on = !!S.checked[g.key]; const isDone = done && on;
  const n = [g.takes.length && `${g.takes.length} ${g.takes.length === 1 ? 'take' : 'takes'}`, g.mems.length && `${g.mems.length} ${g.mems.length === 1 ? 'memory' : 'memories'}`].filter(Boolean).join(' · ');
  const items = [...g.takes.map((x) => `<li>${P.icon(P.APPS[x.take.app].icon, 14)}<span>${snip(x.take.segs, x.i, W, isDone || x.take.segs[x.i].text !== x.take.segs[x.i].h)}</span><span class="t">${P.ago(x.take.at)}</span></li>`),
    ...g.mems.map((x) => `<li>${P.icon('brain', 14)}<span>${snip(x.mem.segs, x.i, W, isDone || x.mem.segs[x.i].text !== x.mem.segs[x.i].h)}</span><span class="t">memory</span></li>`)].slice(0, 4);
  return `<div class="jt-g${on ? ' on' : ''}${g.real ? '' : ' maybe'}${isDone ? ' done' : ''}">
    <div class="jt-gh" data-k="${g.key}"><span class="jt-box">${on ? P.icon('check', 13, 3) : ''}</span><span class="jt-h">${P.esc(g.heard)}</span>${ARC}<span class="jt-m">${P.esc(W)}</span><span class="jt-n">${n}</span></div>
    <ul class="jt-snips">${items.join('')}</ul>
    ${g.real ? '' : `<div class="jt-why">Probably right as written here, so it’s left unchecked.</div>`}
  </div>`;
};
const idleHTML = () => {
  const rules = P.st.rules.filter((r) => r.on);
  const byWord = new Map();
  rules.forEach((r) => { if (!byWord.has(r.meant)) byWord.set(r.meant, { w: r.meant, forms: [], n: 0 }); const x = byWord.get(r.meant); x.forms.push(r.heard); x.n += r.applied; });
  const words = [...byWord.values()];
  const recent = P.st.takes.slice(0, 7);
  return `<section><h3>Your words <small>${words.length ? 'click one to see where it was heard' : ''}</small></h3>
      ${words.length ? `<div class="jt-index">${words.map((x) => `<div data-w="${P.esc(x.w)}"><b>${P.esc(x.w)}</b><span>${x.forms.map(P.esc).join(', ')}</span><small>${x.n ? `fixed ${x.n}×` : 'new'}</small></div>`).join('')}</div>` : '<p style="color:var(--ink3);margin:0">Nothing yet.</p>'}</section>
    <section><h3>Recent takes</h3><ol class="jt-recent">${recent.map((t) => `<li><span class="when">${P.ago(t.at)}</span><span>${t.segs.map((s, i) => s.h == null ? P.esc(s.text) : P.segHTML(t, s, i).replace('class="seg', `class="seg${s.status === 'fixed' || s.status === 'auto' ? ' fixed' : ''}`)).join('')}</span></li>`).join('')}</ol></section>`;
};
const wireIdle = (el) => { el.onclick = (e) => { const d = e.target.closest('[data-w]'); if (d) { S.q = d.dataset.w; S.result = null; S.checked = {}; render(); } }; };

const applyAll = (W0) => {
  const W = W0 || (S.q && (P.WORDS.find((w) => P.eqi(w.w, S.q))?.w || canon(S.q)));
  if (!W) return 0;
  const groups = gather(W);
  groups.forEach((g) => { if (!(g.key in S.checked)) S.checked[g.key] = g.real; });
  const chosen = groups.filter((g) => (W0 ? g.real : S.checked[g.key]));
  let nT = 0, nM = 0; const rules = [], changed = [];
  chosen.forEach((g) => {
    rules.push(P.learn({ heard: g.heard, meant: W, source: 'everywhere' }));
    g.takes.forEach(({ take, i }) => {
      const s = take.segs[i]; changed.push({ s, text: s.text, status: s.status, take, i });
      if (take.live) P.fixSeg(take, i, W); else { s.text = P.fitCase(W, i === 0); s.status = P.eqi(s.text, s.m) ? 'fixed' : 'wrong'; }
      nT++;
    });
    g.mems.forEach(({ mem, i }) => { const s = mem.segs[i]; changed.push({ s, text: s.text, status: s.status }); s.text = W; nM++; });
  });
  P.renderTargets();
  if (!W0) { S.result = { W, groups, nT, nM, rules, changed, forms: chosen.map((g) => g.heard) }; P.step('everywhere'); body(); }
  return { nT, nM, rules, changed };
};
const undoAll = () => {
  const R = S.result; if (!R) return;
  R.rules.forEach(P.unlearn);
  R.changed.forEach((c) => { c.s.text = c.text; c.s.status = c.status; });
  P.renderTargets();
  S.result = null; body();
};

/* ---------- ;;word, anywhere ---------- */
const TRIG = /(^|\s);;([A-Za-z0-9.\-]*)$/;
const lastTakeIn = (app) => P.st.takes.find((t) => t.app === app && t.inserted && !t.seed);
const card = (kind, html) => {
  let c = S.ctx.overlay.querySelector('.jt-card');
  if (!c) { c = P.el(`<div class="jt-card glass"></div>`); S.ctx.overlay.appendChild(c); }
  c.dataset.kind = kind; c.innerHTML = html; S.card = kind;
  S.ctx.overlay.querySelector('#jt-pill').classList.remove('show');
  return c;
};
const closeCard = (ms = 0) => { clearTimeout(S.cardTimer); S.cardTimer = setTimeout(() => { S.ctx.overlay.querySelector('.jt-card')?.remove(); S.card = null; }, ms); };
const lineWithHit = (take, i, W) => take.segs.map((s, k) => k === i ? `<span class="hit"><span class="gh">${P.esc(P.fitCase(W, k === 0))}</span>${P.esc(s.text)}</span>` : P.esc(s.text)).join('');
const preview = (typed, app) => {
  clearTimeout(S.cardTimer);
  const take = lastTakeIn(app);
  const W = canon(typed);
  const i = findIn(take, W);
  if (!take) { card('preview', `<div class="miss">Nothing dictated in ${P.APPS[app].name} yet.</div>`); return; }
  card('preview', `<div class="line">${i >= 0 ? lineWithHit(take, i, W) : P.esc(P.takeText(take))}</div>
    <div class="foot"><code>;;${P.esc(typed)}</code>${W && W !== typed ? `<span><span class="kc">tab</span> ${P.esc(W)}</span>` : ''}<span><span class="kc">space</span> fix</span><span><span class="kc">esc</span> cancel</span>${!typed ? '<span>type the word you meant</span>' : i < 0 && W ? '<span>no sound-alike yet</span>' : ''}</div>`);
};
const fixInline = (typed, app) => {
  const take = lastTakeIn(app);
  const W = canon(typed) || typed;
  const i = findIn(take, W);
  if (i < 0) {
    card('miss', `<div class="miss">Nothing in your last take sounds like <b>${P.esc(W)}</b>. Your text is unchanged.</div>`);
    closeCard(2600); return;
  }
  const heard = take.segs[i].h;
  const rule = P.learn({ heard, meant: W, source: 'inline' });
  const inPlace = P.fixSeg(take, i, W);
  P.step('inline');
  const more = gather(W).filter((g) => g.real);
  const nT = more.reduce((a, g) => a + g.takes.length, 0), nM = more.reduce((a, g) => a + g.mems.length, 0);
  S.pending = { W, rule, take, i, heard };
  card('result', `<div class="jt-res"><span class="h">${P.esc(heard)}</span><svg viewBox="0 0 70 26" aria-hidden="true"><path d="M3 18 C 22 2, 48 2, 67 18"/></svg><b>${P.esc(W)}</b></div>
    <div class="sub">${inPlace ? `Fixed in ${P.APPS[app].name}` : 'Learned'}, and I’ll write it this way from now on.${nT + nM ? ` It was also misheard in ${nT} older ${nT === 1 ? 'take' : 'takes'}${nM ? ` and ${nM} ${nM === 1 ? 'memory' : 'memories'}` : ''}.` : ''}</div>
    <div class="acts">${nT + nM ? `<button class="pri" data-a="all">Fix those too <span class="kc">⌃⌥Space</span></button>` : ''}<button data-a="undo">Undo</button></div>`);
  S.ctx.overlay.querySelector('.jt-card').onclick = (e) => { const b = e.target.closest('button'); if (b) b.dataset.a === 'all' ? fixThoseToo() : undoInline(); };
  closeCard(7000);
  if (!S.q) render();
};
const fixThoseToo = () => {
  const p = S.pending; if (!p) return;
  const r = applyAll(p.W);
  S.pending = null;
  card('result', `<div class="jt-res">${P.icon('check', 20, 2.4)}<b>${P.esc(p.W)}</b></div><div class="sub">Also fixed ${r.nT} older ${r.nT === 1 ? 'take' : 'takes'}${r.nM ? ` and ${r.nM} ${r.nM === 1 ? 'memory' : 'memories'}` : ''}.</div>`);
  closeCard(2400);
  if (!S.q) render();
};
const undoInline = () => {
  const p = S.pending; if (!p) return;
  P.unlearn(p.rule); P.fixSeg(p.take, p.i, p.heard, 'wrong');
  S.pending = null; closeCard(0); if (!S.q) render();
};

V.onEvent = (type, d) => {
  if (!S) return;
  const pill = S.ctx.overlay.querySelector('#jt-pill');
  if (type === 'rec-start') { closeCard(0); clearTimeout(S.pillTimer); pill.innerHTML = `<span class="bars">${'<span></span>'.repeat(6)}</span>`; pill.classList.add('show'); }
  if (type === 'processing') pill.innerHTML = `<span style="color:var(--glass-ink2)">…</span>`;
  if (type === 'inserted') {
    P.step('dictate');
    const auto = d.take.segs.filter((s) => s.status === 'auto');
    if (auto.length) { P.step('again'); pill.innerHTML = `<span style="color:var(--ok)">${P.icon('check', 14, 2.4)}</span>Wrote ${auto.map((s) => `<b>${P.esc(s.text)}</b>`).join(', ')}`; }
    else pill.innerHTML = `<span style="color:var(--glass-ink2)">Wrong word? Type <code style="font-family:var(--mono);color:var(--accent-text)">;;</code> and the right one</span>`;
    S.pillTimer = setTimeout(() => pill.classList.remove('show'), 2600);
    if (!S.q) render();
  }
  if (type === 'typed') {
    const m = TRIG.exec(d.tail);
    if (m) preview(m[2], d.app);
    else if (S.card === 'preview') closeCard(0);
  }
};

V.onKey = (e, k) => {
  if (!S) return false;
  if (S.card === 'field') return false;
  if (k.inField || e.metaKey || e.ctrlKey) return false;
  const app = P.st.front;
  if (app === 'terminal' || app === 'notes') {
    const m = TRIG.exec(P.typedTail(app));
    if (m) {
      if ((e.key === ' ' || e.key === 'Enter') && m[2]) {
        P.trimTyped(app, m[0].length - m[1].length);
        if (P.typedTail(app).endsWith(' ')) P.trimTyped(app, 1);
        fixInline(m[2], app); return true;
      }
      if (e.key === 'Tab') { const c = canon(m[2]); if (c) { P.trimTyped(app, m[2].length); [...c].forEach((ch) => P.typeInto(app, ch)); } return true; }
      if (e.key === 'Escape') { P.trimTyped(app, m[0].length - m[1].length); closeCard(0); return true; }
    }
  }
  if (e.key === 'Escape' && S.card) { closeCard(0); return true; }
  return false;
};

V.onFixKey = () => {
  if (S.card === 'result' && S.pending) { fixThoseToo(); return; }
  const app = P.st.target;
  const take = lastTakeIn(app);
  const c = card('field', `<div class="line" id="jt-fl">${take ? P.esc(P.takeText(take)) : 'Nothing dictated yet.'}</div>
    <div class="fld"><span>Meant</span><input id="jt-fi" autocomplete="off" spellcheck="false" placeholder="the word you meant"></div>
    <div class="foot"><span><span class="kc">↩</span> fix</span><span><span class="kc">tab</span> finish the word</span><span><span class="kc">esc</span> close</span></div>`);
  const inp = c.querySelector('#jt-fi');
  inp.oninput = () => { const W = canon(inp.value); const i = findIn(take, W); c.querySelector('#jt-fl').innerHTML = take ? (i >= 0 ? lineWithHit(take, i, W) : P.esc(P.takeText(take))) : ''; };
  inp.onkeydown = (e) => {
    if (['Enter', 'Escape', 'Tab'].includes(e.key)) { e.preventDefault(); e.stopPropagation(); }
    if (e.key === 'Tab') { const W = canon(inp.value); if (W) { inp.value = W; inp.oninput(); } }
    if (e.key === 'Escape') closeCard(0);
    if (e.key === 'Enter' && inp.value.trim()) { S.card = null; fixInline(inp.value.trim(), app); }
  };
  setTimeout(() => inp.focus(), 20);
};

P.register(V);
})();
