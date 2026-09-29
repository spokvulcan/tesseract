/* PROTOTYPE, THROWAWAY. Variant 1, Roll Call.
   Teach it your words before it gets them wrong. The page is a prompter: the app lists the names and
   terms it found in your projects and memory, you say each one once, and it learns what Whisper hears
   ("SRACT") as that word. A word that slipped through: ⌃⌥Space, type it, say it. */
(() => {
const P = Proto;
let S = null;

const V = {
  id: 'roll-call', num: 1, name: 'Roll Call',
  thesis: 'Teach it your words before it gets them wrong. Say each one once; it learns what it hears.',
  steps: [
    { id: 'dictate', text: 'Hold <b>`</b> and dictate the first line into Terminal' },
    { id: 'rollcall', text: 'Click the Tesseract window and say three words on the card' },
    { id: 'again', text: 'Dictate again. The words you taught come out right' },
    { id: 'teach', text: 'Missed one that isn’t on the list? Press <b>Tab</b>, type it, say it' },
  ],
};

V.css = `
.rc{max-width:700px;margin:0 auto;padding:6px 32px 120px;display:flex;flex-direction:column;gap:22px}
.rc-count{font:500 13px/1.45 var(--sans);color:var(--ink2)}
.rc-count b{color:var(--ink);font-weight:600}
.rc-rail{display:flex;flex-wrap:wrap;gap:6px}
.rc-chip{height:28px;padding:0 11px;border-radius:14px;border:1px solid var(--line);background:transparent;display:inline-flex;align-items:center;gap:6px;font:500 12.5px/1 var(--sans);color:var(--ink2);cursor:pointer;transition:background .2s,color .2s,border-color .2s}
.rc-chip:hover{border-color:var(--ink3)}
.rc-chip.cur{background:var(--ink);color:var(--win-bg);border-color:var(--ink)}
.rc-chip.learned{background:color-mix(in srgb,var(--accent) 16%,transparent);border-color:transparent;color:var(--accent-text)}
.rc-chip.fine{color:var(--ink3);border-style:dashed}
.rc-chip .ic{opacity:.85}

.rc-stage{position:relative;border-radius:22px;padding:34px 36px 26px;min-height:318px;display:flex;flex-direction:column;align-items:center;text-align:center;
  background:radial-gradient(120% 90% at 50% 0%,color-mix(in srgb,var(--accent) 9%,transparent),transparent 62%),var(--fill);
  box-shadow:inset 0 0 0 1px var(--line)}
.rc-src{font:600 11px/1 var(--sans);letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);display:flex;gap:8px;align-items:center}
.rc-src i{font-style:normal;color:var(--ink2);letter-spacing:.02em;text-transform:none;font-weight:500}
.rc-word{font:700 76px/1.02 var(--display);letter-spacing:-.035em;margin:18px 0 6px;color:var(--ink);transition:transform .35s cubic-bezier(.2,.9,.3,1.2),color .3s}
.rc-stage[data-card="listening"] .rc-word{transform:scale(1.03)}
.rc-stage[data-card="learned"] .rc-word,.rc-stage[data-card="fine"] .rc-word{color:var(--ink)}
.rc-meter{display:flex;gap:4px;align-items:center;height:30px;margin:6px 0 10px;opacity:.35;transition:opacity .2s}
.rc-stage[data-card="listening"] .rc-meter{opacity:1}
.rc-meter span{width:4px;height:4px;border-radius:2px;background:var(--ink2)}
.rc-stage[data-card="listening"] .rc-meter span{background:var(--accent)}
.rc-cue{font:15px/1.45 var(--sans);color:var(--ink2);min-height:44px;max-width:44ch}
.rc-cue .kc{font-size:12px}
.rc-heard{display:flex;align-items:center;justify-content:center;gap:14px;min-height:44px;font:15px/1.4 var(--sans);color:var(--ink2)}
.rc-heard .h{font:500 25px/1 var(--mono);color:var(--ink2);position:relative;padding:2px 2px}
.rc-heard .h::after{content:"";position:absolute;left:0;right:0;top:52%;height:2px;background:var(--danger);transform:scaleX(0);transform-origin:left;transition:transform .45s ease .15s}
.rc-stage[data-card="learned"] .rc-heard .h::after{transform:scaleX(1)}
.rc-heard .arrow{color:var(--accent);opacity:0;transform:translateX(-8px);transition:all .4s ease .35s}
.rc-heard .m{font:600 25px/1 var(--display);letter-spacing:-.02em;color:var(--accent-text);opacity:0;transform:translateY(6px);transition:all .45s ease .5s}
.rc-stage[data-card="learned"] .rc-heard .arrow{opacity:1;transform:none}
.rc-stage[data-card="learned"] .rc-heard .m{opacity:1;transform:none}
.rc-stamp{position:absolute;top:22px;right:24px;display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 10px;border-radius:13px;font:700 11px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;opacity:0;transform:scale(.6) rotate(-6deg);transition:all .35s cubic-bezier(.2,1.4,.4,1) .7s}
.rc-stage[data-card="learned"] .rc-stamp{opacity:1;transform:scale(1) rotate(-3deg);background:var(--accent);color:var(--accent-ink)}
.rc-stage[data-card="fine"] .rc-stamp{opacity:1;transform:scale(1) rotate(-3deg);background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok);transition-delay:0s}
.rc-actions{display:flex;gap:18px;justify-content:center;margin-top:14px;font:500 13px/1 var(--sans)}
.rc-actions button{background:none;border:0;padding:6px 2px;color:var(--ink2);cursor:pointer;display:inline-flex;gap:7px;align-items:center}
.rc-actions button:hover{color:var(--ink)}
.rc-actions button.primary{color:var(--accent-text)}
.rc-actions .kc{font-size:11px}

.rc-shelf h3,.rc-takes h3{margin:0 0 10px;font:600 13px/1 var(--sans);color:var(--ink2)}
.rc-list{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:2px 28px}
.rc-list li{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--line);font:15px/1.3 var(--sans)}
.rc-list li b{font-weight:600}
.rc-list .forms{font:12.5px/1.3 var(--mono);color:var(--ink3);text-decoration:line-through;text-decoration-color:color-mix(in srgb,var(--danger) 60%,transparent);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rc-list .src{font:12px/1 var(--sans);color:var(--ink3)}
.rc-list li button{grid-column:3;background:none;border:0;color:var(--ink3);font:12px var(--sans);cursor:pointer;display:none}
.rc-list li:hover .src{display:none}
.rc-list li:hover button{display:block}
.rc-list li.new{animation:rcin .6s ease}
@keyframes rcin{from{background:color-mix(in srgb,var(--accent) 18%,transparent)}}
.rc-fine{margin-top:12px;font:13px/1.5 var(--sans);color:var(--ink3)}
.rc-fine b{color:var(--ink2);font-weight:500}
.rc-add{display:flex;gap:8px;align-items:center;margin-top:14px}
.rc-add input{flex:1;height:32px;border-radius:8px;border:1px solid var(--line);background:var(--win-bg);color:var(--ink);padding:0 11px;font:14px var(--sans)}
.rc-add input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent)}
.rc-empty{font:14px/1.5 var(--sans);color:var(--ink3);padding:6px 0}

.rc-takes ol{list-style:none;margin:0;padding:0}
.rc-takes li{display:grid;grid-template-columns:92px 1fr;gap:14px;padding:10px 0;border-bottom:1px solid var(--line);font:15px/1.45 var(--sans)}
.rc-takes .when{color:var(--ink3);font-variant-numeric:tabular-nums;font-size:13px;padding-top:2px}
.rc-takes .when small{display:block;font-size:12px}
.rc-takes .seg.auto{box-shadow:inset 0 -2px 0 color-mix(in srgb,var(--accent) 70%,transparent)}

/* overlay */
.rc-pill{position:absolute;left:50%;bottom:40px;transform:translateX(-50%) scale(.85);opacity:0;height:38px;padding:0 16px;border-radius:19px;display:flex;align-items:center;gap:10px;font:500 13px/1 var(--sans);transition:opacity .18s,transform .22s cubic-bezier(.2,1.2,.4,1);pointer-events:none;white-space:nowrap}
.rc-pill.show{opacity:1;transform:translateX(-50%) scale(1)}
.rc-bars{display:flex;gap:3px;align-items:center;height:18px}
.rc-bars span{width:3px;height:4px;border-radius:2px;background:var(--danger)}
.rc-dots{display:flex;gap:4px}.rc-dots span{width:5px;height:5px;border-radius:50%;background:var(--glass-ink2);animation:rcdot 1s infinite}
.rc-dots span:nth-child(2){animation-delay:.15s}.rc-dots span:nth-child(3){animation-delay:.3s}
@keyframes rcdot{50%{opacity:.25}}
.rc-pill .ok{color:var(--ok)}
.rc-pill .hint{color:var(--glass-ink2)}
.rc-pill b{font-weight:600}

.rc-teach{position:absolute;left:50%;bottom:36px;width:460px;transform:translateX(-50%);border-radius:22px;padding:18px 20px 16px;display:flex;flex-direction:column;gap:10px;animation:rcpop .28s cubic-bezier(.2,1.2,.4,1)}
@keyframes rcpop{from{opacity:0;transform:translateX(-50%) translateY(10px) scale(.96)}}
.rc-teach .tt{font:600 13px/1 var(--sans);color:var(--glass-ink2);display:flex;justify-content:space-between;align-items:center}
.rc-teach .last{font:13px/1.45 var(--sans);color:var(--glass-ink2);max-height:40px;overflow:hidden}
.rc-teach .field{position:relative;height:44px;border-radius:12px;background:color-mix(in srgb,var(--glass-ink) 7%,transparent);display:flex;align-items:center;padding:0 14px}
.rc-teach input{position:relative;z-index:1;flex:1;border:0;background:transparent;font:600 20px/1 var(--display);color:var(--glass-ink);outline:none;padding:0}
.rc-teach .ghost{position:absolute;left:14px;font:600 20px/1 var(--display);color:var(--glass-ink2);opacity:.45;white-space:pre;pointer-events:none}
.rc-teach .big{font:700 40px/1.05 var(--display);letter-spacing:-.03em;text-align:center;margin:4px 0}
.rc-teach .cue{text-align:center;font:14px/1.4 var(--sans);color:var(--glass-ink2)}
.rc-teach .rule{display:flex;align-items:center;justify-content:center;gap:10px;font:15px/1 var(--sans)}
.rc-teach .rule .h{font:500 17px/1 var(--mono);text-decoration:line-through;text-decoration-color:var(--danger);color:var(--glass-ink2)}
.rc-teach .rule .m{font:600 18px/1 var(--display);color:var(--accent-text)}
.rc-teach .foot{display:flex;justify-content:space-between;align-items:center;font:12px/1 var(--sans);color:var(--glass-ink2)}
.rc-teach .foot button{background:none;border:0;color:var(--accent-text);font:600 12.5px var(--sans);cursor:pointer;padding:4px 0}
.rc-teach .meter{display:flex;gap:4px;align-items:center;justify-content:center;height:26px}
.rc-teach .meter span{width:4px;height:4px;border-radius:2px;background:var(--accent)}
`;

/* ---------- state ---------- */
const listWords = () => P.WORDS.map((info) => ({ w: info.w, info, state: 'pending', said: 0, rules: [] }));

V.mount = (ctx) => {
  S = { ctx, view: 'call', list: listWords(), cur: 0, card: 'ready', beatTimer: 0, teach: null, said: 0, takes: 0 };
  // One word taught yesterday, so the shelf isn't empty on first look.
  const el = S.list.find((x) => x.w === 'ElevenLabs');
  const r = P.learn({ heard: 'Eleven Labs', meant: 'ElevenLabs', source: 'roll call' });
  r.at = new Date(Date.now() - 26 * 3600e3); el.state = 'learned'; el.rules.push(r); el.said = 1;
  S.cur = S.list.findIndex((x) => x.state === 'pending');

  ctx.tools.innerHTML = `<span class="status-chip"><i></i>Ready · <span class="kc">⌥Space</span></span>
    <div class="seg-ctl" id="rc-view"><button data-view="call" class="on">Roll call</button><button data-view="takes">Takes</button></div>`;
  ctx.tools.querySelector('#rc-view').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    S.view = b.dataset.view;
    ctx.tools.querySelectorAll('#rc-view button').forEach((x) => x.classList.toggle('on', x === b));
    render(); arm();
  });
  ctx.overlay.innerHTML = `<div class="rc-pill glass" id="rc-pill"></div>`;
  S.unFrame = P.onFrame(meters);
  render(); arm();
};
V.unmount = () => { S?.unFrame?.(); clearTimeout(S?.beatTimer); S = null; };

/* ---------- page ---------- */
const counts = () => {
  const c = { learned: 0, fine: 0, pending: 0 };
  S.list.forEach((x) => { if (x.state !== 'gone') c[x.state]++; });
  return c;
};
const render = () => {
  const page = S.ctx.page;
  if (S.view === 'takes') { page.innerHTML = `<div class="rc">${takesHTML()}</div>`; return; }
  const c = counts(); const total = S.list.filter((x) => x.state !== 'gone').length;
  page.innerHTML = `<div class="rc">
    <div class="rc-count"><b>${total} words</b> from your projects and memory that dictation hasn’t heard you say yet.
      <span>${c.learned} learned, ${c.fine} already right, ${c.pending} to go.</span></div>
    <div class="rc-rail" id="rc-rail"></div>
    <section class="rc-stage" id="rc-stage"></section>
    <section class="rc-shelf" id="rc-shelf"></section>
  </div>`;
  renderRail(); renderCard(); renderShelf();
};
const renderRail = () => {
  const rail = S.ctx.page.querySelector('#rc-rail'); if (!rail) return;
  rail.innerHTML = S.list.map((x, i) => x.state === 'gone' ? '' :
    `<button class="rc-chip ${x.state}${i === S.cur ? ' cur' : ''}" data-i="${i}">${x.state === 'learned' ? P.icon('check', 12, 2.2) : ''}${P.esc(x.w)}</button>`).join('');
  rail.onclick = (e) => { const b = e.target.closest('.rc-chip'); if (!b) return; S.cur = +b.dataset.i; S.card = S.list[S.cur].state === 'pending' ? 'ready' : S.list[S.cur].state; renderRail(); renderCard(); arm(); };
};
const fromLabel = (info) => info.from === 'project' ? 'From your projects' : info.from === 'you' ? 'You added it' : 'From your memory';
const renderCard = () => {
  const stage = S.ctx.page.querySelector('#rc-stage'); if (!stage) return;
  const x = S.list[S.cur];
  if (!x || S.card === 'done') {
    stage.dataset.card = 'done';
    stage.innerHTML = `<div class="rc-src">Roll call</div><div class="rc-word" style="font-size:44px">That’s everyone.</div>
      <div class="rc-cue">Dictate as usual. I’ll write these the way you spell them, and I’ll add new names here as they show up in your projects.</div>`;
    return;
  }
  const front = P.st.front === 'tess';
  const key = `<span class="kc">⌥Space</span>`;
  let cue = '';
  if (S.card === 'ready') cue = front ? `Hold ${key} and say it the way you usually do.` : `Click this window, then hold ${key} and say it.`;
  else if (S.card === 'short') cue = `Hold ${key} for the whole word.`;
  else if (S.card === 'listening') cue = 'Listening…';
  const lastRule = x.rules[x.rules.length - 1];
  const heard = S.card === 'heard' || S.card === 'learned' || S.card === 'same';
  stage.dataset.card = S.card === 'same' ? 'learned' : S.card;
  stage.innerHTML = `
    <span class="rc-stamp">${S.card === 'fine' ? P.icon('check', 13, 2.4) + 'Already right' : P.icon('check', 13, 2.4) + 'Learned'}</span>
    <div class="rc-src">${fromLabel(x.info)}<i>${P.esc(x.info.why || '')}</i></div>
    <div class="rc-word">${P.esc(x.w)}</div>
    <div class="rc-meter" id="rc-meter">${'<span></span>'.repeat(28)}</div>
    ${heard ? `<div class="rc-heard"><span>I heard</span><span class="h">${P.esc(S.lastHeard)}</span><span class="arrow">${P.icon('arrow', 18, 2)}</span><span class="m">${P.esc(x.w)}</span></div>
      <div class="rc-cue">${S.card === 'same' ? 'Same as last time. Already learned.' : `From now on, when I hear “${P.esc(S.lastHeard)}” I’ll write ${P.esc(x.w)}.`}</div>`
      : S.card === 'fine' ? `<div class="rc-heard"><span>I heard it right. Nothing to learn.</span></div><div class="rc-cue"></div>`
      : `<div class="rc-cue">${cue}</div>`}
    <div class="rc-actions">
      ${S.card === 'learned' || S.card === 'same' ? `<button class="primary" data-a="next">Next word <span class="kc">↩</span></button>
        <button data-a="again">${P.icon('mic', 14)}Say it again</button>
        <button data-a="undo">${P.icon('undo', 14)}Undo</button>`
      : `<button data-a="next">Skip <span class="kc">→</span></button><button data-a="gone">Not my word</button>`}
    </div>`;
  stage.onclick = (e) => {
    const b = e.target.closest('button'); if (!b) return;
    e.stopPropagation();
    if (b.dataset.a === 'next') next();
    if (b.dataset.a === 'gone') { x.state = 'gone'; next(); }
    if (b.dataset.a === 'again') { S.card = 'ready'; renderCard(); arm(); }
    if (b.dataset.a === 'undo') { x.rules.forEach(P.unlearn); x.rules = []; x.state = 'pending'; S.card = 'ready'; render(); arm(); }
  };
};
const renderShelf = () => {
  const shelf = S.ctx.page.querySelector('#rc-shelf'); if (!shelf) return;
  const learned = S.list.filter((x) => x.state === 'learned');
  const taught = P.st.rules.filter((r) => r.on && r.source === 'typed and said' && !learned.some((x) => x.rules.includes(r)));
  const fine = S.list.filter((x) => x.state === 'fine');
  const item = (w, forms, src, newest, key) => `<li class="${newest ? 'new' : ''}"><b>${P.esc(w)}</b><span class="forms">${forms.map(P.esc).join(', ')}</span><span class="src">${src}</span><button data-forget="${key}">Forget</button></li>`;
  const rows = [
    ...taught.map((r) => item(r.meant, [r.heard], 'taught in Terminal', r === S.fresh, 'r' + r.id)),
    ...learned.map((x) => item(x.w, x.rules.map((r) => r.heard), P.ago(x.rules[0].at) === 'just now' ? 'just now' : x.rules[0].at < new Date(Date.now() - 20 * 3600e3) ? 'yesterday' : 'today', x.rules.includes(S.fresh), 'x' + S.list.indexOf(x))),
  ];
  shelf.innerHTML = `<h3>Learned</h3>
    ${rows.length ? `<ul class="rc-list">${rows.join('')}</ul>` : `<div class="rc-empty">Nothing yet. Say the first word on the card.</div>`}
    ${fine.length ? `<div class="rc-fine">Already right: <b>${fine.map((x) => P.esc(x.w)).join(', ')}</b></div>` : ''}
    <div class="rc-add"><input id="rc-add" placeholder="Add a word you say, like a name or a project" autocomplete="off"></div>`;
  S.fresh = null;
  shelf.onclick = (e) => {
    const b = e.target.closest('[data-forget]'); if (!b) return;
    const k = b.dataset.forget;
    if (k[0] === 'r') { const r = P.st.rules.find((x) => x.id === +k.slice(1)); r && P.unlearn(r); }
    else { const x = S.list[+k.slice(1)]; x.rules.forEach(P.unlearn); x.rules = []; x.state = 'pending'; }
    render(); arm();
  };
  shelf.querySelector('#rc-add').onkeydown = (e) => {
    if (e.key !== 'Enter' || !e.target.value.trim()) return;
    addWord(e.target.value.trim()); e.target.value = ''; e.target.blur();
  };
};
const takesHTML = () => `<section class="rc-takes"><h3>Recent takes</h3><ol>${P.st.takes.map((t) => `<li><div class="when">${P.ago(t.at)}<small>${P.APPS[t.app].name}</small></div>
  <div>${t.segs.map((s, i) => s.h != null ? P.segHTML(t, s, i).replace('class="seg', `class="seg${s.status === 'auto' ? ' auto' : ''}`).replace('<span ', `<span ${s.status === 'auto' ? `title="Heard ${P.esc(s.h)}" ` : ''}`) : P.esc(s.text)).join('')}</div></li>`).join('')}</ol></section>`;

/* ---------- saying a word on the card ---------- */
const arm = () => {
  const st = P.st;
  if (S.teach && S.teach.step === 'say') st.capture = teachCapture;
  else if (S.view === 'call' && st.front === 'tess' && S.list[S.cur] && S.card !== 'done' && S.card !== 'fine') st.capture = cardCapture;
  else st.capture = null;
  if (S.card === 'ready' || S.card === 'short') renderCard();
  P.renderHarness();
};
const cardCapture = {
  get hint() { return `<b>${P.esc(S.list[S.cur]?.w || '')}</b>, the word on the card`; },
  onStart() { S.card = 'listening'; renderCard(); },
  onEnd(ms) {
    if (ms < 260) { S.card = 'short'; renderCard(); return; }
    said();
  },
};
const said = () => {
  const x = S.list[S.cur]; const info = x.info;
  S.said++; if (S.said >= 3) P.step('rollcall');
  if (info.right) { x.state = 'fine'; S.card = 'fine'; renderCard(); renderRail(); arm(); setTimeout(() => S && S.list[S.cur] === x && next(), 1500); return; }
  const forms = [info.heard, ...(info.alts || [])];
  const heard = forms[Math.min(x.said, forms.length - 1)];
  x.said++;
  S.lastHeard = heard;
  if (x.rules.some((r) => P.eqi(r.heard, heard))) { S.card = 'same'; renderCard(); return; }
  S.card = 'heard'; renderCard();
  setTimeout(() => {
    if (!S || S.list[S.cur] !== x) return;
    x.rules.push(P.learn({ heard, meant: x.w, source: 'roll call' }));
    x.state = 'learned'; S.card = 'learned';
    renderCard(); renderRail(); renderShelf();
    S.ctx.page.querySelector('.rc-count span').textContent = (() => { const c = counts(); return `${c.learned} learned, ${c.fine} already right, ${c.pending} to go.`; })();
  }, 700);
};
const next = () => {
  const n = S.list.length;
  for (let k = 1; k <= n; k++) {
    const i = (S.cur + k) % n;
    if (S.list[i].state === 'pending') { S.cur = i; S.card = 'ready'; render(); arm(); return; }
  }
  S.card = 'done'; render(); arm();
};
const addWord = (w) => {
  const known = P.word(w);
  const info = known || { w, heard: w === 'a PR' ? 'APR' : w, right: w !== 'a PR', kind: 'Word', from: 'you', why: 'you added it' };
  let i = S.list.findIndex((x) => P.eqi(x.w, w));
  if (i < 0) { S.list.push({ w: info.w, info: { ...info, from: known ? info.from : 'you' }, state: 'pending', said: 0, rules: [] }); i = S.list.length - 1; }
  S.cur = i; S.card = 'ready'; S.view = 'call'; render(); arm();
  return S.list[i];
};

/* ---------- overlay: the pill, and the teach card (⌃⌥Space) ---------- */
const pill = () => S.ctx.overlay.querySelector('#rc-pill');
const showPill = (html, ms) => {
  const p = pill(); clearTimeout(S.beatTimer);
  p.innerHTML = html; p.classList.add('show');
  if (ms) S.beatTimer = setTimeout(() => p.classList.remove('show'), ms);
};
const meters = (lv) => {
  if (!S) return;
  const phase = P.st.phase;
  const bars = pill()?.querySelectorAll('.rc-bars span');
  if (bars) bars.forEach((b, i) => { b.style.height = `${4 + lv * 14 * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7 + performance.now() / 170)))}px`; });
  const m = S.ctx.page.querySelector('#rc-meter');
  if (m) m.querySelectorAll('span').forEach((b, i) => { b.style.height = phase === 'listening' && P.st.capture === cardCapture ? `${4 + lv * 24 * Math.abs(Math.sin(i * 0.9 + performance.now() / 140))}px` : '4px'; });
  const tm = S.ctx.overlay.querySelector('.rc-teach .meter');
  if (tm) tm.querySelectorAll('span').forEach((b, i) => { b.style.height = phase === 'listening' ? `${4 + lv * 20 * Math.abs(Math.sin(i * 1.1 + performance.now() / 150))}px` : '4px'; });
};

V.onEvent = (type, d) => {
  if (!S) return;
  if (type === 'learn') S.fresh = d.rule;
  if (type === 'focus') arm();
  if (type === 'rec-start') showPill(`<span class="rc-bars">${'<span></span>'.repeat(7)}</span>`);
  if (type === 'processing') showPill(`<span class="rc-dots"><span></span><span></span><span></span></span>`);
  if (type === 'inserted') {
    const t = d.take; S.takes++;
    P.step('dictate');
    const auto = t.segs.filter((s) => s.status === 'auto');
    if (auto.length) { P.step('again'); showPill(`<span class="ok">${P.icon('check', 15, 2.4)}</span><span>Wrote <b>${auto.map((s) => P.esc(s.text)).join('</b>, <b>')}</b> from your words</span>`, 2600); }
    else if (S.takes <= 2 && P.mistakes(t).length) showPill(`<span class="hint">A word wrong? <span class="kc">⌃⌥Space</span> to teach it</span>`, 3200);
    else pill().classList.remove('show');
    if (S.view === 'takes') render();
  }
};

V.onFixKey = () => {
  if (S.teach) return;
  S.teach = { step: 'type', take: P.lastLive() || P.st.last };
  pill().classList.remove('show');
  drawTeach();
};
const closeTeach = (ms = 0) => {
  const t = S.teach;
  setTimeout(() => { if (!S || S.teach !== t) return; S.teach = null; S.ctx.overlay.querySelector('.rc-teach')?.remove(); arm(); }, ms);
};
const drawTeach = () => {
  const t = S.teach; const ov = S.ctx.overlay;
  let card = ov.querySelector('.rc-teach');
  if (!card) { card = P.el(`<div class="rc-teach glass"></div>`); ov.appendChild(card); }
  const last = t.take ? P.takeHTML(t.take) : '<i>Nothing dictated yet.</i>';
  if (t.step === 'type') {
    card.innerHTML = `<div class="tt"><span>Teach a word</span><span><span class="kc">esc</span></span></div>
      <div class="last">${last}</div>
      <div class="field"><span class="ghost" id="rc-ghost"></span><input id="rc-teach-in" placeholder="Type it the way it’s spelled" autocomplete="off" spellcheck="false"></div>
      <div class="foot"><span>Then say it once, and I’ll find it in your last take.</span><span><span class="kc">↩</span></span></div>`;
    const inp = card.querySelector('#rc-teach-in'); const ghost = card.querySelector('#rc-ghost');
    const sugg = () => { const v = inp.value; const w = v && P.WORDS.find((x) => x.w.toLowerCase().startsWith(v.toLowerCase())); return w && w.w.length > v.length ? w.w : ''; };
    inp.oninput = () => { const s = sugg(); ghost.textContent = s ? inp.value + s.slice(inp.value.length) : ''; };
    inp.onkeydown = (e) => {
      if (['Enter', 'Escape', 'Tab'].includes(e.key)) e.stopPropagation();
      if ((e.key === 'Tab' || e.key === 'ArrowRight') && sugg()) { e.preventDefault(); inp.value = sugg(); ghost.textContent = ''; return; }
      if (e.key === 'Escape') { e.preventDefault(); closeTeach(); return; }
      if (e.key === 'Enter' && inp.value.trim()) { e.preventDefault(); t.word = inp.value.trim(); t.step = 'say'; inp.blur(); drawTeach(); arm(); }
    };
    setTimeout(() => inp.focus(), 30);
  } else if (t.step === 'say') {
    card.innerHTML = `<div class="tt"><span>Now say it</span><span><span class="kc">esc</span></span></div>
      <div class="big">${P.esc(t.word)}</div><div class="meter">${'<span></span>'.repeat(20)}</div>
      <div class="cue">Hold <span class="kc">⌥Space</span> and say it the way you usually do.</div>`;
  } else {
    card.innerHTML = t.html;
  }
};
const teachCapture = {
  get hint() { return `<b>${P.esc(S.teach?.word || '')}</b>, the word you’re teaching`; },
  onStart() {},
  onEnd(ms) {
    const t = S.teach; if (!t) return;
    if (ms < 260) return;
    const take = t.take;
    const i = take ? take.segs.findIndex((s) => s.h != null && P.eqi(s.m, t.word) && s.text === s.h) : -1;
    const known = P.word(t.word);
    if (i >= 0) {
      const heard = take.segs[i].h;
      const r = P.learn({ heard, meant: t.word, source: 'typed and said' });
      const inPlace = P.fixSeg(take, i, t.word);
      const x = S.list.find((y) => P.eqi(y.w, t.word)); if (x) { x.rules.push(r); x.state = 'learned'; }
      P.step('teach');
      t.step = 'done';
      t.html = `<div class="tt"><span>${inPlace ? 'Fixed in your last take' : 'Learned. That take was already sent, so only your history changed'}</span><span class="ok">${P.icon('check', 15, 2.4)}</span></div>
        <div class="rule"><span>I heard</span><span class="h">${P.esc(heard)}</span>${P.icon('arrow', 16, 2)}<span class="m">${P.esc(t.word)}</span></div>
        <div class="foot"><span>It’s on your Roll Call page now.</span><button id="rc-undo">Undo</button></div>`;
      drawTeach(); render();
      S.ctx.overlay.querySelector('#rc-undo').onclick = () => { P.unlearn(r); P.fixSeg(take, i, heard, 'wrong'); if (x) { x.rules = x.rules.filter((y) => y !== r); if (!x.rules.length) x.state = 'pending'; } closeTeach(); render(); };
      closeTeach(4200);
    } else {
      const x = addWord(t.word);
      if (known && !known.right) { x.rules.push(P.learn({ heard: known.heard, meant: known.w, source: 'roll call' })); x.state = 'learned'; }
      t.step = 'done';
      t.html = `<div class="tt"><span>${known && !known.right ? `Learned “${P.esc(known.heard)}” as ${P.esc(known.w)}` : 'Added to your roll call'}</span></div>
        <div class="cue" style="text-align:left">Nothing in your last take sounds like “${P.esc(t.word)}”${known && !known.right ? ', so the take stays as it is. Next time it’ll come out right.' : '. Say it on the Roll Call page any time.'}</div>`;
      drawTeach(); render();
      closeTeach(3600);
    }
    arm();
  },
};

V.onKey = (e, k) => {
  if (!S) return false;
  if (S.teach && e.key === 'Escape') { closeTeach(); return true; }
  if (k.inField) return false;
  if (S.view === 'call' && P.st.front === 'tess' && !S.teach) {
    if (e.key === 'ArrowRight' || (e.key === 'Enter' && (S.card === 'learned' || S.card === 'same'))) { next(); return true; }
    if (e.key === 'Enter') return true;
  }
  return false;
};

P.register(V);
})();
