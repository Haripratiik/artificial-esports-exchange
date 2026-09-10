/* Boot, socket, store, router.
 *
 * One WebSocket carries the live market at 20Hz; everything slow-moving
 * (contract terms, the agent roster, diagnostics, session configuration) comes
 * over REST and is refreshed on a much lazier timer. Pushing all of it down the
 * socket would mean re-serialising static contract specs twenty times a second
 * to say nothing new.
 *
 * The store is a plain object and the views are pure functions of it, so a tick
 * is just: merge, re-render. The one thing deliberately *not* re-rendered on a
 * tick is the order ticket, rewriting an input while somebody is typing into
 * it is the fastest way to make a trading screen unusable.
 */

import { clock, count, impliedProbability, money, move, percent, price, signed, cls,
         walkBook, worstCase } from './format.js';
import { lab, markets, matches, portfolio, research, trade } from './views.js';
import { countTo, press, revealAll } from './motion.js';

const store = {
  view: 'markets',
  symbol: null,
  snapshot: null,
  instruments: [],
  session: null,
  agents: [],
  diagnostics: null,
  depth: null,
  history: {},          // symbol -> array of mids, built client-side from ticks
  side: 'buy',
  generation: 0,
  query: '',
  // Whether to show what a contract will settle at. Off: a market that
  // prints its own answer has nothing left to discover.
  reveal: false,
  // Last mark seen per symbol, so a change can be shown as a change.
  marks: {},
  // Subjects whose instrument list is open on the markets screen. Held here
  // rather than read off the DOM because the renderer replaces whole regions,
  // and an <details>-style open flag would be lost on the next price tick.
  expanded: new Set(),
  // Which asset class the markets list is narrowed to, or null for all.
  classFilter: null,
};

const VIEWS = { markets, trade, portfolio, research, lab };
const main = document.getElementById('main');
const HISTORY_POINTS = 400;

/* ── socket ──────────────────────────────────────────────────────────── */

let socket = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.ack) return acknowledge(payload.ack);

    // A rebuild resets every series: the old prices belong to a market that no
    // longer exists, and splicing them onto the new one would draw a jump that
    // never happened.
    if (payload.generation !== store.generation) {
      store.generation = payload.generation;
      store.history = {};
      // A new market restarts its clock at zero, so a fresh event can carry
      // the same stamp as one already announced. Forgetting the old session's
      // events is the only way the new one's get heard.
      announced.clear();
      announcing = false;
      refreshSlow();
    }

    store.snapshot = payload;
    recordHistory(payload);
    announceRejections(payload);
    if (!store.symbol) store.symbol = Object.keys(payload.books)[0] ?? null;
    render();
  };

  socket.onclose = () => {
    toast('Disconnected, retrying', true);
    setTimeout(connect, 1200);
  };
}

function recordHistory(payload) {
  for (const [symbol, book] of Object.entries(payload.books || {})) {
    const series = (store.history[symbol] ||= []);
    const value = Number(book.mark);
    if (Number.isFinite(value)) series.push(value);
    if (series.length > HISTORY_POINTS) series.splice(0, series.length - HISTORY_POINTS);
  }
}

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function acknowledge(ack) {
  if (ack.ok === false) toast(ack.error ?? 'Rejected', true);
  // "Accepted" was a promise the acknowledgement cannot make. An order is
  // *queued* here, and travels a latency link before the venue ever looks at
  // it, so the collateral check, the price band and the closing bell all
  // answer later. Saying "Sent" and letting announceRejections carry what came
  // back is the difference between a status and a guess.
  else if (ack.speed === undefined) toast('Sent to the exchange');
}

/*
 * Say out loud when the venue refuses an order.
 *
 * Rejections arrive asynchronously, in the private event log, and the only
 * place that log is drawn is the Activity panel on the Portfolio screen,
 * which is not where anybody is standing when they press Place Order. So the
 * screen said "Accepted", the order never appeared in the book or the
 * blotter beside it, and nothing on that page ever explained why. Insufficient
 * collateral, a price band, a closed contract: all of them looked identical to
 * nothing happening.
 */
const announced = new Set();
let announcing = false;

function announceRejections(payload) {
  const fresh = [];
  for (const entry of payload.log ?? []) {
    const key = `${entry.t}|${entry.type}|${entry.symbol}|${entry.reason ?? ''}|${entry.order_id ?? ''}`;
    if (announced.has(key)) continue;
    announced.add(key);
    if (entry.type === 'reject') fresh.push(entry);
  }
  // The first snapshot carries whatever history the account already had, and
  // replaying old refusals as if they had just happened would be a lie about
  // when. Everything in it is marked seen and nothing is announced.
  if (!announcing) {
    announcing = true;
    return;
  }
  if (announced.size > 400) {
    for (const key of [...announced].slice(0, announced.size - 200)) announced.delete(key);
  }
  for (const entry of fresh) {
    toast(`${entry.symbol ?? 'Order'} refused: ${String(entry.reason ?? 'rejected').replace(/_/g, ' ')}`, true);
  }
}

/* ── slow data ───────────────────────────────────────────────────────── */

/**
 * The operator token, remembered per browser.
 *
 * Five routes on this server reach past the caller and change the market for
 * everyone in it, rebuild, halt, uncross, kill and revive, and they are now
 * gated. `POST /api/config` discards every account and working order for every
 * connected user, and `kill` takes an arbitrary agent id, so before the gate
 * one visitor could end another's session.
 *
 * Held in sessionStorage rather than localStorage: an operator token should
 * not outlive the tab it was pasted into. Wrapped in try/catch because a
 * browser with site data blocked throws on access rather than returning null,
 * and the Lab should degrade to asking again rather than failing to render.
 */
const OPERATOR_HEADER = 'arena-operator-token';

function operatorToken() {
  try {
    return sessionStorage.getItem('operator-token') || '';
  } catch (_) {
    return '';
  }
}

export function setOperatorToken(value) {
  try {
    if (value) sessionStorage.setItem('operator-token', value);
    else sessionStorage.removeItem('operator-token');
  } catch (_) { /* storage unavailable; the token lives for this render only */ }
}

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

/**
 * A call to one of the operator routes, carrying the token.
 *
 * They answer 404 rather than 401 without it, deliberately: a 401 confirms the
 * route exists and invites guessing. That means a missing token and a wrong
 * one are indistinguishable from here, so the message says what to do rather
 * than guessing which it was.
 */
async function operatorJson(url, options = {}) {
  const token = operatorToken();
  if (!token) {
    const entered = window.prompt(
      'Operator token. These controls change the market for everyone. '
      + 'The server prints a token at startup; set ARENA_OPERATOR_TOKEN to '
      + 'choose your own.',
    );
    if (!entered) throw new Error('operator token required');
    setOperatorToken(entered.trim());
  }
  const headers = { ...(options.headers || {}), [OPERATOR_HEADER]: operatorToken() };
  const response = await fetch(url, { ...options, headers });
  if (response.status === 404) {
    setOperatorToken('');
    throw new Error('operator token rejected, or no such control');
  }
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

async function refreshSlow() {
  try {
    const [instruments, session, agents] = await Promise.all([
      json('/api/instruments'),
      json('/api/session'),
      json('/api/agents'),
    ]);
    store.instruments = instruments.instruments;
    store.session = session;
    store.agents = agents.agents;
    render({ force: true });
  } catch (failure) {
    // A slow-data hiccup must not take the live view down with it.
    console.warn('slow refresh failed', failure);
  }
}

async function refreshSymbol() {
  if (!store.symbol) return;
  const symbol = store.symbol;
  try {
    const [depth, diagnostics] = await Promise.all([
      json(`/api/book/${encodeURIComponent(symbol)}?levels=18`),
      json(`/api/diagnostics/${encodeURIComponent(symbol)}`),
    ]);
    // Guard against a slow response landing after the user moved on.
    if (store.symbol !== symbol) return;
    store.depth = depth;
    store.diagnostics = diagnostics;
    render({ force: true });
  } catch (failure) {
    console.warn('symbol refresh failed', failure);
  }
}

/* ── render ──────────────────────────────────────────────────────────── */

/*
 * Rendering is decoupled from the socket.
 *
 * Snapshots arrive twenty times a second. Rebuilding the main panel that often
 * was wasteful, the display cannot show more than sixty frames and the socket
 * bursts can stack several messages into one frame, but the real damage was
 * not the CPU:
 *
 *   * **Focus was destroyed every 50ms.** Replacing the subtree removes the
 *     focused element, so focus fell back to <body>. Tabbing into the ladder
 *     and pressing Enter was impossible; the focus ring added for keyboard
 *     users had nothing to sit on.
 *   * **Scroll position reset**, so the tape and the agent roster snapped back
 *     to the top while you were reading them.
 *   * **Text selection was cleared**, so a price could not be copied.
 *
 * So: renders are coalesced into an animation frame (which also stops entirely
 * when the tab is hidden), the cheap header updates every frame because that is
 * where the fast-moving numbers are, and the expensive subtree is rebuilt on a
 * slower cadence with focus and scroll carried across.
 */

let ticketSignature = null;
let dirty = false;
let frame = null;
let lastHeavy = 0;

/*
 * How often each screen redraws, in milliseconds.
 *
 * A trading screen wants to feel live: eight passes a second is fast enough
 * that a ladder moves under you and slow enough that you can still click it.
 * A browsing grid does not: nobody reads a card seven times a second, and its
 * sparkline changes on every tick regardless of whether the price did, so at
 * the trading cadence the cards were rebuilding continuously to show a line
 * one pixel longer.
 */
const HEAVY_INTERVAL_MS = 120;
const VIEW_INTERVAL_MS = { markets: 500, portfolio: 400, research: 800, lab: 800 };

function render({ force = false } = {}) {
  dirty = true;
  if (force) lastHeavy = 0;
  if (frame === null) frame = requestAnimationFrame(paint);
}

function paint(now) {
  frame = null;
  renderHeader();
  renderIdentity();
  const interval = VIEW_INTERVAL_MS[store.view] ?? HEAVY_INTERVAL_MS;
  if (now - lastHeavy < interval) return;            // the next message re-arms
  lastHeavy = now;
  dirty = false;
  renderWatchlist();
  renderMain();
}

/** Where focus was, in terms that survive the subtree being replaced. */
function captureFocus() {
  const active = document.activeElement;
  if (!active || !main.contains(active)) return null;
  if (active.id) return `#${CSS.escape(active.id)}`;
  const { symbol, price, act, order } = active.dataset;
  if (act) return `[data-act="${CSS.escape(act)}"]`;
  if (price) return `[data-price="${CSS.escape(price)}"]`;
  if (symbol) return `[data-symbol="${CSS.escape(symbol)}"]`;
  return null;
}

/**
 * Update only what changed.
 *
 * The previous version replaced the whole subtree on every pass, and that is
 * what people saw as flicker. Destroying a node restarts every CSS animation on
 * it, so the tape flashed, rows re-entered and charts repainted several times a
 * second whether or not a single number had moved. Measured in the browser: a
 * tagged node did not survive three seconds.
 *
 * Every panel now carries a `data-region`. A pass renders the view into a
 * detached container, compares each region's markup against the one on screen,
 * and swaps only those that differ. In a quiet market that is nothing at all;
 * when one trade prints, it is one panel.
 */
function renderIdentity() {
  const label = document.getElementById('whoami-name');
  if (!label) return;
  const name = store.snapshot?.you?.name;
  const shown = name ?? '\u2014';
  if (label.textContent !== shown) label.textContent = shown;
}

function renderMain() {
  const view = VIEWS[store.view] ?? markets;
  const signature = `${store.view}:${store.symbol}`;
  const sameShape = ticketSignature === signature;

  const fresh = document.createElement('div');
  fresh.innerHTML = view(store);

  if (!sameShape) {
    main.replaceChildren(...fresh.childNodes);
    ticketSignature = signature;
    // Staggered only when the screen itself changed. On a tick it would be an
    // animation nobody asked for, eight times a second.
    revealAll(main.querySelectorAll('.card, .market-main > *, .figures > *'));
    bind();
    return;
  }

  const live = new Map(
    [...main.querySelectorAll('[data-region]')].map((n) => [n.dataset.region, n]),
  );
  const incoming = [...fresh.querySelectorAll('[data-region]')];

  // A region appearing or disappearing is a change of shape rather than of
  // content, and is the one case that still needs the screen rebuilt.
  const sameRegions =
    incoming.length === live.size && incoming.every((n) => live.has(n.dataset.region));
  if (!sameRegions) {
    main.replaceChildren(...fresh.childNodes);
    bind();
    return;
  }

  const selector = captureFocus();
  let touched = 0;
  for (const next of incoming) {
    const node = live.get(next.dataset.region);
    if (node.innerHTML === next.innerHTML) continue;   // nothing moved here
    // The ticket holds live input and is never rewritten under a keystroke;
    // its figures are patched in place by updatePreview instead.
    if (next.querySelector('#ticket')) continue;

    const scrolled = [...node.querySelectorAll('.panel-body')].map((n) => n.scrollTop);
    node.innerHTML = next.innerHTML;
    [...node.querySelectorAll('.panel-body')].forEach((n, i) => {
      if (scrolled[i]) n.scrollTop = scrolled[i];
    });
    touched += 1;
  }

  if (touched) {
    if (selector) main.querySelector(selector)?.focus({ preventScroll: true });
    bind();
  }
}

function renderHeader() {
  const s = store.snapshot;
  if (!s) return;
  document.getElementById('clock').textContent = clock(s.clock);
  document.getElementById('events').textContent = count(s.events);
  // Money slides to its new figure. A number that moves says "this changed and
  // by roughly this much", which the digits alone cannot; and because there is
  // always a static cue beside it, nothing is lost if the motion is switched
  // off or missed.
  countTo(document.getElementById('equity'), Number(s.account.equity), money);

  const pnl = document.getElementById('pnl');
  countTo(pnl, Number(s.account.pnl), money);
  pnl.className = `mono ${cls(Number(s.account.pnl))}`;

  const conserved = Number(s.conservation) === 0;
  const health = document.getElementById('health');
  health.classList.toggle('bad', !conserved);
  document.getElementById('health-text').textContent = conserved
    ? 'Conserved'
    : `Leak of ${s.conservation}`;
}

let watchlistHtml = '';

function renderWatchlist() {
  const s = store.snapshot;
  if (!s) return;
  const list = document.getElementById('watchlist');

  const hits = Object.entries(s.books).filter(([sym, book]) => matches(sym, book, store.query));
  // A rail is navigation, not a directory.
  //
  // It rendered every listed symbol, which was right for a fixed listing of
  // forty-seven and stopped being right the moment matches began listing: one
  // solo match carries 270 contracts, so the rail became 355 rows deep while
  // the browse view beside it correctly collapsed that same match to a single
  // line. Scrolling a market you cannot see is not a watchlist.
  //
  // Capped rather than filtered, so nothing is unreachable. The search box
  // already narrows on symbol and on contract text, so typing a competitor or
  // a match reaches anything the cap hides, and the count below says how much
  // there is to reach.
  const CAP = 40;
  const shown = hits.slice(0, CAP);
  const hidden = hits.length - shown.length;
  const html = (shown.length
    ? shown
    : [])
    .map(([symbol, book]) => {
      const series = (store.history[symbol] || []).slice(-60);
      const first = series.find((v) => v != null);
      const last = series.length ? series[series.length - 1] : null;
      const change = first != null && last != null ? last - first : 0;
      const pct = first ? (change / first) * 100 : 0;
      const current = symbol === store.symbol;
      // Motion that carries information rather than decorating: a price that
      // just moved flashes in the direction it moved. This is the one
      // animation on a trading screen that earns its place, because it says
      // something the static number cannot, that this is new.
      const now = Number(book.mark);
      const was = store.marks[symbol];
      const tick = !Number.isFinite(was) || now === was ? '' : now > was ? 'tick-up' : 'tick-down';
      store.marks[symbol] = now;
      return `<button type="button" class="watch ${tick}" data-symbol="${symbol}"
                   ${current ? 'aria-current="true"' : ''}
                   aria-label="${symbol}, ${price(book.mark)}, ${signed(pct)} percent">
        <span class="sym">${symbol}</span>
        <span class="px ${cls(change)}">${price(book.mark)}</span>
        <span class="chg ${cls(change)}">${move(change, first).text}</span>
      </button>`;
    })
    .join('') || `<p class="empty">No market matches that.</p>`;

  const footer = hidden > 0
    ? `<p class="empty">${hidden} more. Search to narrow.</p>`
    : '';
  const rendered = html + footer;

  // Same reason the panels are diffed: rewriting this list recreates every row,
  // which restarts the tick highlight on rows whose price never moved. Twelve
  // animations were running at once on a list of seven.
  if (rendered === watchlistHtml) return;
  watchlistHtml = rendered;
  list.innerHTML = rendered;
}

function sparklineFor(series) {
  // Imported lazily through views' formatter to keep one implementation.
  return series.length > 1 ? sparkInline(series) : '';
}

function sparkInline(values) {
  const pts = values.filter((v) => Number.isFinite(v));
  if (pts.length < 2) return '';
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = hi - lo || 1;
  const step = 100 / (pts.length - 1);
  const d = pts
    .map((v, i) => `${i ? 'L' : 'M'}${(i * step).toFixed(1)},${(14 - ((v - lo) / span) * 12).toFixed(1)}`)
    .join('');
  const rising = pts[pts.length - 1] >= pts[0];
  return `<svg class="chart" viewBox="0 0 100 16" preserveAspectRatio="none">
    <path d="${d}" fill="none" stroke="${rising ? 'var(--up)' : 'var(--down)'}"
          stroke-width="1" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

/* ── events ──────────────────────────────────────────────────────────── */

function bind() {
  main.querySelectorAll('[data-symbol]').forEach((node) => {
    if (node.dataset.act) return;              // control buttons carry a symbol too
    node.addEventListener('click', () => select(node.dataset.symbol));
  });

  main.querySelectorAll('[data-class]').forEach((node) => {
    node.addEventListener('click', () => {
      store.classFilter = node.dataset.class || null;
      render({ force: true });
    });
  });

  main.querySelectorAll('[data-subject]').forEach((node) => {
    node.addEventListener('click', () => {
      const key = node.dataset.subject;
      if (store.expanded.has(key)) store.expanded.delete(key);
      else store.expanded.add(key);
      render({ force: true });
    });
  });

  main.querySelectorAll('.lad-row').forEach((row) => {
    row.addEventListener('click', () => {
      // Clicking a level fills the limit box, so open Advanced to show where it
      // went, because otherwise the click appears to do nothing at all.
      const advanced = main.querySelector('.advanced');
      if (advanced) advanced.open = true;
      const input = document.getElementById('t-px');
      if (input) input.value = row.dataset.price;
      updatePreview();
    });
  });

  const sendButtonEl = main.querySelector('#t-send');
  if (sendButtonEl) sendButtonEl.addEventListener('pointerdown', () => press(sendButtonEl));

  main.querySelectorAll('.sides button').forEach((button) => {
    button.addEventListener('click', () => {
      store.side = button.dataset.side;
      main.querySelectorAll('.sides button').forEach((b) =>
        b.setAttribute('aria-pressed', String(b.dataset.side === store.side)));
      updatePreview();
    });
  });

  const sendButton = document.getElementById('t-send');
  if (sendButton) sendButton.addEventListener('click', submitOrder);

  // The preview has to follow keystrokes, not renders: the ticket is
  // deliberately not re-rendered while someone is typing into it.
  ['t-qty', 't-px', 't-tif', 't-stop', 't-show'].forEach((id) => {
    const field = document.getElementById(id);
    if (field && !field.dataset.wired) {
      field.dataset.wired = '1';
      field.addEventListener('input', updatePreview);
      field.addEventListener('change', updatePreview);
    }
  });
  main.querySelectorAll('.quick button').forEach((button) => {
    button.addEventListener('click', () => {
      const field = document.getElementById('t-qty');
      if (field) field.value = button.dataset.qty;
      updatePreview();
    });
  });
  updatePreview();

  main.querySelectorAll('[data-act]').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      act(button.dataset.act, button.dataset);
    });
  });

  main.querySelectorAll('.spoiler').forEach((node) => {
    node.addEventListener('toggle', () => {
      store.reveal = node.open;
      render({ force: true });
    });
    node.open = store.reveal;
  });

  const whoami = document.getElementById('whoami');
  if (whoami && !whoami.dataset.wired) {
    whoami.dataset.wired = '1';
    whoami.addEventListener('click', rename);
  }

  const apply = document.getElementById('c-apply');
  if (apply) apply.addEventListener('click', rebuild);
}

/**
 * What the order would cost, and what it could lose.
 *
 * Every exchange shows this before you commit, and this one could not: you
 * typed a size into a box and pressed send with no idea what you were about to
 * pay. It is all computable from the depth already on screen.
 *
 * The worst case is exact rather than estimated, because every contract here
 * settles inside a known interval, a long position cannot lose more than the
 * distance from its price down to the floor, and a short cannot lose more than
 * the distance up to the ceiling. That is the same arithmetic the venue uses to
 * hold collateral, so the number on the ticket is the number being reserved.
 */
function updatePreview() {
  const panel = document.getElementById('t-preview');
  if (!panel) return;

  const book = store.snapshot?.books?.[store.symbol];
  const quantity = Number(document.getElementById('t-qty')?.value);
  if (!book || !Number.isFinite(quantity) || quantity <= 0) {
    panel.innerHTML = '';
    return;
  }

  const buying = store.side === 'buy';
  const raw = document.getElementById('t-px')?.value.trim() ?? '';
  const limit = raw === '' ? null : Number(raw);
  const stop = document.getElementById('t-stop')?.value.trim() ?? '';
  if (stop !== '') {
    // A stop does not trade now, so a cost taken off today's book would be a
    // guess dressed as a quotation. What it *is* is the size of the
    // commitment, which is what the venue reserves against, and the only
    // honest thing to show before it triggers.
    //
    // It was `stop * quantity`, which is the notional, not the reservation.
    // Those agree only for a long on a claim whose floor is zero. A sell of
    // ten futures stopped at 4,600 said 46,000 against the 54,000 the venue
    // actually held; a spread stopped at zero said nothing was reserved at
    // all, against 100,000. Understating every time, which is the direction
    // that gets the next order refused for a reason the screen never showed.
    const reserved = worstCase(quantity, stop, book.bounds, buying);
    panel.innerHTML = `<dl>
      <div><dt>Waits until</dt><dd class="mono">${price(stop)}</dd></div>
      <div><dt>Then</dt><dd class="mono">${limit === null
        ? 'goes to market' : `works a limit at ${price(limit)}`}</dd></div>
      <div><dt>Reserved now</dt><dd class="mono">${
        reserved == null ? '--' : money(reserved)}</dd></div>
    </dl>
    <p class="note">Nobody can see a stop while it waits, and it is
      collateralised from the moment you place it.</p>`;
    return;
  }

  // Marketable orders walk the opposite side; a resting limit fills at its own
  // price or better, so its own price is the honest estimate.
  const ladder = buying
    ? (store.depth?.asks ?? book.asks ?? [])
    : (store.depth?.bids ?? book.bids ?? []);
  const marketable =
    limit === null ||
    (ladder.length && (buying ? limit >= Number(ladder[0][0]) : limit <= Number(ladder[0][0])));

  const walk = marketable ? walkBook(ladder, quantity) : null;
  const average = walk ? walk.average : limit;
  if (!Number.isFinite(average)) {
    panel.innerHTML = '';
    return;
  }

  // The same number as "Reserved now", by the same expression, because they
  // are the same quantity: what this position can lose is what the venue
  // holds against it. Two formulas for one figure is how one of them drifted.
  const risk = worstCase(quantity, average, book.bounds, buying);

  const payoff = book.contract?.payoff;
  const odds = impliedProbability(average, payoff);

  const rows = [
    ['Avg price', price(average)],
    [buying ? 'Cost' : 'Proceeds', money(average * quantity)],
  ];
  if (risk != null) rows.push(['Max loss', money(risk)]);

  // A binary pays a fixed amount, so the useful framing is what you win and
  // what you staked, not a notional.
  if (payoff?.kind === 'binary') {
    const payout = Number(payoff.payout) * quantity;
    const stake = buying ? average * quantity : (Number(payoff.payout) - average) * quantity;
    rows.push([buying ? 'Pays if yes' : 'Pays if no', money(buying ? payout : payout)]);
    rows.push(['Profit if right', money(payout - stake)]);
    if (odds != null) rows.push(['Implied odds', percent(buying ? odds : 1 - odds)]);
  }

  const warning = walk && !walk.complete
    ? `<div class="warn">Book holds only ${count(walk.filled)}, ${count(walk.shortfall)} would not fill.</div>`
    : '';

  panel.innerHTML =
    rows.map(([label, value]) =>
      `<div><span>${label}</span><b class="mono">${value}</b></div>`).join('') + warning;
}

/*
 * Send exactly the order the ticket describes.
 *
 * The stop trigger and the iceberg size were collected, previewed and then
 * dropped on the floor: the panel said "Waits until 8,900.00" and "Reserved
 * now ...", and what left the browser was a plain limit at 9,000 that rested
 * in full, visible, immediately. Both fields have been supported by the server
 * and by the venue the whole time; the submitter simply never read them.
 *
 * Raw strings rather than `Number`, deliberately: `Number('')` is 0 and
 * `Number('9,233.75')` is NaN, and JSON turns NaN into null, so a size box the
 * server should refuse would have arrived looking like a field nobody filled
 * in. The server parses and answers in the terms of the box.
 */
function submitOrder() {
  const field = (id) => document.getElementById(id)?.value.trim() ?? '';
  send({
    action: 'submit',
    symbol: store.symbol,
    side: store.side,
    quantity: field('t-qty'),
    price: field('t-px') || null,
    tif: field('t-tif'),
    stop: field('t-stop') || null,
    display: field('t-show') || 0,
  });
}

async function act(action, data) {
  // The symbol travels with the id, because an id is only unique inside its
  // own book: order 5 exists on every contract at once.
  if (action === 'cancel') {
    return send({ action: 'cancel', order_id: Number(data.order), symbol: data.symbol });
  }
  if (action === 'cancel_all') return send({ action: 'cancel_all' });
  // The bluntest control an exchange has, and the one that has to keep
  // working when everything else is going wrong. What comes back is the list
  // of books it pulled orders from, because "done" is not an answer an
  // operator can act on.
  if (action === 'kill' || action === 'revive') {
    try {
      const result = await operatorJson(`/api/participant/${encodeURIComponent(data.agent)}/${action}`,
                                { method: 'POST' });
      toast(action === 'kill'
        ? `Stopped ${result.agent_id}, pulled ${(result.symbols ?? []).length} book(s)`
        : `${result.agent_id} let back in`);
      refreshSlow();
    } catch (failure) {
      toast(String(failure), true);
    }
    return;
  }
  if (action === 'flatten') {
    // Destructive and irreversible: it sells every position at market, and
    // there is no undo once the prints land.
    const open = (store.snapshot?.account?.positions ?? []).length;
    if (!open) return toast('Already flat.');
    if (!confirm(`Close ${open} position${open === 1 ? '' : 's'} at market? This cannot be undone.`)) return;
    return send({ action: 'flatten' });
  }

  if (action === 'halt' || action === 'uncross') {
    try {
      const result = await operatorJson(
        `/api/session/${encodeURIComponent(data.symbol)}/${action}`, { method: 'POST' });
      toast(result.ok ? `${data.symbol} ${result.session}` : result.error, !result.ok);
      refreshSlow();
    } catch (failure) {
      toast(String(failure), true);
    }
  }
}

/**
 * Change the name other traders see.
 *
 * A prompt rather than a modal, deliberately: there is exactly one field, no
 * validation the server does not already do, and a dialog for it would be more
 * chrome than the decision deserves. There is no password here and the page
 * does not pretend there is: the account lives in a signed cookie, and losing
 * the cookie loses the account. That is the honest shape for an exchange whose
 * capital is imaginary.
 */
async function rename() {
  const current = store.snapshot?.you?.name ?? '';
  const chosen = window.prompt('Name other traders will see', current);
  if (chosen === null) return;
  try {
    const result = await json('/api/me', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: chosen }),
    });
    toast(`Trading as ${result.name}`);
  } catch (failure) {
    toast(String(failure), true);
  }
}

async function rebuild() {
  const band = document.getElementById('c-band').value.trim();
  const body = {
    seed: Number(document.getElementById('c-seed').value),
    flow_traders: Number(document.getElementById('c-flow').value),
    fees: document.getElementById('c-fees').value,
    price_band: band === '' ? null : Number(band),
    arbitrageur: document.getElementById('c-arb').checked,
    makers: Number(document.getElementById('c-makers').value),
    opening_auction: document.getElementById('c-auction').checked,
    surface: document.getElementById('c-surface').checked,
    mechanism: document.getElementById('c-mechanism').value,
    speed: store.snapshot?.speed ?? 1,
  };
  try {
    const result = await operatorJson('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    toast(`Rebuilt, generation ${result.generation}`);
    store.history = {};
    refreshSlow();
  } catch (failure) {
    toast(String(failure), true);
  }
}

function select(symbol) {
  if (!symbol || symbol === store.symbol) {
    if (symbol) store.view = 'trade';
  } else {
    store.symbol = symbol;
    store.depth = null;
    store.diagnostics = null;
    store.view = 'trade';
    refreshSymbol();
  }
  syncNav();
  syncUrl();
  render({ force: true });
}

function syncNav() {
  document.querySelectorAll('#nav button').forEach((b) => {
    if (b.dataset.view === store.view) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
}

/*
 * The URL carries which screen and which instrument you are looking at, so a
 * view can be linked, bookmarked and reloaded into. `replaceState` rather than
 * `pushState`: clicking between instruments is browsing a live screen, not
 * navigating, and filling the back button with every glance would make the
 * gesture useless.
 */
function syncUrl() {
  const url = new URL(location.href);
  url.searchParams.set('view', store.view);
  if (store.symbol) url.searchParams.set('symbol', store.symbol);
  history.replaceState(null, '', url);
}

function readUrl() {
  const params = new URLSearchParams(location.search);
  const view = params.get('view');
  if (view && view in VIEWS) store.view = view;
  const symbol = params.get('symbol');
  if (symbol) store.symbol = symbol;
}

document.getElementById('nav').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  store.view = button.dataset.view;
  syncNav();
  syncUrl();
  if (store.view === 'research') refreshSymbol();
  if (store.view === 'lab') refreshSlow();
  render({ force: true });
});

document.getElementById('watchlist').addEventListener('click', (event) => {
  const row = event.target.closest('.watch');
  if (row) select(row.dataset.symbol);
});

const search = document.getElementById('q');
search.addEventListener('input', () => {
  store.query = search.value;
  // Filtering changes what is on screen rather than what the numbers say, so
  // it repaints immediately instead of waiting out the panel interval.
  render({ force: true });
});

const speed = document.getElementById('speed');
speed.addEventListener('input', () => {
  document.getElementById('speed-label').textContent = `${Number(speed.value).toFixed(1)}×`;
  send({ action: 'speed', value: Number(speed.value) });
});

/* ── toast ───────────────────────────────────────────────────────────── */

let toastTimer = null;
const toaster = document.getElementById('toaster');

function toast(message, bad = false) {
  // Rendered into the document's polite live region rather than appended to
  // the body, so a screen reader hears the acknowledgement a sighted user sees.
  toaster.replaceChildren();
  const node = document.createElement('p');
  node.className = `toast${bad ? ' bad' : ''}`;
  node.textContent = message;
  toaster.append(node);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toaster.replaceChildren(), 2600);
}

/* ── go ──────────────────────────────────────────────────────────────── */

readUrl();
syncNav();
connect();
refreshSlow();
setInterval(refreshSlow, 8000);
setInterval(refreshSymbol, 2500);
