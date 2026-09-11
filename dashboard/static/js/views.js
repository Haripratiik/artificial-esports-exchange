/* The screens.
 *
 * Each view is a pure function from store state to HTML. Keeping them pure
 * means the socket can re-render without any view holding state that could
 * drift from the market's. The one exception is the order ticket, which holds
 * live input and is moved across a re-render rather than rebuilt.
 *
 * The information order on the trade screen is not arbitrary. Prediction-market
 * design guidance is consistent about it, and this interface previously had it
 * backwards:
 *
 *   1. what the contract is, and the odds
 *   2. **how it resolves**, above the fold, not buried
 *   3. the price history
 *   4. recent trades
 *   5. the order book, *collapsed by default*
 *   6. contract specifications
 *
 * Leading with a depth ladder is right for an operator and wrong for everyone
 * else: it was the most technical thing on the page and it was the first thing
 * you saw, while the resolution rules were squeezed into a strip above it,
 * which is the documented anti-pattern, burying the terms beneath the pricing.
 */

import {
  clock, cls, count, describe, esc, impliedProbability, money, move, percent,
  price, priceChart, signed, sparkline,
} from './format.js';

/* ── markets ─────────────────────────────────────────────────────────── */

/**
 * Asset classes, in the words a person would use.
 *
 * The venue's own vocabulary is precise and unhelpful for browsing: "event" is
 * a prediction market and "call"/"put" are options. Grouping under these makes
 * the shape of the exchange visible, which a flat grid of seven tiles does not
 *, you could not tell, looking at the old page, that this venue lists
 * prediction markets and options at all.
 */
const CLASS_GROUPS = [
  ['event', 'Prediction Markets', 'Pays a fixed amount if the outcome happens.'],
  ['future', 'Futures', 'Settles at the measured rate itself.'],
  ['call', 'Options', 'The right to what lies past a strike.'],
  ['put', 'Options', 'The right to what lies past a strike.'],
  ['equity', 'Shares', 'Pays out every week it is alive, then expires. Worth the payments that are left.'],
  ['volatility', 'Volatility', 'How unevenly a competitor performs across the formats it plays, not how well.'],
  ['commodity', 'Commodities', 'An amount delivered in one week, not a rate. Each week trades separately.'],
  ['spread', 'Spreads', 'One competitor priced against another.'],
  ['index', 'Indices', 'A weighted basket of several competitors.'],
];

const GROUP_ORDER = ['Prediction Markets', 'Futures', 'Shares', 'Commodities',
                     'Options', 'Volatility', 'Spreads', 'Indices', 'Other'];

/**
 * The subject a contract is written on, which is what a person thinks they are
 * trading. Not the same thing as the underlying: `BASTION_SOLO_W1` and
 * `BASTION_SOLO_W4` are different underlyings, because the observation windows
 * differ, but they are the same subject and belong on one row.
 */
export function subjectsOf(node) {
  if (!node) return [];
  if (node.kind === 'single') return node.ref?.subject ? [node.ref.subject] : [];
  if (node.kind === 'difference') {
    return [...subjectsOf(node.left), ...subjectsOf(node.right)];
  }
  if (node.kind === 'basket') {
    return (node.legs || []).flatMap((l) => subjectsOf(l.leg));
  }
  return [];
}

/**
 * The match a contract belongs to, or null if it is not on a match.
 *
 * A match ref carries its mode and its match number, so grouping is read off
 * the contract rather than parsed out of the symbol. Grouping these by subject
 * instead would scatter one match across the whole roster: a ten-competitor
 * solo match lists 270 contracts and every one of them names a different
 * competitor, so the board would show the roster rather than the match.
 */
function refsOf(node) {
  if (!node) return [];
  if (node.kind === 'single') return node.ref ? [node.ref] : [];
  if (node.kind === 'difference') return [...refsOf(node.left), ...refsOf(node.right)];
  if (node.kind === 'basket') return (node.legs || []).flatMap((l) => refsOf(l.leg));
  return [];
}

export function matchOf(book) {
  // Walked rather than read off the top, because a head to head contract is a
  // difference of two competitors and its match tag sits on the legs. Reading
  // only the root left ninety of a solo match's contracts ungrouped, and they
  // came back as their own rows named after the pair, which is the crowded
  // board this grouping exists to avoid.
  const refs = refsOf(book?.contract?.underlying);
  for (const ref of refs) {
    const tag = (ref?.maps || []).find((m) => String(m).startsWith('match-'));
    if (!tag) continue;
    const mode = (ref.modes || []).find((m) => m && m !== 'ALL') || 'match';
    return mode.toUpperCase() + ' #' + String(tag).slice('match-'.length);
  }
  return null;
}

export function subjectOf(book) {
  const match = matchOf(book);
  if (match) return match;
  const found = [...new Set(subjectsOf(book?.contract?.underlying))];
  if (!found.length) return 'Other';
  return found.length === 1 ? found[0] : found.join(' + ');
}

export function groupOf(assetClass) {
  return CLASS_GROUPS.find(([key]) => key === assetClass)?.[1] ?? 'Other';
}

function groupBlurb(title) {
  return CLASS_GROUPS.find(([, name]) => name === title)?.[2] ?? '';
}


/**
 * Discovery. A card carries the question, the odds, activity and time left,
 * and one action.
 *
 * Deliberately *not* on a card: order-book depth, tick size, settlement bounds,
 * spec digests. Those are specifications, and putting them on a browsing
 * surface is the fastest way to make an exchange feel like a database viewer.
 */
export function markets(store) {
  const { snapshot, instruments, history } = store;
  const books = snapshot?.books ?? {};
  const symbols = Object.keys(books).filter((s) => matches(s, books[s], store.query));
  if (!symbols.length) {
    return `<div class="view"><div class="onboard">
      <h2>${store.query ? 'Nothing matches that' : 'Connecting to the exchange&hellip;'}</h2>
      <p>${store.query
        ? `No market matches &ldquo;${esc(store.query)}&rdquo;. Try a ticker, a competitor, or a class like future or event.`
        : 'Contracts here settle on matches in a synthetic esport, and on the statistics those matches produce.'}</p>
    </div></div>`;
  }

  const stats = (symbol) => {
    const book = books[symbol];
    const series = (history[symbol] || []).slice(-90);
    const first = series.find((v) => v != null);
    const last = series.length ? series[series.length - 1] : null;
    const change = first != null && last != null ? last - first : 0;
    const odds = impliedProbability(book.mark, book.contract?.payoff);
    return {
      book, series, first, change,
      headline: odds != null ? percent(odds, 0) : price(book.mark),
      session: snapshot.sessions?.[symbol] ?? 'continuous',
    };
  };

  /* One instrument, one line. Identity left, every number right on a shared
   * grid, so scanning a list is a glance down two fixed columns rather than a
   * re-read of each item. */
  const row = (symbol) => {
    const { book, series, first, change, headline, session } = stats(symbol);
    const current = symbol === store.symbol;
    return `<button type="button" class="mrow" data-symbol="${esc(symbol)}"
                 data-session="${esc(session)}" ${current ? 'aria-current="true"' : ''}
                 aria-label="${esc(symbol)}, ${headline}, ${move(change, first).text}">
      <span class="m-caret" aria-hidden="true"></span>
      <span class="m-id">
        <b class="m-sym">${esc(symbol)}</b>
        <span class="question">${question(book.contract)}</span>
      </span>
      <span class="m-kind">${session !== 'continuous'
        ? `<b class="badge ${esc(session)}">${esc(session.replace('_', ' '))}</b>`
        : esc(book.class ?? '')}</span>
      <span class="m-px mono ${cls(change)}">${headline}</span>
      <span class="m-chg mono ${cls(change)}">${move(change, first).text}</span>
      <span class="m-vol mono">${count(book.trades)}</span>
      <span class="m-spark" aria-hidden="true">${sparkline(series, { width: 64, height: 16 })}</span>
    </button>`;
  };

  /* Grouped by SUBJECT, not by asset class.
   *
   * Every listing venue that carries a lot of markets does this, and the
   * numbers are why: Kalshi's browse page shows 60 entries covering 574
   * tradeable markets, and Polymarket 40 entries covering 1,625. Both list the
   * *event* and put the instrument count on the row. This screen listed one
   * card per instrument, a 1:1 ratio, which is why 28 markets felt heavier
   * than Kalshi's hundreds. Kalshi at 1:1 would render 574 cards.
   *
   * The ratio is now the whole argument rather than a refinement. Matches
   * carry most of the listing, and one solo match is 270 contracts on its own,
   * so 1:1 would render a page nobody could read. Measured on the live
   * listing: 355 markets present as 9 rows, and 316 of them are prediction
   * markets against 8 before matches existed.
   *
   * Asset class was the old grouping and is now a field on the row. A future,
   * a call, a put and a weekly on one competitor are one thing to a person,
   * and splitting them across four sections is what made 28 look like a wall. */
  /* Asset class as a filter rather than a heading.
   *
   * It was the grouping, which scattered one competitor's future, calls, puts
   * and weeklies across four separate sections, so a person looking for
   * "VANTA" had to visit four places and 28 instruments read as more than 28.
   * As a chip it stays visible, stays countable, and narrows the list instead
   * of fragmenting it. */
  const classCounts = new Map();
  for (const symbol of symbols) {
    const title = groupOf(books[symbol].class);
    classCounts.set(title, (classCounts.get(title) ?? 0) + 1);
  }
  const active = store.classFilter && classCounts.has(store.classFilter)
    ? store.classFilter
    : null;
  const chips = `<div class="chips" role="group" aria-label="Filter by asset class">
    <button type="button" class="chip${active ? '' : ' on'}" data-class="">
      All <b class="mono">${symbols.length}</b>
    </button>
    ${GROUP_ORDER.filter((title) => classCounts.has(title)).map((title) => `
      <button type="button" class="chip${active === title ? ' on' : ''}"
              data-class="${esc(title)}" aria-pressed="${active === title}">
        ${esc(title)} <b class="mono">${classCounts.get(title)}</b>
      </button>`).join('')}
  </div>`;

  const shown = active
    ? symbols.filter((s) => groupOf(books[s].class) === active)
    : symbols;

  const groups = new Map();
  for (const symbol of shown) {
    const key = subjectOf(books[symbol]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(symbol);
  }

  const ordered = [...groups.entries()].sort(
    (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
  );

  const sections = ordered.map(([subject, syms]) => {
    // The group speaks with its most representative instrument: the outright
    // future where there is one, since that is the subject's own price rather
    // than a claim about part of it.
    const lead = syms.find((s) => books[s].class === 'future') ?? syms[0];
    const { series, first, change, headline } = stats(lead);
    // A search narrows to what matched, so opening everything is what the
    // person asked for. Otherwise only the group they are trading is open.
    const open = store.query || active
      ? true
      : (store.expanded?.has(subject) ?? false) || syms.includes(store.symbol);

    return `<section class="subject" data-region="subject:${esc(subject)}">
      <button type="button" class="subject-head" data-subject="${esc(subject)}"
              aria-expanded="${open}">
        <span class="s-caret" aria-hidden="true">${open ? '▾' : '▸'}</span>
        <span class="s-name">${esc(subject)}</span>
        <span class="s-px mono ${cls(change)}">${headline}</span>
        <span class="s-chg mono ${cls(change)}">${move(change, first).text}</span>
        <span class="s-spark" aria-hidden="true">${sparkline(series, { width: 64, height: 16 })}</span>
        <span class="s-count mono">${syms.length} market${syms.length === 1 ? '' : 's'}</span>
      </button>
      ${open ? `<div class="subject-body">${syms.map(row).join('')}</div>` : ''}
    </section>`;
  }).join('');

  return `<div class="view markets-list">
    ${chips}
    <div class="list-head">
      <span aria-hidden="true"></span>
      <span class="lh-sym">Market</span><span class="lh-kind">Class</span>
      <span class="lh-px">Price</span><span class="lh-chg">Change</span>
      <span class="lh-vol">Trades</span><span class="lh-spark"></span>
    </div>
    ${sections}
  </div>`;
}

/**
 * Does this market match what someone typed?
 *
 * Searches the ticker, the asset class and the plain-language question, because
 * those are the three things a person might have in mind. Matching only the
 * ticker would mean you had to already know the ticker, which defeats the
 * point of searching.
 */
export function matches(symbol, book, query) {
  const needle = (query ?? '').trim().toLowerCase();
  if (!needle) return true;
  const haystack = [
    symbol,
    book?.class ?? '',
    question(book?.contract),
    subjectName(book?.contract?.underlying),
  ].join(' ').toLowerCase();
  return needle.split(/\s+/).every((word) => haystack.includes(word));
}

/** The contract as a question, which is how a person holds it in their head. */
/**
 * What to show where a spread would go, when there is no spread.
 *
 * A book in a call phase is crossed on purpose, so it has no spread and does
 * have an indicative price, the number the auction would clear at right now,
 * which is what a real venue publishes during one.
 */
export function marketState(book) {
  if (!book?.session || book.session === 'continuous') return null;
  const words = { pre_open: 'Opening auction', auction: 'Halted, auction',
                  closed: 'Closed' };
  const label = words[book.session] ?? book.session;
  return book.indicative ? `${label}, indicative ${price(book.indicative)}` : label;
}

export function question(contract) {
  const p = contract?.payoff;
  if (!p) return '';
  const subject = esc(subjectName(contract?.underlying));
  // Only a single-competitor contract reads naturally with the metric
  // attached. A basket or a difference already names what it measures, and
  // forcing the metric in produced "the duellist index's win rate" and "VANTA
  // vs QUILL's win rate", grammatical wreckage on the most-read line of the
  // whole exchange.
  const simple = contract?.underlying?.kind === 'single';
  const what = simple ? `${subject}'s ${esc(metricName(contract.underlying))}` : subject;

  if (p.kind === 'binary') {
    const direction = String(p.comparison).includes('>') ? 'above' : 'below';
    return `Will ${what} finish ${direction} ${level(p.threshold)}?`;
  }
  // For a delivery contract the week *is* the contract: the same deliverable in
  // a different week is a different instrument, which is what gives a commodity
  // its term structure. Saying "where it settles" would hide the only thing
  // distinguishing one rung from the next.
  // A share is worth the stream, so the stream is the question. Saying
  // "where it settles" would be the one number that is always zero.
  const stream = contract?.distribution;
  if (stream) {
    return `A share of ${what}, paid out over ${count(stream.periods)} weeks`;
  }
  // A volatility contract is written on a spread, so saying "where the rate
  // settles" would name the wrong moment entirely.
  if (contract?.underlying?.ref?.kind === 'dispersion') {
    return `How unevenly ${subject} performs across the formats it plays`;
  }
  if (contract?.underlying?.ref?.kind === 'quantity') {
    return `How many thousand matches ${subject} plays, delivered ${esc(week(contract))}`;
  }
  if (p.kind === 'call') return `${what} above ${level(p.strike)} at settlement`;
  if (p.kind === 'put') return `${what} below ${level(p.strike)} at settlement`;
  return `Where ${what} settles`;
}

/**
 * The competitor a contract is written on.
 *
 * The metric reference arrives under `ref`, not `metric`, and reading the wrong
 * key made every question on the exchange read "Will the metric finish above
 * 0.48?", which is a contract nobody could identify. It failed silently
 * because the fallback was a plausible English phrase rather than an error.
 */
/** "week to 7 Sep", from a contract's expiry. */
function week(contract) {
  const raw = contract?.expiry;
  if (!raw) return 'that week';
  const when = new Date(raw);
  if (Number.isNaN(when.getTime())) return 'that week';
  return `week to ${when.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}`;
}

/**
 * A strike or a threshold, written the way a person writes it.
 *
 * These arrive as JSON numbers, so they arrive as doubles, and a double does
 * not hold 0.06. Interpolated raw, the most-read line on the exchange asked
 * "Will VANTA's win rate finish above 0.060000000000000005?", which is a
 * contract nobody would take seriously about a venue whose whole claim is that
 * its arithmetic is exact.
 *
 * Ten significant figures is far more precision than any level this venue
 * lists carries, and stops short of where the representation noise lives. The
 * prices themselves never take this path: they are strings on the wire for
 * exactly this reason, and only the contract's own terms come across as
 * numbers.
 */
function level(raw) {
  const value = Number(raw);
  if (!Number.isFinite(value)) return esc(String(raw));
  return String(Number(value.toPrecision(10)));
}

function subjectName(underlying) {
  if (!underlying) return 'the metric';
  if (underlying.kind === 'single') return underlying.ref?.subject ?? 'the metric';
  if (underlying.kind === 'difference') {
    return `the gap between ${subjectName(underlying.left)} and ${subjectName(underlying.right)}`;
  }
  if (underlying.kind === 'basket') return 'the duellist index';
  return 'the metric';
}

/** "eliminations per match", from "eliminations_per_match". */
function metricName(underlying) {
  const raw = underlying?.ref?.metric ?? underlying?.left?.ref?.metric;
  return raw ? String(raw).replace(/_/g, ' ') : 'win rate';
}

function expiry(contract, meta) {
  const raw = contract?.expiry ?? meta.expiry;
  if (!raw) return '';
  const days = Math.round((new Date(raw) - Date.now()) / 86_400_000);
  if (!Number.isFinite(days)) return esc(String(raw));
  if (days < 0) return 'settled';
  if (days === 0) return 'settles today';
  return `${days}d left`;
}

/* ── trade ───────────────────────────────────────────────────────────── */

export function trade(store) {
  const { snapshot, instruments, symbol, history, depth } = store;
  const book = snapshot?.books?.[symbol];
  if (!book) {
    return `<div class="view"><div class="onboard">
      <h2>Pick a market</h2><p>Choose a contract from the list to start trading.</p>
    </div></div>`;
  }

  const meta = instruments.find((x) => x.symbol === symbol) || {};
  const session = snapshot.sessions?.[symbol] ?? 'continuous';
  const series = history[symbol] || [];
  const position = (snapshot.account?.positions ?? []).find((p) => p.symbol === symbol);

  return `<div class="market">
    <div class="market-main">
      <div data-region="head">${contractHead(symbol, book, meta, session)}</div>
      <div data-region="resolution">${resolution(book, meta)}</div>

      <div class="panel" data-region="chart">
        <h2>Price <em>${series.length} points</em></h2>
        <div class="panel-body chart-body">
          ${priceChart(series, {
            settlesAt: store.reveal ? (meta.settles_at ?? null) : null,
            label: symbol,
          })}
        </div>
      </div>

      <div class="panel" data-region="tape">
        <h2>Recent Trades</h2>
        <div class="panel-body">${tape(snapshot, symbol)}</div>
      </div>

      <div class="panel" data-region="counterparties">
        <h2>Who Filled You <em>${esc(symbol)}</em></h2>
        <div class="panel-body">${counterparties(snapshot, symbol)}</div>
      </div>

      <details class="panel drop" data-region="book">
        <summary><h2>Order Book</h2><span class="hint">depth at every price</span></summary>
        <div class="panel-body">${ladder(book, depth)}</div>
      </details>

      <details class="panel drop" data-region="spec">
        <summary><h2>Contract Specification</h2><span class="hint">the exact terms</span></summary>
        <div class="panel-body">${specification(book)}</div>
      </details>
    </div>

    <aside class="market-side">
      <div class="panel tkt" data-region="ticket">
        <h2>Trade</h2>
        <div class="panel-body">
          <div class="ticket" id="ticket">${ticket(symbol, book, session)}</div>
        </div>
      </div>

      <div class="panel" data-region="position">
        <h2>Your Position</h2>
        <div class="panel-body">${positionCard(position)}</div>
      </div>

      <div class="panel" data-region="orders">
        <h2>Working Orders <em>${(snapshot.orders || []).length}</em></h2>
        <div class="panel-body">${orders(snapshot)}</div>
      </div>
    </aside>
  </div>`;
}

function contractHead(symbol, book, meta, session) {
  const odds = impliedProbability(book.mark, book.contract?.payoff);
  return `<header class="contract" data-session="${esc(session)}">
    <div class="contract-id">
      <span class="sym mono">${esc(symbol)}</span>
      <span class="badge ${esc(session)}">${esc(session.replace('_', ' '))}</span>
      <span class="kind">${esc(book.class ?? '')}</span>
    </div>
    <h1 class="question">${question(book.contract)}</h1>
    ${marketState(book) ? `<p class="halted">${esc(marketState(book))}</p>` : ''}
    <div class="contract-figures">
      ${odds != null
        ? `<div class="stat big"><b class="mono amber">${percent(odds)}</b>
             <span>Implied Odds</span></div>`
        : ''}
      <div class="stat big"><b class="mono">${price(book.mark)}</b><span>Last</span></div>
      <div class="stat"><b class="mono">${book.spread == null ? '-' : price(book.spread)}</b><span>Spread</span></div>
      <div class="stat"><b class="mono">${count(book.trades)}</b><span>Trades</span></div>
      <div class="stat"><b class="mono">${expiry(book.contract, meta)}</b><span>Expiry</span></div>
    </div>
  </header>`;
}

/**
 * How this contract resolves, stated plainly and placed above the fold.
 *
 * Burying resolution rules beneath the price is the documented anti-pattern,
 * and it is the one that matters most: a contract whose terms nobody can read
 * is not a market, it is a slot machine with a chart attached.
 */
function resolution(book, meta) {
  const p = book.contract?.payoff ?? {};
  const settles = meta.settles_at;
  const bounds = (book.bounds ?? []).map((b) => price(b)).join(' and ');
  // A share settles at nothing and is worth the stream, so the range is what
  // it can be *worth*, and the revealed number is what it pays out in total.
  const stream = book.contract?.distribution;
  return `<section class="resolution">
    <h2>How This Resolves</h2>
    <p>${describe(book.contract)}</p>
    <ul>
      <li><span>Measured over</span>
          <b>the observation window ending ${esc(book.contract?.expiry ?? 'expiry')}</b></li>
      <li><span>${stream ? 'Can be worth' : 'Settles between'}</span>
          <b class="mono">${bounds || '-'}</b></li>
      ${stream
        ? `<li><span>Pays</span><b>${count(stream.periods)} times, the last on
             ${esc(stream.last)}</b></li>`
        : ''}
      ${p.kind === 'binary'
        ? `<li><span>Pays</span><b class="mono">${price(p.payout)} if it happens, ${price(0)} if not</b></li>`
        : ''}
    </ul>
    ${settles != null
      ? `<details class="spoiler">
           <summary>Reveal what this is worth</summary>
           <p>${stream
             ? `This share pays out <b class="mono up">${price(settles)}</b> in total
                across its life.`
             : `This contract settles at <b class="mono up">${price(settles)}</b>.`}</p>
           <p class="note">Only a simulation can tell you this, and printing it on
             the page turns a prediction market into a countdown: there is nothing
             left to discover and nothing to disagree about. Kept behind a click so
             the market can do its job, and available because being able to check
             the price against the truth is the whole point of building one.</p>
         </details>`
      : ''}
  </section>`;
}

function specification(book) {
  const rows = [
    ['Contract', book.contract?.id],
    ['Class', book.class],
    ['Tick size', book.tick],
    [book.contract?.distribution ? 'Value range' : 'Settlement range',
     (book.bounds ?? []).join(' … ')],
    ['Expiry', book.contract?.expiry],
    ['Spec digest', book.contract?.digest],
  ];
  return `<table><tbody>${rows
    .map(([label, value]) => `<tr>
      <td style="text-align:left" class="faint">${esc(label)}</td>
      <td>${value == null || value === '' ? '-' : esc(String(value))}</td>
    </tr>`)
    .join('')}</tbody></table>`;
}

/**
 * The ladder, now behind a disclosure.
 *
 * Bids and asks share a price column so the shape of the book is one vertical
 * scan; the bars show *cumulative* size, because that is what answers the
 * question anyone sizing an order is actually asking, what it costs to get
 * through a level. Each row fills the ticket at its price.
 */
function ladder(book, depth) {
  const bids = (depth?.bids ?? book.bids ?? []).map(([p, q]) => [Number(p), q]);
  const asks = (depth?.asks ?? book.asks ?? []).map(([p, q]) => [Number(p), q]);
  if (!bids.length && !asks.length) return `<div class="empty">No resting orders.</div>`;

  const byPrice = new Map();
  bids.forEach(([p, q]) => byPrice.set(p, { ...(byPrice.get(p) || {}), bid: q }));
  asks.forEach(([p, q]) => byPrice.set(p, { ...(byPrice.get(p) || {}), ask: q }));

  const cumulativeBid = new Map();
  let runningBid = 0;
  for (const [p, q] of bids) { runningBid += q; cumulativeBid.set(p, runningBid); }
  const cumulativeAsk = new Map();
  let runningAsk = 0;
  for (const [p, q] of asks) { runningAsk += q; cumulativeAsk.set(p, runningAsk); }

  const peak = Math.max(1, runningBid, runningAsk);
  const mark = Number(book.mark);
  const rows = [...byPrice.entries()].sort((a, b) => b[0] - a[0]);
  let nearest = 0;
  rows.forEach(([p], i) => {
    if (Math.abs(p - mark) < Math.abs(rows[nearest][0] - mark)) nearest = i;
  });

  return `<div class="ladder">${rows
    .map(([p, side], i) => {
      const bidW = ((cumulativeBid.get(p) || 0) / peak) * 46;
      const askW = ((cumulativeAsk.get(p) || 0) / peak) * 46;
      return `<button type="button" class="lad-row ${i === nearest ? 'at-mark' : ''}"
                   data-price="${p}"
                   aria-label="Price ${price(p)}, ${side.bid || 0} bid, ${side.ask || 0} offered">
        <span class="bar bid" style="transform:scaleX(${(bidW / 100).toFixed(4)})" aria-hidden="true"></span>
        <span class="bar ask" style="transform:scaleX(${(askW / 100).toFixed(4)})" aria-hidden="true"></span>
        <span class="bidq">${side.bid ? count(side.bid) : ''}</span>
        <span class="px">${price(p)}</span>
        <span class="askq">${side.ask ? count(side.ask) : ''}</span>
      </button>`;
    })
    .join('')}</div>`;
}

/*
 * What the send button says, and whether it does anything.
 *
 * It used to be disabled in every phase but `continuous`, under a label
 * reading "pre open, orders will rest". Both halves were wrong at once, and
 * they were wrong on the screen a stranger meets first: the exchange opens
 * with a call auction, so every contract is in `pre_open` on the first page
 * load. Orders in a call phase *are* accepted, they *do* rest, and they set
 * the price the market opens at: the venue takes them, publishes an
 * indicative price off them and clears them at the uncross. The one thing a
 * visitor could not do was place one.
 *
 * `closed` is the only phase that refuses orders (SessionState.accepts_orders),
 * and there "orders will rest" was a lie in the other direction.
 */
const PHASE = {
  pre_open: { open: true, label: 'Place Order, joins the opening call' },
  auction: { open: true, label: 'Place Order, joins the reopening call' },
  closed: { open: false, label: 'Closed, no new orders' },
};

export function sendButton(session) {
  return PHASE[session] ?? { open: true, label: 'Place Order' };
}

/**
 * The ticket, in two layers.
 *
 * Everything a first-time trader needs is visible: which side, how many, and
 * what it costs against what it pays. Limit price, time in force and post-only
 * are real and reachable, but they sit behind a disclosure, a form that opens
 * on "time in force" has already lost most of the people looking at it.
 */
function ticket(symbol, book, session) {
  const phase = sendButton(session);
  const binary = book.contract?.payoff?.kind === 'binary';
  return `<div class="sides">
      <button type="button" data-side="buy" aria-pressed="true">
        ${binary ? 'Yes' : 'Buy'}<em>${binary ? 'it happens' : 'go long'}</em>
      </button>
      <button type="button" data-side="sell" aria-pressed="false">
        ${binary ? 'No' : 'Sell'}<em>${binary ? 'it does not' : 'go short'}</em>
      </button>
    </div>

    <div class="field">
      <label for="t-qty">Contracts</label>
      <input id="t-qty" type="number" min="1" step="1" value="10"
             inputmode="numeric" autocomplete="off" spellcheck="false">
    </div>
    <div class="quick" role="group" aria-label="Quick size">
      ${[5, 25, 100, 250].map((n) => `<button type="button" data-qty="${n}">${n}</button>`).join('')}
    </div>

    <div class="preview" id="t-preview" aria-live="polite"></div>

    <button type="button" class="send" id="t-send" ${phase.open ? '' : 'disabled'}>
      ${phase.label}
    </button>

    <details class="advanced">
      <summary>Advanced</summary>
      <div class="field">
        <label for="t-px">Limit price (blank trades at market)</label>
        <input id="t-px" type="text" inputmode="decimal" placeholder="4660.25&hellip;"
               autocomplete="off" spellcheck="false">
      </div>
      <div class="row2">
        <div class="field">
          <label for="t-stop">Stop trigger <span class="opt">optional</span></label>
          <input id="t-stop" type="text" inputmode="decimal" placeholder="4600.00&hellip;"
                 autocomplete="off" spellcheck="false">
        </div>
        <div class="field">
          <label for="t-show">Show at a time (blank shows all)</label>
          <input id="t-show" type="number" min="0" step="1" placeholder="10"
                 inputmode="numeric" autocomplete="off" spellcheck="false">
        </div>
      </div>
      <p class="note">A <b>stop</b> waits off the book until the market trades at
        or through your trigger, then goes to market (or to your limit)
        price, if you gave one. Nobody can see it while it waits. An
        <b>iceberg</b> shows part of your size at a time and refreshes the rest
        behind whoever queued up meanwhile, which is what hiding costs.</p>
      <div class="field">
        <label for="t-tif">Time in force</label>
        <select id="t-tif">
          <option value="gtc">Good till cancelled</option>
          <option value="ioc">Immediate or cancel</option>
          <option value="fok">Fill or kill</option>
          <option value="post_only">Post only</option>
        </select>
      </div>
      <div class="row2">
        <button type="button" class="minor" data-act="cancel_all">Cancel All</button>
        <button type="button" class="minor danger" data-act="flatten">Flatten</button>
      </div>
      <div class="row2">
        <button type="button" class="minor" data-act="halt" data-symbol="${esc(symbol)}"
                aria-label="Halt trading in ${esc(symbol)}">Halt</button>
        <button type="button" class="minor" data-act="uncross" data-symbol="${esc(symbol)}"
                aria-label="Run the reopening auction for ${esc(symbol)}">Uncross</button>
      </div>
    </details>

    <p class="note">Your order joins the same queue as every algorithm here and
      travels the same latency link. Nothing about it is privileged.</p>`;
}

/**
 * The other side of your own fills, by name.
 *
 * "Is anything actually on the other side of this?" is a fair question to ask
 * of a simulated exchange, and a roster of agents does not answer it. This
 * does: every fill here names the participant that took it.
 */
function counterparties(snapshot, symbol) {
  const rows = (snapshot.counterparties ?? []).filter((c) => c.symbol === symbol);
  if (!rows.length) {
    return `<div class="empty">Once you trade, the participants who took the
      other side appear here by name.</div>`;
  }

  // Aggregated per counterparty and side. One order that swept a book produced
  // twenty near-identical rows, "buy 30 at 0.06 from mm-1" over and over,
  // which is a log, not an answer. What a person wants to know is who they are
  // trading against and at what average, and that is four numbers.
  const totals = new Map();
  for (const fill of rows) {
    const key = `${fill.counterparty}|${fill.side}`;
    const acc = totals.get(key) ?? { ...fill, quantity: 0, notional: 0, fills: 0 };
    acc.quantity += fill.quantity;
    acc.notional += fill.quantity * Number(fill.price);
    acc.fills += 1;
    totals.set(key, acc);
  }

  const summary = [...totals.values()].sort((a, b) => b.quantity - a.quantity);
  return `<table>
    <thead><tr><th>Counterparty</th><th>Side</th><th>Contracts</th>
               <th>Avg price</th><th>Fills</th></tr></thead>
    <tbody>${summary
      .map((c) => `<tr>
        <td class="mono" style="text-align:left">${esc(c.counterparty)}</td>
        <td class="${c.side === 'buy' ? 'up' : 'down'}" style="text-align:left">${esc(c.side)}</td>
        <td>${count(c.quantity)}</td>
        <td>${price(c.notional / c.quantity)}</td>
        <td class="faint">${count(c.fills)}</td>
      </tr>`)
      .join('')}</tbody></table>`;
}

/*
 * A closed position is not the same thing as never having traded.
 *
 * Anything with a quantity of zero was answered with "No position in this
 * contract", which threw away the only number that survives closing a trade:
 * what it made. Someone who bought, sold and came out ahead was told, on the
 * screen they traded from, that there was nothing here, and had to go to
 * another screen to find out whether they had won or lost. The server already
 * keeps the row alive for exactly this reason (it filters on quantity *and*
 * volume), so the information was arriving and being discarded on the way to
 * the page.
 */
function positionCard(position) {
  if (!position) {
    return `<div class="empty">No position in this contract.</div>`;
  }
  const flat = position.quantity === 0;
  if (flat && !Number(position.realized)) {
    return `<div class="empty">No position in this contract.</div>`;
  }
  return `<div class="pos">
    ${flat ? `<p class="note">Closed. What is left is what it made.</p>` : ''}
    <div><span>Contracts</span><b class="mono ${cls(position.quantity)}">${signed(position.quantity, 0)}</b></div>
    ${flat ? '' : `<div><span>Average price</span><b class="mono">${price(position.average_price)}</b></div>`}
    ${flat ? '' : `<div><span>Unrealised</span><b class="mono ${cls(Number(position.unrealized))}">${money(position.unrealized)}</b></div>`}
    <div><span>Realised</span><b class="mono ${cls(Number(position.realized))}">${money(position.realized)}</b></div>
  </div>`;
}

function orders(snapshot) {
  const rows = snapshot.orders || [];
  if (!rows.length) return `<div class="empty">Nothing working.</div>`;
  return `<table><tbody>${rows
    .map((o) => `<tr>
      <td style="text-align:left">${esc(o.symbol)}</td>
      <td class="faint">#${o.order_id}</td>
      <td><button type="button" class="minor" data-act="cancel" data-order="${o.order_id}"
              data-symbol="${esc(o.symbol)}"
            aria-label="Cancel order ${o.order_id} in ${esc(o.symbol)}">Cancel</button></td>
    </tr>`)
    .join('')}</tbody></table>`;
}

function tape(snapshot, symbol) {
  const rows = (snapshot.tape || []).filter((t) => t.symbol === symbol).slice(0, 30);
  if (!rows.length) return `<div class="empty">No trades yet.</div>`;
  return `<table>
    <thead><tr><th>Time</th><th>Price</th><th>Size</th><th>Taker</th></tr></thead>
    <tbody>${rows
      .map((t) => `<tr>
        <td class="faint">${clock(t.t)}</td>
        <td>${price(t.price)}</td>
        <td>${count(t.quantity)}</td>
        <td class="${t.side === 'buy' ? 'up' : 'down'}">${esc(t.side)}</td>
      </tr>`)
      .join('')}</tbody></table>`;
}

/* ── portfolio ───────────────────────────────────────────────────────── */

export function portfolio(store) {
  const s = store.snapshot;
  if (!s) return `<div class="view"><div class="onboard"><h2>Connecting&hellip;</h2></div></div>`;
  const a = s.account;
  const positions = a.positions || [];

  const figure = (label, value, klass = '') =>
    `<div class="panel figure"><span>${label}</span><b class="mono ${klass}">${value}</b></div>`;

  return `<div class="view stack">
    <div class="figures" data-region="figures">
      ${figure('Account Value', money(a.equity))}
      ${figure('Profit &amp; Loss', money(a.pnl), cls(Number(a.pnl)))}
      ${figure('Available to Trade', money(a.free_cash))}
      ${figure('Held as Collateral', money(a.collateral))}
    </div>

    <div class="panel" data-region="positions">
      <h2>Positions <em>${positions.length}</em></h2>
      <div class="panel-body">${
        positions.length
          ? `<table>
              <thead><tr><th>Contract</th><th>Qty</th><th>Avg</th><th>Unrealised</th><th>Realised</th></tr></thead>
              <tbody>${positions
                .map((p) => `<tr data-symbol="${esc(p.symbol)}">
                  <td>${esc(p.symbol)}</td>
                  <td class="${cls(p.quantity)}">${signed(p.quantity, 0)}</td>
                  <td>${price(p.average_price)}</td>
                  <td class="${cls(Number(p.unrealized))}">${money(p.unrealized)}</td>
                  <td class="${cls(Number(p.realized))}">${money(p.realized)}</td>
                </tr>`)
                .join('')}</tbody></table>`
          : `<div class="empty">You are flat. Pick a market to place your first trade.</div>`
      }</div>
    </div>

    <div class="panel grow" data-region="activity">
      <h2>Activity <em>${(s.log || []).length}</em></h2>
      <div class="panel-body">${blotter(s.log)}</div>
    </div>
  </div>`;
}

/**
 * The blotter: one row per private event the venue sent back.
 *
 * These arrive as structured events, not sentences. An earlier version
 * interpolated the whole object into a cell, which rendered `[object Object]`
 * whenever anything happened.
 */
function blotter(log) {
  if (!log || !log.length) return `<div class="empty">Nothing yet.</div>`;
  const tone = { fill: 'up', reject: 'down', cancel: 'dim', ack: 'faint' };
  return `<table>
    <thead><tr><th>Time</th><th>Event</th><th>Contract</th><th>Side</th>
               <th>Qty</th><th>Price</th><th>Detail</th></tr></thead>
    <tbody>${log
      .map((e) => {
        const detail = e.reason
          ? esc(e.reason)
          : e.remaining != null && e.type !== 'ack'
            ? `${count(e.remaining)} left`
            : '';
        return `<tr>
          <td class="faint">${clock(e.t)}</td>
          <td class="${tone[e.type] ?? 'dim'}" style="text-align:left">${esc(e.type ?? '?')}</td>
          <td style="text-align:left">${esc(e.symbol ?? '')}</td>
          <td class="${e.side === 'buy' ? 'up' : e.side === 'sell' ? 'down' : 'faint'}">${esc(e.side ?? '')}</td>
          <td>${e.quantity == null ? '' : count(e.quantity)}</td>
          <td>${e.price == null ? '' : price(e.price)}</td>
          <td class="faint" style="text-align:left">${detail}</td>
        </tr>`;
      })
      .join('')}</tbody></table>`;
}

/* ── research (operator) ─────────────────────────────────────────────── */

export function research(store) {
  const { diagnostics, agents, symbol } = store;

  const rows = (diagnostics?.verdicts ?? [])
    .map((v) => `<tr>
      <td style="text-align:left">${esc(v.name)}</td>
      <td>${v.value == null ? '-' : Number(v.value).toFixed(3)}</td>
      <td class="faint" style="text-align:left">${esc(v.expected)}</td>
      <td class="${v.verdict === 'as expected' ? 'verdict-ok' : 'verdict-no'}">${esc(v.verdict)}</td>
    </tr>`)
    .join('');

  const roster = (agents || [])
    .map((a) => `<tr>
      <td style="text-align:left">${esc(a.id)}</td>
      <td style="text-align:left">${esc(a.role ?? a.kind)}
        <span class="faint mono" style="margin-left:6px">${esc(a.kind)}</span></td>
      <td>${count(a.fills)}</td>
      <td class="${a.rejects ? 'down' : 'faint'}">${count(a.rejects)}</td>
      <td>${a.equity == null ? '-' : money(a.equity)}</td>
      <td><button type="button" class="minor ${a.halted ? '' : 'danger'}"
              data-act="${a.halted ? 'revive' : 'kill'}" data-agent="${esc(a.id)}">
            ${a.halted ? 'Let back in' : 'Stop'}</button></td>
    </tr>`)
    .join('');

  return `<div class="view">
    <div class="panel" style="margin-bottom:12px">
      <h2>Stylized Facts <em>${esc(symbol ?? '')}</em></h2>
      <div class="panel-body">${
        diagnostics?.pending
          ? `<div class="empty">Collecting observations&hellip; (${diagnostics.observations ?? 0} so far)</div>`
          : rows
            ? `<table>
                <thead><tr><th>Statistic</th><th>Value</th><th>Expected</th><th>Verdict</th></tr></thead>
                <tbody>${rows}</tbody></table>`
            : `<div class="empty">No diagnostics yet.</div>`
      }</div>
    </div>

    <p class="note" style="margin-bottom:12px">The same estimators the research
      harness uses, run on the live price series , not a second
      implementation that could disagree with it. A verdict of
      &ldquo;unexpected&rdquo; is information, not a failure.</p>

    <div class="panel">
      <h2>Participants <em>${(agents || []).length}</em></h2>
      <div class="panel-body">${
        roster
          ? `<table>
              <thead><tr><th>Agent</th><th>Kind</th><th>Fills</th><th>Rejects</th><th>Equity</th></tr></thead>
              <tbody>${roster}</tbody></table>`
          : `<div class="empty">Loading&hellip;</div>`
      }</div>
    </div>
  </div>`;
}

/* ── lab (operator) ──────────────────────────────────────────────────── */

export function lab(store) {
  const s = store.session;
  if (!s) return `<div class="view"><div class="empty">Loading&hellip;</div></div>`;
  const c = s.config;

  const schedules = Object.entries(s.fee_schedules || {})
    .map(([name, f]) => `<option value="${esc(name)}" ${name === c.fees ? 'selected' : ''}>
        ${esc(name)}, taker ${f.taker_bps}bp / maker ${f.maker_bps}bp
      </option>`)
    .join('');

  const halts = (s.halts || []).slice().reverse();

  return `<div class="view">
    <p class="note" style="margin-bottom:12px">Operator controls. Changing any of
      these starts a <em>new</em> session rather than editing the running one
      . A population edited mid-flight would produce a market no seed could
      reproduce, and reproducibility is most of what makes a result here worth
      anything.</p>

    <div class="two">
      <div class="panel">
        <h2>Configuration <em>generation ${s.generation}</em></h2>
        <div class="panel-body">
          <div class="controls">
            <div class="row2">
              <div class="field">
                <label for="c-seed">Seed</label>
                <input id="c-seed" type="number" value="${c.seed}"
                       inputmode="numeric" autocomplete="off" spellcheck="false">
              </div>
              <div class="field">
                <label for="c-flow">Flow traders</label>
                <input id="c-flow" type="number" min="0" max="24" value="${c.flow_traders}"
                       inputmode="numeric" autocomplete="off" spellcheck="false">
              </div>
              <div class="field">
                <label for="c-makers">Market makers</label>
                <input id="c-makers" type="number" min="1" max="8" value="${c.makers ?? 3}"
                       inputmode="numeric" autocomplete="off" spellcheck="false">
              </div>
            </div>
            <div class="field">
              <label for="c-fees">Fee schedule</label>
              <select id="c-fees">${schedules}</select>
            </div>
            <div class="field">
              <label for="c-mechanism">Mechanism</label>
              <select id="c-mechanism">
                <option value="book" ${c.mechanism === 'scoring-rule' ? '' : 'selected'}>
                  Order book : every contract</option>
                <option value="scoring-rule" ${c.mechanism === 'scoring-rule' ? 'selected' : ''}>
                  Scoring rule (event contracts only)</option>
              </select>
              <p class="note">A logarithmic scoring rule is defined on a partition of
                outcomes, so it can quote a coin flip and not a future. On it the venue
                is the market maker and subsidises the market instead of profiting from
                it. Experiment 2 compared the two over 200 paired trials and found the
                mechanism explains none of the difference in what a market learns.</p>
            </div>
            <div class="field">
              <label for="c-band">Price band <span class="opt">optional</span></label>
              <input id="c-band" type="text" inputmode="decimal"
                     value="${c.price_band ?? ''}" placeholder="0.10&hellip;"
                     autocomplete="off" spellcheck="false">
            </div>
            <label class="check">
              <input id="c-arb" type="checkbox" ${c.arbitrageur ? 'checked' : ''}>
              Cross-instrument arbitrageur
            </label>
            <label class="check">
              <input id="c-auction" type="checkbox" ${c.opening_auction === false ? '' : 'checked'}>
              Open with a call auction
            </label>
            <label class="check">
              <input id="c-surface" type="checkbox" ${c.surface === false ? '' : 'checked'}>
              Price options off one distribution
            </label>
            <button type="button" class="send" id="c-apply">Rebuild Market</button>
          </div>
        </div>
      </div>

      <div>
        <div class="panel" style="margin-bottom:12px">
          <h2>Venue</h2>
          <div class="panel-body">
            <table><tbody>
              <tr><td style="text-align:left">Taker fee</td><td>${s.fees.taker_bps} bp</td></tr>
              <tr><td style="text-align:left">Maker fee</td><td>${s.fees.maker_bps} bp</td></tr>
              <tr><td style="text-align:left">Fees collected</td><td>${money(s.fees_collected)}</td></tr>
              <tr><td style="text-align:left">Price band</td><td>${s.price_band ?? '-'}</td></tr>
              <tr><td style="text-align:left">Uptime</td><td>${(s.uptime || 0).toFixed(0)}s</td></tr>
            </tbody></table>
          </div>
        </div>

        <div class="panel">
          <h2>Sessions</h2>
          <div class="panel-body">
            <table><tbody>${Object.entries(s.sessions || {})
              .map(([sym, state]) => `<tr>
                <td style="text-align:left">${esc(sym)}</td>
                <td><span class="badge ${esc(state)}">${esc(state.replace('_', ' '))}</span></td>
                <td>
                  <button type="button" class="minor" data-act="halt" data-symbol="${esc(sym)}"
                          aria-label="Halt trading in ${esc(sym)}">Halt</button>
                  <button type="button" class="minor" data-act="uncross" data-symbol="${esc(sym)}"
                          aria-label="Run the reopening auction for ${esc(sym)}">Uncross</button>
                </td>
              </tr>`)
              .join('')}</tbody></table>
          </div>
        </div>
      </div>
    </div>

    <div class="panel" style="margin-top:12px">
      <h2>Halts <em>${halts.length}</em></h2>
      <div class="panel-body">${
        halts.length
          ? `<table>
              <thead><tr><th>Contract</th><th>Reason</th><th>Reference</th><th>Price</th></tr></thead>
              <tbody>${halts
                .map((h) => `<tr>
                  <td style="text-align:left">${esc(h.symbol)}</td>
                  <td style="text-align:left" class="${h.reason === 'price_band' ? 'down' : 'dim'}">${esc(h.reason)}</td>
                  <td>${h.reference ?? '-'}</td>
                  <td>${h.price ?? '-'}</td>
                </tr>`)
                .join('')}</tbody></table>`
          : `<div class="empty">No halts this session.</div>`
      }</div>
    </div>
  </div>`;
}
