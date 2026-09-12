/* The standings: every account on this exchange, ranked by what it made.
 *
 * The venue ships with a population of agents, and it also lets outside
 * systems in through the public API. The question this screen exists to answer
 * is whether those outside systems beat the agents the market ships with, so
 * residents and seats are ranked in one list against each other, and a seat is
 * marked to be findable at a glance down a column of two dozen agents.
 *
 * The venue's own accounts are listed apart and are not ranked at all. A fee
 * treasury does have a return and the arithmetic behind it is sound, but it
 * measures fees taken against fees taken rather than a trader against a
 * market. In the same list it would win, which is the wrong answer to the only
 * question the screen is asking.
 */

import { clock, cls, count, esc, money, signed } from './format.js';

/** The mark `format.js` prints where a number is absent. One glyph, one meaning. */
const ABSENT = '-';

/* `clock` reads the nanoseconds the tape is stamped in. `as_of` counts in
 * simulated seconds, the same clock the lab screen reports as uptime. */
const NS_PER_SECOND = 1e9;

const KIND_WORDS = { resident: 'Resident', seat: 'Seat', operator: 'Venue' };

/**
 * The number a field holds, or null where it holds none.
 *
 * `Number(null)` and `Number('')` are both 0, and 0 is finite, so every absent
 * field on this payload would otherwise sort, colour and print as a flat zero.
 * An account that is flat and an account whose equity the venue could not
 * compute are different facts, and a leaderboard is exactly the table that
 * would quietly confuse them.
 */
function num(value) {
  if (value == null || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/**
 * A return, as a percentage.
 *
 * `return_pct` arrives already in percent units, so `percent()` is the wrong
 * helper here: it multiplies by a hundred and would report 4.555 as 455.5%.
 * `signed` gives a gain the plus a leaderboard needs and `money` does not
 * supply, and refuses anything it cannot make a number of.
 */
function ret(value) {
  const text = signed(value, 2);
  return text === ABSENT ? ABSENT : `${text}%`;
}

/** Money, carrying the sign a gain does not otherwise get. */
function gain(value) {
  const text = money(value);
  if (text === ABSENT) return ABSENT;
  return (num(value) ?? 0) > 0 ? `+${text}` : text;
}

function whole(value) {
  const n = num(value);
  return n == null ? ABSENT : count(n);
}

/**
 * Best first, and unknown last.
 *
 * The venue ranks, and its rank is the authority wherever it gave one. What
 * this settles is the rest: a row the venue left unranked, or whose return it
 * does not know, cannot be placed against one it does, so it goes to the end
 * rather than to the top, where a missing number read as a zero would put an
 * unmeasured account above everything that lost money.
 */
function byStanding(a, b) {
  const ra = num(a.rank);
  const rb = num(b.rank);
  if (ra != null && rb != null) return ra - rb;
  if (ra != null || rb != null) return ra != null ? -1 : 1;
  const pa = num(a.return_pct);
  const pb = num(b.return_pct);
  if (pa != null && pb != null) return pb - pa;
  if (pa != null || pb != null) return pa != null ? -1 : 1;
  return 0;
}

/**
 * The kind, as the one thing on the row a reader is scanning for.
 *
 * Amber is the chrome colour on this terminal and is what carries emphasis;
 * green and red are spoken for by direction and may not be borrowed to mean a
 * category. Only the seat is badged, because a badge on all three kinds is a
 * uniform, and a uniform distinguishes nobody.
 */
function tag(kind) {
  const word = KIND_WORDS[kind] ?? (kind == null ? ABSENT : String(kind));
  if (kind === 'seat') return `<span class="badge amber">${esc(word)}</span>`;
  return `<span class="faint">${esc(word)}</span>`;
}

/* A halted account is a fact about this session; a quiet one is not. Only
 * `true` earns the badge, so a venue that says nothing makes no claim. */
function who(entry) {
  const label = String(entry.label ?? entry.id ?? '');
  const id = String(entry.id ?? '');
  return `<td style="text-align:left">
    <b>${esc(label)}</b>
    ${id && id !== label ? `<span class="faint"> ${esc(id)}</span>` : ''}
    ${entry.halted === true ? ' <span class="badge closed">Halted</span>' : ''}
  </td>`;
}

function traderRow(entry) {
  return `<tr data-kind="${esc(entry.kind ?? 'unknown')}">
    <td style="text-align:right" class="faint">${whole(entry.rank)}</td>
    ${who(entry)}
    <td style="text-align:left">${tag(entry.kind)}</td>
    <td class="${cls(num(entry.return_pct))}">${ret(entry.return_pct)}</td>
    <td class="${cls(num(entry.pnl))}">${gain(entry.pnl)}</td>
    <td>${money(entry.equity)}</td>
    <td class="faint">${whole(entry.positions)}</td>
  </tr>`;
}

/* No rank and no return, which is the whole point of the separation: both
 * compute for a fee account, and neither says anything true about it. */
function venueRow(entry) {
  return `<tr data-kind="${esc(entry.kind ?? 'operator')}">
    ${who(entry)}
    <td>${money(entry.equity)}</td>
    <td class="${cls(num(entry.pnl))}">${gain(entry.pnl)}</td>
    <td class="faint">${whole(entry.positions)}</td>
  </tr>`;
}

function figure(label, value, klass = '') {
  return `<div class="panel figure">
    <span>${esc(label)}</span><b class="mono ${klass}">${value}</b>
  </div>`;
}

/** Where the best outside system sits in the field, the screen in one number. */
function bestSeat(traders) {
  const seat = traders.find((entry) => entry.kind === 'seat');
  if (!seat) return ABSENT;
  const at = num(seat.rank);
  return at == null ? ABSENT : `#${count(at)}`;
}

function balance(conserved) {
  if (conserved === true) return ['Yes', ''];
  if (conserved === false) return ['No', 'amber'];
  return [ABSENT, 'faint'];
}

/* Conservation is the caveat on every number under it, so it is stated above
 * the table rather than beneath it. */
function caveat(conserved) {
  if (conserved === false) {
    return `The venue's books do not balance. Cash and positions across every
      account should sum to what the session started with and right now they do
      not, so read the table as an instrument known to be miscalibrated rather
      than as a result.`;
  }
  if (conserved === true) {
    return `Cash and positions across every account sum to what the session
      started with, so nothing here was created or destroyed on the way.
      Returns are measured on starting cash, which means an account seated a
      minute ago is not comparable to one that has run all session.`;
  }
  return `The venue did not say whether its books balance this session. A
    leaderboard read off a market that is not conserving is not evidence of
    anything, so treat this one as unverified.`;
}

export function standings(store) {
  const table = store?.standings;
  const rows = Array.isArray(table?.standings)
    ? table.standings.filter((entry) => entry && typeof entry === 'object')
    : [];

  // Nothing yet and nothing at all are different states, and the payload
  // arrives on the slow timer, so the first paint of this screen is usually
  // the first of them.
  if (!rows.length) {
    return `<div class="view"><div class="onboard">
      <h2>${table ? 'No accounts on the venue' : 'Reading the standings&hellip;'}</h2>
      <p>${table
        ? 'The exchange published no accounts. Nobody has been seated and no resident agent holds one.'
        : 'Every trader here ranked by what they made: the agents the venue ships with, and the outside systems that connect to it through the public API.'}</p>
    </div></div>`;
  }

  // An unfamiliar kind is ranked with the traders rather than dropped. A
  // participant the screen cannot classify is still a participant, and leaving
  // it out would make the field look smaller than it is.
  //
  // The rank on a row is the venue's, taken over the whole population, so
  // lifting the venue's own accounts out of the list can leave a gap in the
  // sequence. Closing it would mean re-ranking, which is a claim this screen
  // is not the one entitled to make.
  const venue = rows.filter((entry) => entry.kind === 'operator');
  const traders = rows.filter((entry) => entry.kind !== 'operator').sort(byStanding);
  const seats = num(table.counts?.seat)
    ?? traders.filter((entry) => entry.kind === 'seat').length;
  const [balanced, balanceClass] = balance(table.conserved);
  const at = num(table.as_of);
  // Amber is emphasis, and there is nothing to emphasise about a figure the
  // venue could not give: an absent value is greyed like every other one.
  const best = bestSeat(traders);

  return `<div class="view">
    <div class="figures" data-region="standings:figures">
      ${figure('Best Outside Seat', best, best === ABSENT ? 'faint' : 'amber')}
      ${figure('Seats Trading', count(seats))}
      ${figure('As Of', at == null ? ABSENT : clock(at * NS_PER_SECOND))}
      ${figure('Books Balance', balanced, balanceClass)}
    </div>

    <p class="note" style="margin-bottom:16px" data-region="standings:caveat">
      ${caveat(table.conserved)}</p>

    <section class="class-block" data-region="standings:traders">
      <div class="class-head">
        <h2>Standings</h2>
        <span>Outside systems and resident agents, ranked together on return over starting cash.</span>
        <b class="mono">${count(traders.length)}</b>
      </div>
      <div class="panel"><div class="panel-body">
        <table>
          <thead><tr>
            <th style="text-align:right">Rank</th>
            <th style="text-align:left">Trader</th>
            <th style="text-align:left">Kind</th>
            <th>Return</th><th>P&amp;L</th><th>Equity</th><th>Positions</th>
          </tr></thead>
          <tbody>${traders.map(traderRow).join('')}</tbody>
        </table>
      </div></div>
    </section>

    ${venue.length ? `<section class="class-block" data-region="standings:venue">
      <div class="class-head">
        <h2>Venue Accounts</h2>
        <span>The session operator, the match operator and the fee treasury. Not traders, so not ranked.</span>
        <b class="mono">${count(venue.length)}</b>
      </div>
      <div class="panel"><div class="panel-body">
        <table>
          <thead><tr>
            <th>Account</th><th>Equity</th><th>P&amp;L</th><th>Positions</th>
          </tr></thead>
          <tbody>${venue.map(venueRow).join('')}</tbody>
        </table>
      </div></div>
    </section>` : ''}
  </div>`;
}
