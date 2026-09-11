/* The players screen: who is on this exchange, and what each of them is for.
 *
 * The research screen already answers "is anything actually on the other side
 * of this?" with fills, rejects and equity. It cannot answer "why is that
 * thing here at all", and for a venue whose counterparties are all software,
 * that is the question a reader asks first. This screen carries the answer per
 * player: what it is, whether it is live right now, what it does, and what the
 * market gets from having it.
 *
 * Two fields keep it from being a brochure. `measured` is a number taken from
 * the codebase rather than an adjective about it, and `source` is the file
 * that produces the behaviour, so every claim on the screen is one a reader
 * can go and check. They are given the weight to match.
 */

import { count, esc } from './format.js';

/* Residents first, because they are the market. The outside systems come last:
 * they only matter once there is a market for them to connect to. Unknown
 * kinds keep the order the venue sent them in, since the sort is stable. */
const KIND_ORDER = ['resident', 'operator', 'connected'];

/* The venue's own counts, in the words a person would use. A key nobody
 * anticipated still renders, under its own name, rather than vanishing. */
const COUNT_LABELS = {
  resident: 'Resident agents',
  operator: 'Operator seats',
  seats: 'Operator seats',
  connected: 'Connected systems',
};

/**
 * Absent, as distinct from empty.
 *
 * Every field but the id and the name is nullable, and a null here must leave
 * no trace: a card with an empty ruled paragraph on it reads as a field the
 * venue failed to fill rather than one it never had.
 */
function said(value) {
  if (value == null) return null;
  const words = String(value).trim();
  return words || null;
}

function paragraph(value, klass) {
  const words = said(value);
  return words ? `<p class="${klass}">${esc(words)}</p>` : '';
}

/**
 * The state light, and the reason behind it.
 *
 * Off is not broken. The arbitrageur ships switched off, and a screen that
 * shows that as a fault is describing a configuration as a failure. Green is
 * reserved for a player the payload actually asserts is live, so anything it
 * stays quiet about gets the neutral ribbon instead of a claim.
 */
function state(player, kind) {
  if (player.live === true) return { badge: '<span class="badge">Live</span>', note: '' };
  if (player.live !== false) return { badge: '', note: '' };
  const why = kind === 'connected'
    ? 'No system is attached to this seat right now.'
    : 'Off in this session, by configuration rather than by fault.';
  return {
    badge: '<span class="badge closed">Off</span>',
    note: `<p class="hint">${why}</p>`,
  };
}

/* One market maker and three behave differently at the same spread, so how
 * many share a role belongs on the card rather than in a footnote. */
function facts(player) {
  const items = [];
  const seats = Number(player.count);
  if (Number.isFinite(seats) && seats > 1) items.push(`${seats} seated`);

  const held = Number(player.positions);
  if (player.positions != null && Number.isFinite(held)) {
    items.push(`${count(held)} position${held === 1 ? '' : 's'}`);
  }

  // Already formatted by the venue, which is the side that knows the currency
  // and the rounding. Running it back through a number formatter here would
  // turn "412,003.55" into NaN.
  const equity = said(player.equity);
  if (equity) items.push(`equity ${equity}`);

  if (!items.length) return '';
  // One wrapping line rather than a flex row: at a card's width the row shrank
  // every item to its narrowest and stacked each one over two lines, so three
  // short facts read as six ragged ones.
  return `<p class="hint mono">${items.map((f) => esc(f)).join(' &middot; ')}</p>`;
}

/* Given the treatment a contract's terms get, for the same reason: it is the
 * one line on the card that could be wrong in a way somebody could catch. */
function measured(value) {
  const words = said(value);
  if (!words) return '';
  return `<div class="resolution">
    <h2>Measured</h2>
    <div class="question">${esc(words)}</div>
  </div>`;
}

/* A path, not a link. The file is in the repository the reader already has
 * open, and an href to a host nobody configured would rot on the first move. */
function source(value) {
  const path = said(value);
  if (!path) return '';
  return `<p class="hint">Source <code class="mono dim">${esc(path)}</code></p>`;
}

function card(player, kind) {
  const { badge, note } = state(player, kind);
  const body = [
    note,
    paragraph(player.role, 'question'),
    paragraph(player.strategy, 'hint'),
    paragraph(player.why, 'note'),
    measured(player.measured),
    source(player.source),
  ].filter(Boolean);

  const ribbon = player.live === true ? '' : ' data-session="closed"';
  return `<article class="card"${ribbon}>
    <div class="card-top">
      <span class="sym">${esc(player.id ?? '')}</span>
      ${badge}
    </div>
    <h3 class="question">${esc(player.name ?? player.id ?? '')}</h3>
    ${facts(player)}
    ${body.length ? `<div class="field">${body.join('')}</div>` : ''}
  </article>`;
}

function group(entry, index) {
  const kind = said(entry?.kind) ?? `group-${index}`;
  const blurb = said(entry?.blurb);
  const roster = (Array.isArray(entry?.players) ? entry.players : [])
    .filter((p) => p && typeof p === 'object');

  return `<section class="class-block" data-region="players:${esc(kind)}">
    <div class="class-head">
      <h2>${esc(said(entry?.title) ?? 'Players')}</h2>
      <span>${blurb ? esc(blurb) : ''}</span>
      <b class="mono">${roster.length}</b>
    </div>
    ${roster.length
      ? `<div class="grid">${roster.map((p) => card(p, kind)).join('')}</div>`
      : `<div class="empty">Nobody is seated here.</div>`}
  </section>`;
}

function figures(counts) {
  const cells = Object.entries(counts ?? {})
    .filter(([, value]) => value != null)
    .map(([key, value]) => {
      const label = COUNT_LABELS[key] ?? String(key).replace(/_/g, ' ');
      const shown = Number.isFinite(Number(value)) ? count(value) : String(value);
      return `<div class="panel figure">
        <span>${esc(label)}</span><b class="mono">${esc(shown)}</b>
      </div>`;
    });
  if (!cells.length) return '';
  return `<div class="figures" data-region="players:counts">${cells.join('')}</div>`;
}

function rank(kind) {
  const at = KIND_ORDER.indexOf(kind);
  return at < 0 ? KIND_ORDER.length : at;
}

export function players(store) {
  const roster = store?.players;
  const groups = (Array.isArray(roster?.groups) ? roster.groups : [])
    .filter((g) => g && typeof g === 'object');

  // The roster arrives over REST on the lazy timer, so the first paint of this
  // screen usually has nothing to draw. Waiting and empty are different states
  // and are worded differently.
  if (!groups.length) {
    return `<div class="view"><div class="onboard">
      <h2>${roster ? 'Nobody is on the venue' : 'Reading the roster&hellip;'}</h2>
      <p>${roster
        ? 'The exchange published no participants. A book nothing is quoting has no prices to publish either.'
        : 'Everyone on this exchange: the agents built into the venue, the seats the operator trades from, and the outside systems that connect to it.'}</p>
    </div></div>`;
  }

  const ordered = [...groups].sort((a, b) => rank(a.kind) - rank(b.kind));
  return `<div class="view">
    ${figures(roster.counts)}
    ${ordered.map(group).join('')}
  </div>`;
}
