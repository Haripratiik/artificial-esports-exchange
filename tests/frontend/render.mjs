/* Render every view against a real market snapshot, without a browser.
 *
 * The views are pure functions of the store, which is what makes this possible:
 * node can import them directly and check the HTML they produce. What this
 * catches is the whole class of template bugs that look fine until someone
 * opens the page, a field renamed on the server showing up as `undefined`, a
 * number arriving as a string and rendering `NaN`, an object interpolated into
 * text as `[object Object]`.
 *
 * It does not replace looking at the page. It does mean a server-side rename
 * fails a test rather than silently blanking a panel.
 *
 *     node tests/frontend/render.mjs <fixture.json>
 */

import { readFileSync } from 'node:fs';
import * as views from '../../dashboard/static/js/views.js';
import * as fmt from '../../dashboard/static/js/format.js';

const fixture = JSON.parse(readFileSync(process.argv[2] ?? '/tmp/fixture.json', 'utf8'));

const failures = [];
const check = (name, condition, detail = '') => {
  if (!condition) failures.push(`${name}${detail ? `: ${detail}` : ''}`);
};

// Build the store exactly as main.js does after a tick.
const history = {};
for (const [symbol, book] of Object.entries(fixture.snapshot.books)) {
  const base = Number(book.mark);
  // A plausible path, so the chart code actually has something to draw.
  history[symbol] = Array.from({ length: 80 }, (_, i) => base * (1 + Math.sin(i / 9) * 0.004));
}

/* The blotter only appears once the venue has sent something back, so a quiet
 * market leaves that code path unrendered. It is forced here instead: this test
 * first failed one run in three, purely because whether the bug showed up
 * depended on whether a fill happened to land during the fixture window. */
fixture.snapshot.log = [
  { t: 1_500_000_000, symbol: 'EMBER_OBJECTIVE_WR', type: 'ack',
    sequence: 1, agent_id: 'you', order_id: 1, side: 'buy', quantity: 10, price: null },
  { t: 1_600_000_000, symbol: 'EMBER_OBJECTIVE_WR', type: 'fill',
    sequence: 2, agent_id: 'you', order_id: 1, side: 'buy', quantity: 4,
    price: '4660.25', aggressor: true, remaining: 6 },
  { t: 1_700_000_000, symbol: 'VANTA_SOLO_GT140', type: 'reject',
    sequence: 3, agent_id: 'you', reason: 'insufficient_collateral', order_id: 2 },
  { t: 1_800_000_000, symbol: 'EMBER_OBJECTIVE_WR', type: 'cancel',
    sequence: 4, agent_id: 'you', order_id: 1, remaining: 6 },
  ...(fixture.snapshot.log ?? []),
];

const store = {
  view: 'markets',
  symbol: 'EMBER_OBJECTIVE_WR',
  snapshot: fixture.snapshot,
  instruments: fixture.instruments,
  session: fixture.session,
  agents: fixture.agents,
  diagnostics: fixture.diagnostics,
  depth: fixture.depth,
  history,
  side: 'buy',
  generation: fixture.snapshot.generation ?? 0,
};

const LEAKS = ['undefined', 'NaN', '[object Object]', 'null%', '$NaN'];

for (const name of ['markets', 'trade', 'portfolio', 'research', 'lab']) {
  let html;
  try {
    html = views[name](store);
  } catch (error) {
    failures.push(`${name} threw: ${error.message}`);
    continue;
  }
  check(`${name} produced HTML`, typeof html === 'string' && html.length > 200,
        `${html?.length ?? 0} chars`);
  for (const leak of LEAKS) {
    check(`${name} free of "${leak}"`, !html.includes(leak));
  }
  // Tags must balance well enough that the panel does not swallow the page.
  const open = (html.match(/<div/g) || []).length;
  const close = (html.match(/<\/div>/g) || []).length;
  check(`${name} div tags balance`, open === close, `${open} open, ${close} close`);
}

/* A view rendered with nothing yet loaded must degrade, not explode: this is
 * the state every view is in for the first second after the page opens. */
const cold = { ...store, snapshot: null, session: null, diagnostics: null, depth: null,
               agents: [], instruments: [], history: {} };
for (const name of ['markets', 'trade', 'portfolio', 'research', 'lab']) {
  try {
    const html = views[name](cold);
    check(`${name} survives an empty store`, typeof html === 'string' && html.length > 0);
  } catch (error) {
    failures.push(`${name} threw on empty store: ${error.message}`);
  }
}

/* Content that must actually reach the page. */
const trade = views.trade(store);
check('trade shows the symbol', trade.includes('EMBER_OBJECTIVE_WR'));
check('trade renders a ladder', trade.includes('lad-row'), 'no ladder rows');
check('trade renders a chart', trade.includes('<svg'), 'no chart svg');
check('trade offers post-only', trade.includes('post_only'));

/* Nothing may be unreachable.
 *
 * The list opens collapsed -- one row per subject with its instrument count --
 * so most instruments are deliberately not in the default markup. That is the
 * whole point of the collapse, and it is also exactly how a market could go
 * missing without anyone noticing. So the reachability check is done with
 * every group open, which is the state a search or a class filter produces.
 */
const markets = views.markets(store);
const everySubject = new Set(
  Object.values(fixture.snapshot.books).map((b) => views.subjectOf(b)),
);
for (const subject of everySubject) {
  check(`the list offers ${subject}`, markets.includes(subject));
}
check('the list collapses', everySubject.size < Object.keys(fixture.snapshot.books).length,
      `${everySubject.size} subjects for ${Object.keys(fixture.snapshot.books).length} instruments`);

const expandedAll = views.markets({ ...store, expanded: everySubject });
for (const symbol of Object.keys(fixture.snapshot.books)) {
  check(`expanding reaches ${symbol}`, expandedAll.includes(symbol));
}

const blotter = views.portfolio(store);
check('blotter renders event types', blotter.includes('reject') && blotter.includes('fill'));
check('blotter shows the symbol', blotter.includes('VANTA_SOLO_GT140'));
check('blotter shows a reject reason', blotter.includes('insufficient_collateral'));
check('blotter formats the fill price', blotter.includes('4,660.25'),
      'price not converted from ticks');

const lab = views.lab(store);
check('lab shows the seed', lab.includes('c-seed'));
check('lab lists fee schedules', lab.includes('maker-taker'));

/* Formatting edge cases, every one of these has a wrong answer that renders. */
check('price(null) is a dash', fmt.price(null) === '-', fmt.price(null));
check('price("") is a dash', fmt.price('') === '-', fmt.price(''));
check('price(0) is still zero', fmt.price(0) === '0.00', fmt.price(0));
check('money(null) is a dash', fmt.money(null) === '-', fmt.money(null));
check('money(0) is still zero', fmt.money(0) === '0.00', fmt.money(0));
check('signed(null) is a dash', fmt.signed(null) === '-', fmt.signed(null));
check('price("4660.25") formats', fmt.price('4660.25') === '4,660.25', fmt.price('4660.25'));
check('money handles millions', fmt.money(2_500_000) === '2.50M', fmt.money(2_500_000));
check('money marks negatives', fmt.money(-42).startsWith('−'), fmt.money(-42));
check('signed marks positives', fmt.signed(1.5) === '+1.50', fmt.signed(1.5));
check('clock formats minutes', fmt.clock(90e9) === '1m 30s', fmt.clock(90e9));
check('esc neutralises markup',
      fmt.esc('<img src=x onerror=1>') === '&lt;img src=x onerror=1&gt;');
check('sparkline of one point is empty', fmt.sparkline([1]) === '');
check('chart of one point degrades', fmt.priceChart([1]).includes('Waiting'));

/* ── the ticket's arithmetic ──────────────────────────────────────────────
 *
 * These two decide what a person is *told* an order will cost before they
 * commit to it, so being subtly wrong here is worse than showing nothing.
 */

const twoLevels = [[100, 5], [101, 10]];

const partial = fmt.walkBook(twoLevels, 8);
check('walk fills across levels', partial.filled === 8, `${partial?.filled}`);
check('walk costs 5x100 + 3x101', partial.cost === 803, `${partial?.cost}`);
check('walk averages correctly', Math.abs(partial.average - 100.375) < 1e-9,
      `${partial?.average}`);
check('walk knows it completed', partial.complete === true);

const beyond = fmt.walkBook(twoLevels, 20);
check('walk reports a shortfall', beyond.complete === false && beyond.shortfall === 5,
      `filled ${beyond?.filled}, short ${beyond?.shortfall}`);
check('walk never invents depth', beyond.filled === 15, `${beyond?.filled}`);

check('walk of nothing is null', fmt.walkBook(twoLevels, 0) === null);
check('walk of an empty book is null', fmt.walkBook([], 10) === null);
check('walk of a bad size is null', fmt.walkBook(twoLevels, NaN) === null);

const binary = { kind: 'binary', payout: 1.0, threshold: 0.48, comparison: '>' };
check('a binary price is its probability',
      fmt.impliedProbability(0.42, binary) === 0.42,
      `${fmt.impliedProbability(0.42, binary)}`);
check('probability scales with the payout',
      fmt.impliedProbability(50, { kind: 'binary', payout: 200 }) === 0.25);
check('probability clamps above one',
      fmt.impliedProbability(1.4, binary) === 1);
check('a future has no implied probability',
      fmt.impliedProbability(4660, { kind: 'linear', scale: 10000 }) === null);
check('a missing payoff has no probability', fmt.impliedProbability(1, null) === null);
check('percent formats', fmt.percent(0.425) === '42.5%', fmt.percent(0.425));
check('percent of nothing is a dash', fmt.percent(null) === '-');

/* ── a market must not print its own answer ───────────────────────────────
 *
 * The trade page stated "will actually settle at 0.00" beside the price, and
 * drew it on the chart as a target line. That is not a prediction market; it is
 * a countdown. Nothing was left to discover and nothing to disagree about, so
 * the price sat pinned for whole sessions.
 */
const revealed = { ...store, reveal: false };
const quiet = views.trade(revealed);
check('the settlement value is not on the page', !quiet.includes('Will actually settle at'));
check('there is a way to reveal it', quiet.includes('Reveal what this is worth'));
check('the reveal starts closed', !/<details class="spoiler"[^>]*open/.test(quiet));
// The target line is the only thing that draws a dashed stroke, so that is
// what to look for -- the word "settles" also appears in the resolution copy
// and in the question itself, which made a looser check match the wrong text.
check('the chart draws no target line by default',
      !/stroke-dasharray/.test(quiet), 'the answer is still drawn on the chart');
check('revealing puts it back',
      /stroke-dasharray/.test(views.trade({ ...store, reveal: true })));

/* ── the contract has to name itself ──────────────────────────────────────
 *
 * The metric reference arrives under `ref`. Reading the wrong key made every
 * question on the exchange read "Will the metric finish above 0.48?" -- and it
 * failed silently, because the fallback was a plausible English phrase rather
 * than an error. Nothing crashed, nothing logged, and every card was
 * unidentifiable.
 */
const marketsHtml = views.markets(store);
check('questions name their subject', !marketsHtml.includes('Will the metric'),
      'the fallback is showing instead of the competitor');

/* Every group the view forms has to be named on the page, whatever it groups by.
 *
 * This used to read `contract.underlying.ref.subject` off each book and demand
 * that string appear. It held while every book was a statistical contract,
 * because `subjectOf` returns exactly that for one and the section header
 * prints it, so the check was passing on a coincidence rather than on the
 * claim. Match contracts group by the match instead, and a competitor's own
 * name is on a row rather than on a header, so it shows only once that group is
 * expanded: measured on the live listing, 426 of the books in the fixture are
 * match contracts and every one of them failed the old reading while the page
 * was rendering perfectly well.
 *
 * What the view promises is that a group is named, so that is what is asserted,
 * and it still covers the subject reading everywhere the two coincide. */
for (const [symbol, book] of Object.entries(fixture.snapshot.books)) {
  const group = views.subjectOf(book);
  if (!group || group === 'Other') continue;
  check(`${symbol} is grouped under ${group}`, marketsHtml.includes(group),
        'the group this book belongs to is not named on the page');
}

// Asset class still has to be visible and countable. It is a filter now rather
// than a heading: grouping by it scattered one competitor's future, calls,
// puts and weeklies across four sections, so finding one name meant visiting
// four
// places. As a chip it narrows the list instead of fragmenting it.
check('asset classes are offered as filters', marketsHtml.includes('class="chips"'));
check('prediction markets are named', marketsHtml.includes('Prediction Markets'));
check('options are named', marketsHtml.includes('Options'));
check('the venue vocabulary is translated', views.groupOf('event') === 'Prediction Markets');
check('an unknown class still groups', views.groupOf('zzz') === 'Other');
check('shares are named', marketsHtml.includes('Shares'));
check('commodities are named', marketsHtml.includes('Commodities'));
check('every chip carries its count', /class="chip[^"]*"[^>]*>[^<]*<b class="mono">\d+/.test(marketsHtml));

/* A share and a commodity each say something a future does not, and each has a
 * way of silently reading as a future instead. The share's terminal payoff is
 * zero, so anything that describes it from the payoff alone truthfully reports
 * that it settles at nothing -- which is the one sentence guaranteed to make a
 * reader skip it. The commodity's week *is* the contract, so a description
 * that omits the week makes four distinct instruments read identically. */
for (const [symbol, book] of Object.entries(fixture.snapshot.books)) {
  const terms = fmt.describe(book.contract);
  if (book.class === 'equity') {
    check(`${symbol} is described as a share`, /share of/i.test(terms), terms);
    check(`${symbol} does not read as worthless`, !/Settles at 0 /.test(terms), terms);
    check(`${symbol} says how many payments`, /weeks/.test(terms), terms);
  }
  if (book.class === 'commodity') {
    check(`${symbol} names its delivery week`,
          /delivered/.test(views.question(book.contract) ?? ''),
          'a delivery contract that does not say when is four copies of itself');
  }
}

// Region keys are what stop the screen being rebuilt under the reader.
const tradeRegions = (views.trade(store).match(/data-region="/g) || []).length;
check('the trade screen is divided into regions', tradeRegions >= 8, `${tradeRegions}`);
// Regions are per SUBJECT now, not per instrument: the markets screen groups
// by the thing being traded and puts the instrument count on the row, the way
// every venue that carries a lot of markets does. Kalshi's browse page shows
// 60 entries across 574 tradeable markets; this screen used to be 1:1.
check('subjects are their own regions', /data-region="subject:/.test(marketsHtml));

/* ── search and counterparties ────────────────────────────────────────── */

// Mirrors the real payload: the metric reference lives under `ref`. The first
// version of this fixture used `metric`, which is the same mistake the view
// itself was making -- so the fixture agreed with the bug and the test passed
// while every card on the exchange read "the metric".
const futureBook = {
  class: 'future',
  contract: { payoff: { kind: 'linear', scale: 10000 },
              underlying: { kind: 'single', ref: { subject: 'EMBER', metric: 'win_rate' } } },
};
check('an empty query matches everything', views.matches('EMBER_OBJECTIVE_WR', futureBook, ''));
check('a null query matches everything', views.matches('EMBER_OBJECTIVE_WR', futureBook, null));
check('search finds a ticker', views.matches('EMBER_OBJECTIVE_WR', futureBook, 'ember'));
check('search finds an asset class', views.matches('EMBER_OBJECTIVE_WR', futureBook, 'future'));
check('search finds the subject', views.matches('XYZ', futureBook, 'ember'),
      'searching only the ticker means you must already know the ticker');
check('search rejects a miss', !views.matches('EMBER_OBJECTIVE_WR', futureBook, 'zzzz'));
check('search requires every word',
      !views.matches('EMBER_OBJECTIVE_WR', futureBook, 'ember zzzz'));
check('search is case insensitive', views.matches('EMBER_OBJECTIVE_WR', futureBook, 'EmBeR'));

// A query that matches nothing must explain itself rather than showing a void.
const noMatch = views.markets({ ...store, query: 'zzzzz' });
check('an empty result explains itself', noMatch.includes('Nothing matches'));
check('an empty result echoes the query safely', noMatch.includes('zzzzz'));

// Naming the other side of a fill is what answers "is anything really there?".
const filled = views.trade({
  ...store,
  snapshot: { ...store.snapshot, counterparties: [
    { symbol: 'EMBER_OBJECTIVE_WR', side: 'buy', quantity: 20, price: '4670.25', counterparty: 'mm-1' },
  ] },
});
check('counterparties are named', filled.includes('mm-1'));
check('the counterparty panel is present', filled.includes('Who Filled You'));

const unfilled = views.trade({ ...store, snapshot: { ...store.snapshot, counterparties: [] } });
check('an untraded market says so', unfilled.includes('other side'));

/* ── information order ────────────────────────────────────────────────────
 *
 * Prediction-market design guidance is consistent that resolution rules belong
 * above the pricing detail and that the order book belongs collapsed. This
 * interface had both backwards, so the ordering is pinned rather than trusted
 * to survive the next edit.
 */

const tradeHtml = views.trade(store);

const atResolution = tradeHtml.indexOf('How This Resolves');
const atBook = tradeHtml.indexOf('Order Book');
const atChart = tradeHtml.indexOf('<svg');
check('resolution rules are present', atResolution > -1);
check('resolution comes before the order book', atResolution < atBook,
      `resolution at ${atResolution}, book at ${atBook}`);
check('resolution comes before the chart', atResolution < atChart,
      `resolution at ${atResolution}, chart at ${atChart}`);

// <details> without an `open` attribute is closed. Leading with a depth ladder
// is an operator's view of the world.
const bookBlock = tradeHtml.slice(tradeHtml.indexOf('<details'), atBook + 40);
check('the order book is a disclosure', bookBlock.includes('<details'));
check('the order book is closed by default', !/<details[^>]*open/.test(tradeHtml),
      'a disclosure is open on first render');

// The simple layer is visible; the technical layer is not.
const beforeAdvanced = tradeHtml.slice(0, tradeHtml.indexOf('<details class="advanced"'));
check('quantity is on the simple layer', beforeAdvanced.includes('t-qty'));
check('the cost preview is on the simple layer', beforeAdvanced.includes('t-preview'));
check('time in force is behind Advanced', !beforeAdvanced.includes('t-tif'),
      'the ticket opens on time-in-force');
check('limit price is behind Advanced', !beforeAdvanced.includes('t-px'));

// The browse list is for browsing, not for specifications.
const cardHtml = views.markets(store);
// A row showing only `VANTA_SOLO_GT100` is the mystery-meat identifier NN/g warns
// about: readable only by someone who already knows the ticker. The plain
// question rides alongside it.
check('rows ask the question', /class="question"/.test(cardHtml));
check('rows carry no spec digest', !cardHtml.includes('digest'));
check('rows carry no tick size', !/tick size/i.test(cardHtml));
// Expiry left the browse row deliberately -- it is a specification, and the
// contract header on the trade screen carries it where it is actually needed.
check('the trade screen still shows time remaining', /left|settles/.test(views.trade(store)));
/* The collapse itself, asserted: far fewer groups than instruments.
 *
 * Against the book count rather than against a constant. The constant was 12,
 * which was a little over the subject count of a 47 contract listing, and it
 * stopped being a statement about collapsing the moment matches began listing
 * at runtime: the group count then depends on how many matches have opened by
 * the time the fixture is taken, so the same check read 7 subjects on a fixture
 * captured early and 15 on one captured after the module had advanced the
 * clock, and only the second tripped it. Neither number says anything about
 * whether the list collapses. The ratio does, and it is what `views.markets`
 * argues for in its own comment: 355 markets presenting as 9 rows. */
const subjectCount = (cardHtml.match(/data-region="subject:/g) || []).length;
const rowCount = (cardHtml.match(/class="mrow"/g) || []).length;
const bookCount = Object.keys(fixture.snapshot.books).length;
check('the list collapses instruments into subjects',
      subjectCount > 0 && subjectCount * 4 <= bookCount,
      `${subjectCount} subjects for ${bookCount} books, ${rowCount} rows shown`);

/* ── accessibility, as assertions ─────────────────────────────────────────
 *
 * These encode findings from an audit against the Web Interface Guidelines.
 * Writing them down as tests is the difference between fixing an issue once
 * and keeping it fixed: the clickable-div problem in particular is the kind of
 * thing that creeps back in the next time a row needs to be clickable.
 */

const everyView = ['markets', 'trade', 'portfolio', 'research', 'lab']
  .map((n) => views[n](store))
  .join('\n');

// Rows and cards are actions. A div with a click handler cannot be reached by
// keyboard and is not announced as interactive.
check('no clickable divs remain',
      !/<div[^>]*\bdata-(symbol|price)=/.test(everyView),
      'a div is carrying a click target');
check('market rows are buttons', /<button[^>]*class="mrow"/.test(everyView));
check('subject headers are buttons', /<button[^>]*class="subject-head"/.test(everyView));
check('ladder rows are buttons', /<button[^>]*class="lad-row/.test(everyView));

// Every button must be announceable: visible text or an explicit label.
const buttons = everyView.match(/<button[\s\S]*?<\/button>/g) || [];
check('there are buttons to check', buttons.length > 5, `${buttons.length}`);
for (const button of buttons) {
  const labelled = /aria-label=/.test(button);
  const text = button.replace(/<[^>]*>/g, '').replace(/&[a-z]+;/g, '').trim();
  check('every button is announceable', labelled || text.length > 0,
        button.slice(0, 70));
}

// Every form control needs a label. Three forms are all valid: a `for=`
// pointing at it, an `aria-label`, or a <label> wrapped around it -- the last
// of which is what a checkbox wants anyway, since it makes the text part of
// the hit target instead of a dead zone next to it.
const wrapped = (everyView.match(/<label[\s\S]*?<\/label>/g) || []).join(' ');
const inputs = everyView.match(/<(input|select)[^>]*>/g) || [];
check('there are inputs to check', inputs.length > 3, `${inputs.length}`);
for (const input of inputs) {
  const id = /id="([^"]+)"/.exec(input)?.[1];
  const byFor = id ? everyView.includes(`for="${id}"`) : false;
  const byWrap = wrapped.includes(input);
  check('every input has a label', byFor || byWrap || /aria-label=/.test(input),
        input.slice(0, 70));
}

// Decorative charts must not be announced as content.
check('sparklines are hidden from assistive tech',
      !/<span class="spark">/.test(everyView),
      'a sparkline is missing aria-hidden');

/* ── css hygiene ──────────────────────────────────────────────────────── */

const css = readFileSync(
  new URL('../../dashboard/static/css/terminal.css', import.meta.url), 'utf8');

check('no transition: all', !/transition:\s*all\b/.test(css),
      'animates properties nobody asked for');
check('focus is visible', css.includes(':focus-visible'),
      'keyboard users cannot see where they are');
check('reduced motion is honoured', css.includes('prefers-reduced-motion'));
check('taps are not delayed', css.includes('touch-action'));

const html = readFileSync(
  new URL('../../dashboard/static/index.html', import.meta.url), 'utf8');
check('dark colour-scheme declared', /color-scheme:\s*dark/.test(html));
check('theme-color matches the background', /name="theme-color"/.test(html));
check('font host is preconnected', html.includes('rel="preconnect"'));
check('fonts are not @imported from css', !css.includes('@import'),
      'an @import serialises the stylesheet and the font request');
check('there is a skip link', html.includes('class="skip"'));
check('there is exactly one h1', (html.match(/<h1/g) || []).length === 1);
check('async regions are announced', html.includes('aria-live'));
check('zoom is not disabled',
      !/user-scalable=no|maximum-scale=1/.test(html));

if (failures.length) {
  console.error(`FAILED (${failures.length})`);
  failures.forEach((f) => console.error('  - ' + f));
  process.exit(1);
}
console.log('all frontend render checks passed');
