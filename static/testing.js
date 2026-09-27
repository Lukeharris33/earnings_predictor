// Testing rig: model accuracy charts + batch predictions vs actual prices.
// Chart helpers come from charts.js.

const LIVE_REFRESH_MS = 60_000;

// ============================================================ model section
const MODELS = JSON.parse(el("rig-models").textContent || "[]");

function modelMetrics(m) {
  const out = {};
  for (const h of HORIZONS) {
    const hm = (m.metrics || {})[h];
    if (hm && hm.directional_accuracy !== undefined) out[h] = hm;
  }
  return out;
}


function renderModel() {
  const model = MODELS.find((m) => m.id === el("rig-model").value);
  const metrics = model ? modelMetrics(model) : {};
  const hs = HORIZONS.filter((h) => metrics[h]);
  const empty = el("model-empty");
  const body = el("model-body");
  if (!hs.length) {
    empty.textContent = model ? "This model has no out-of-sample metrics to chart." : "No models yet. Train one on the dashboard.";
    empty.classList.remove("hidden");
    body.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  body.classList.remove("hidden");
  const v2 = hs.some((h) => metrics[h].baseline_majority_accuracy !== undefined);

  // KPI row: one tile per horizon.
  const kpis = el("model-kpis");
  kpis.replaceChildren(...hs.map((h) => {
    const m = metrics[h];
    if (!v2) return kpiTile(`${h} accuracy`, pctPts(m.directional_accuracy), "no baseline recorded (v1 model)");
    const edge = (m.directional_accuracy - m.baseline_majority_accuracy) * 100;
    return kpiTile(
      `${h} accuracy`,
      pctPts(m.directional_accuracy),
      `${signed(edge)} pts vs baseline · IC ${signed(m.ic, 3)} · ${m.n.toLocaleString()} tests`,
      edge > 0 && m.ic > 0 ? "is-good" : "is-bad",
    );
  }));

  if (!v2) {
    ["chart-edge", "chart-ic-years", "chart-families", "chart-spread"].forEach((id) => {
      el(id).querySelector(".chart-body").textContent = "Only models trained with walk-forward evaluation (v2) have this breakdown.";
    });
    return;
  }

  const edges = hs.map((h) => (metrics[h].directional_accuracy - metrics[h].baseline_majority_accuracy) * 100);
  mountChart(el("chart-edge"), (holder, w) => drawColumns(holder, w, {
    categories: hs,
    series: [{ name: "edge vs baseline", color: (v) => (v >= 0 ? C.pos : C.neg), values: edges }],
    fmt: (v) => `${signed(v)} pts`,
  }), {
    headers: ["Horizon", "Accuracy", "Baseline", "Edge (pts)"],
    rows: hs.map((h, i) => [h, pctPts(metrics[h].directional_accuracy), pctPts(metrics[h].baseline_majority_accuracy), signed(edges[i])]),
  });

  const years = [...new Set(hs.flatMap((h) => (metrics[h].folds || []).map((f) => f.test_period)))].sort();
  const icSeries = hs.map((h, i) => ({
    name: h, color: C.series[i],
    values: years.map((yr) => { const f = (metrics[h].folds || []).find((f) => f.test_period === yr); return f && f.ic !== null ? f.ic : NaN; }),
  }));
  mountChart(el("chart-ic-years"), (holder, w) => drawCategoryLines(holder, w, {
    categories: years, series: icSeries, fmt: (v) => (Number.isFinite(v) ? signed(v, 3) : "—"),
  }), {
    headers: ["Test year", ...hs],
    rows: years.map((yr, yi) => [yr, ...icSeries.map((s) => (Number.isFinite(s.values[yi]) ? signed(s.values[yi], 3) : "—"))]),
  }, icSeries.map((s) => ({ label: s.name, color: s.color, kind: "line" })));

  const families = ["nn", "gbm", "blend"].filter((f) => hs.some((h) => (metrics[h].by_family || {})[f]));
  const familyName = { nn: "Network ensemble", gbm: "Boosted trees", blend: "Blend" };
  const famSeries = families.map((f, i) => ({
    name: familyName[f], color: C.series[i],
    values: hs.map((h) => { const x = (metrics[h].by_family || {})[f]; return x && x.ic !== null ? x.ic : NaN; }),
  }));
  mountChart(el("chart-families"), (holder, w) => drawColumns(holder, w, {
    categories: hs, series: famSeries, fmt: (v) => (Number.isFinite(v) ? signed(v, 3) : "—"),
  }), {
    headers: ["Horizon", ...famSeries.map((s) => s.name)],
    rows: hs.map((h, i) => [h, ...famSeries.map((s) => (Number.isFinite(s.values[i]) ? signed(s.values[i], 3) : "—"))]),
  }, famSeries.map((s) => ({ label: s.name, color: s.color, kind: "rect" })));

  const spreads = hs.map((h) => metrics[h].top_minus_bottom_quintile);
  mountChart(el("chart-spread"), (holder, w) => drawColumns(holder, w, {
    categories: hs,
    series: [{ name: "top minus bottom fifth", color: (v) => (v >= 0 ? C.pos : C.neg), values: spreads.map((v) => (v === null ? NaN : v)) }],
    fmt: (v) => `${signed(v, 2)}%`,
  }), {
    headers: ["Horizon", "Top − bottom fifth"],
    rows: hs.map((h, i) => [h, spreads[i] === null ? "—" : `${signed(spreads[i], 2)}%`]),
  });
}

function renderAllModels() {
  const rows = MODELS.map((m) => {
    const metrics = modelMetrics(m);
    return [m.name, (m.created_at || "").slice(0, 10), ...HORIZONS.map((h) => {
      const x = metrics[h];
      if (!x) return "—";
      const base = x.baseline_majority_accuracy !== undefined ? ` / ${pctPts(x.baseline_majority_accuracy)}` : "";
      const ic = x.ic !== undefined && x.ic !== null ? ` · IC ${signed(x.ic, 3)}` : "";
      return `${pctPts(x.directional_accuracy)}${base}${ic}`;
    })];
  });
  el("all-models-table").replaceChildren(buildTable(["Model", "Trained", ...HORIZONS.map((h) => `${h} acc / base`)], rows));
}

// ============================================================ batch section
let BATCH = null;       // payload from /api/testing/batch
let SELECTED = null;    // symbol shown in the detail chart
let DETAIL_RANGE = "3m";
let liveTimer = null;

// Close on or before `date` for a {dates, closes} series.
function closeAt(series, date) {
  const d = series.dates;
  let lo = 0, hi = d.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (d[mid] <= date) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans < 0 ? NaN : series.closes[ans];
}
function lastDate(series) { return series.dates[series.dates.length - 1]; }
function excessBetween(sym, start, end) {
  const s = BATCH.prices[sym];
  if (!s || !start) return NaN;
  const stock = closeAt(s, end) / closeAt(s, start) - 1;
  const bench = closeAt(BATCH.benchmark, end) / closeAt(BATCH.benchmark, start) - 1;
  return (stock - bench) * 100;
}

// Window, and what happened in it, for one prediction at one horizon.
function outcome(p, h) {
  const d = (p.decisions || {})[h];
  if (!d) return null;
  const start = h === "1d" ? p.baseline_date : p.day1_date;
  const prices = BATCH.prices[p.symbol];
  if (!start || !prices) return { d, start, closed: false, actual: NaN };
  const closed = Boolean(d.closed);
  let actual;
  if (closed) {
    actual = d.actual_pct_change !== undefined && d.actual_pct_change !== null
      ? d.actual_pct_change : excessBetween(p.symbol, start, d.window_end);
  } else {
    actual = excessBetween(p.symbol, start, lastDate(prices));
  }
  return { d, start, closed, actual };
}

function renderBatchKpis(h) {
  const rows = BATCH.predictions.map((p) => ({ p, o: outcome(p, h) })).filter((r) => r.o);
  const closed = rows.filter((r) => r.o.closed && Number.isFinite(r.o.actual));
  const open = rows.filter((r) => !r.o.closed);
  const hits = closed.filter((r) => Math.sign(r.o.d.predicted_pct_change) === Math.sign(r.o.actual));
  const upShare = closed.length ? closed.filter((r) => r.o.actual > 0).length / closed.length : NaN;
  const base = Math.max(upShare, 1 - upShare);
  const margin = closed.length ? 1.96 * Math.sqrt(0.25 / closed.length) * 100 : NaN;
  const avg = (arr) => (arr.length ? arr.reduce((a, r) => a + r.o.actual, 0) / arr.length : NaN);
  const buys = closed.filter((r) => r.o.d.action === "BUY");
  const sells = closed.filter((r) => r.o.d.action === "SELL");
  const openRunning = open.filter((r) => Number.isFinite(r.o.actual));
  const openHits = openRunning.filter((r) => Math.sign(r.o.d.predicted_pct_change) === Math.sign(r.o.actual));

  const tiles = [
    kpiTile(`${h} windows closed`, `${closed.length} of ${rows.length}`, open.length ? `${open.length} still open` : "all windows final"),
    closed.length
      ? kpiTile("Direction right", pctPts(hits.length / closed.length, 0),
        `${hits.length}/${closed.length} · baseline ${pctPts(base, 0)} · ±${margin.toFixed(0)} pts margin`,
        hits.length / closed.length > base ? "is-good" : "is-bad")
      : kpiTile("Direction right", "—", "no closed windows yet"),
    kpiTile("BUY vs SELL, actual", buys.length || sells.length ? `${pctOrDash(avg(buys))} / ${pctOrDash(avg(sells))}` : "—",
      buys.length || sells.length ? `avg move after ${buys.length} BUY and ${sells.length} SELL calls` : "no closed BUY or SELL calls"),
    openRunning.length
      ? kpiTile("Open windows so far", `${openHits.length}/${openRunning.length} on track`, "direction right at the latest price; not final")
      : kpiTile("Open windows so far", "—", "nothing still open at this horizon"),
  ];
  el("batch-kpis").replaceChildren(...tiles);
}

function renderScatter(h) {
  const points = BATCH.predictions.map((p) => {
    const o = outcome(p, h);
    if (!o || !Number.isFinite(o.actual)) return null;
    return { symbol: p.symbol, x: o.d.predicted_pct_change, y: o.actual, action: o.d.action, closed: o.closed, windowEnd: o.d.window_end };
  }).filter(Boolean);
  const nClosed = points.filter((p) => p.closed).length;
  el("scatter-sub").textContent = `${h}: each mark is one ticker. Filled = window closed, hollow = still open (latest price). On the diagonal = a perfect forecast; top-right and bottom-left = right direction. ${nClosed} closed, ${points.length - nClosed} open.`;
  mountChart(el("chart-scatter"), (holder, w) => drawScatter(holder, w, { points }), {
    headers: ["Ticker", "Call", "Predicted", "Actual", "Window", "Ends"],
    rows: points.map((p) => [p.symbol, p.action, `${signed(p.x, 2)}%`, `${signed(p.y, 2)}%`, p.closed ? "closed" : "open", p.windowEnd]),
  }, [
    { label: "BUY", color: C.buy, kind: "up" },
    { label: "SELL", color: C.sell, kind: "down" },
    { label: "HOLD", color: C.hold, kind: "dot" },
    { label: "window still open", color: C.muted, kind: "hollow" },
  ]);
}

// Move vs benchmark since the pre-release close, and each call's target on
// that same scale (for later windows, the first-day move plus the call).
function sinceRelease(p) {
  const s = BATCH.prices[p.symbol];
  if (!s) return null;
  const i0 = s.dates.findIndex((d) => d >= p.baseline_date);
  if (i0 < 0) return null;
  const dates = s.dates.slice(i0);
  const values = dates.map((d) => excessBetween(p.symbol, p.baseline_date, d));
  const markers = [];
  for (const h of HORIZONS) {
    const o = outcome(p, h);
    if (!o || !o.d.window_end) continue;
    const offset = h === "1d" || !p.day1_date ? 0 : excessBetween(p.symbol, p.baseline_date, p.day1_date);
    const end = o.closed ? dates.filter((d) => d <= o.d.window_end).pop() : null;
    markers.push({ h, date: end || o.d.window_end, value: offset + o.d.predicted_pct_change, action: o.d.action, closed: o.closed, o });
  }
  return { dates, values, markers };
}

function verdict(o) {
  if (!o || !Number.isFinite(o.actual)) return { cls: "", text: "—" };
  const right = Math.sign(o.d.predicted_pct_change) === Math.sign(o.actual);
  if (!o.closed) return { cls: "muted", text: right ? "on track" : "off track" };
  return right ? { cls: "is-good", text: "✓ right" } : { cls: "is-bad", text: "✗ wrong" };
}

function renderMultiples() {
  const box = el("multiples");
  box.replaceChildren();
  for (const p of BATCH.predictions) {
    const data = sinceRelease(p);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `multiple${p.symbol === SELECTED ? " is-selected" : ""}`;
    card.addEventListener("click", () => { SELECTED = p.symbol; renderMultiples(); renderDetail(); el("chart-detail").scrollIntoView({ behavior: "smooth", block: "nearest" }); });

    const head = document.createElement("div");
    head.className = "multiple-head";
    const sym = document.createElement("strong"); sym.textContent = p.symbol;
    const now = document.createElement("span");
    now.className = "mono";
    now.textContent = data && data.values.length ? `${signed(data.values[data.values.length - 1])}% since release` : "no prices";
    head.append(sym, now);
    card.appendChild(head);

    const holder = document.createElement("div");
    holder.className = "multiple-spark";
    card.appendChild(holder);

    const calls = document.createElement("div");
    calls.className = "multiple-calls";
    for (const h of HORIZONS) {
      const o = outcome(p, h);
      if (!o) continue;
      const v = verdict(o);
      const row = document.createElement("span");
      row.className = `multiple-call${o.closed ? "" : " is-open"}`;
      const shape = svg("svg", { width: 10, height: 10, "aria-hidden": "true" });
      marker(shape, ACTION_SHAPE[o.d.action], 5, 5, ACTION_COLOR[o.d.action], { r: 3, hollow: !o.closed });
      const label = document.createElement("span");
      label.textContent = `${h} ${o.d.action} ${signed(o.d.predicted_pct_change)}%`;
      const res = document.createElement("span");
      res.className = v.cls;
      res.textContent = v.text;
      row.append(shape, label, res);
      calls.appendChild(row);
    }
    card.appendChild(calls);
    box.appendChild(card);
    if (data) drawSpark(holder, holder.clientWidth || 200, { dates: data.dates, values: data.values, markers: data.markers.filter((m) => m.closed) });
  }
}

function renderDetail() {
  const fig = el("chart-detail");
  const p = BATCH.predictions.find((x) => x.symbol === SELECTED) || BATCH.predictions[0];
  if (!p) return;
  SELECTED = p.symbol;
  const s = BATCH.prices[p.symbol];
  el("detail-title").textContent = `${p.symbol}${p.company ? ` · ${p.company}` : ""}`;
  if (!s) {
    el("detail-sub").textContent = "No price history for this ticker.";
    fig.querySelector(".chart-body").replaceChildren();
    return;
  }
  const i0 = Math.max(0, s.dates.findIndex((d) => d >= p.baseline_date) - 5);
  const dates = s.dates.slice(i0);
  const prices = s.closes.slice(i0);
  const base = closeAt(s, p.baseline_date);
  const benchBase = closeAt(BATCH.benchmark, p.baseline_date);
  // What the stock would be worth had it exactly matched the benchmark since the release.
  const matched = dates.map((d) => base * closeAt(BATCH.benchmark, d) / benchBase);
  const latest = lastDate(s);

  const markers = [];
  const vlines = [{ date: p.baseline_date, label: "release" }];
  for (const h of HORIZONS) {
    const o = outcome(p, h);
    if (!o || !o.start || !o.d.window_end) continue;
    const startPrice = closeAt(s, o.start);
    const benchEnd = closeAt(BATCH.benchmark, o.closed ? o.d.window_end : latest);
    const benchMove = benchEnd / closeAt(BATCH.benchmark, o.start) - 1;
    const target = startPrice * (1 + benchMove + o.d.predicted_pct_change / 100);
    const actualPrice = o.closed ? closeAt(s, o.d.window_end) : NaN;
    vlines.push({ date: o.d.window_end, label: h });
    markers.push({
      date: o.d.window_end, value: target, action: o.d.action, closed: o.closed,
      label: `${h} call: ${o.d.action} ${signed(o.d.predicted_pct_change, 2)}% vs ${BATCH.benchmark.symbol}`,
      tipRows: [
        { label: "target price", value: money(target) },
        o.closed
          ? { label: `actual close ${shortDate(o.d.window_end)}`, value: money(actualPrice) }
          : { label: "window closes", value: shortDate(o.d.window_end) },
        { label: o.closed ? "actual vs benchmark" : "so far vs benchmark", value: `${signed(o.actual, 2)}%` },
      ],
    });
  }
  const spans = { "1m": 31, "3m": 92, "6m": 183, "1y": 380 };
  const xEnd = addDays(p.baseline_date, spans[DETAIL_RANGE]);
  const xDomain = [dates[0], xEnd > latest ? xEnd : latest];
  el("detail-sub").textContent = `Close price since the release on ${shortDate(p.baseline_date)}, next to where it would be had it simply matched ${BATCH.benchmark.symbol}. Markers are each call's target price at the end of its window (hollow = window still open; target moves with the benchmark until it closes). Latest: ${money(prices[prices.length - 1])} on ${shortDate(latest)}.`;

  mountChart(fig, (holder, w) => drawTimeChart(holder, w, {
    lines: [
      { name: p.symbol, color: C.series[0], dates, values: prices, endLabel: true },
      { name: `matched ${BATCH.benchmark.symbol}`, color: C.hold, dates, values: matched, width: 1.5 },
    ],
    markers, vlines, xDomain, fmt: (v) => money(v),
  }), {
    headers: ["Date", p.symbol, `Matched ${BATCH.benchmark.symbol}`],
    rows: dates.map((d, i) => [d, money(prices[i]), money(matched[i])]),
  }, [
    { label: p.symbol, color: C.series[0], kind: "line" },
    { label: `if it had matched ${BATCH.benchmark.symbol}`, color: C.hold, kind: "line" },
    { label: "BUY target", color: C.buy, kind: "up" },
    { label: "SELL target", color: C.sell, kind: "down" },
    { label: "HOLD target", color: C.hold, kind: "dot" },
  ]);
}

function renderBatch() {
  if (!BATCH) return;
  const h = el("rig-horizon").value;
  renderBatchKpis(h);
  renderScatter(h);
  renderMultiples();
  renderDetail();
}

function setLiveStatus(marketOpen, asOf) {
  const box = el("live-status");
  const quoteTimes = Object.values(BATCH ? BATCH.prices : {}).map((s) => lastDate(s)).sort();
  const last = quoteTimes[quoteTimes.length - 1];
  const at = asOf ? new Date(asOf).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  box.className = `live-status ${marketOpen ? "is-live" : ""}`;
  box.textContent = marketOpen
    ? `● Live · prices refresh every minute · updated ${at}`
    : `Market closed · showing last close${last ? ` (${shortDate(last)})` : ""} · checked ${at}`;
}

async function loadBatch() {
  const id = el("rig-batch").value;
  if (!id) return;
  const body = el("batch-body");
  body.classList.add("is-loading");
  try {
    const res = await fetch(`/api/testing/batch/${encodeURIComponent(id)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not load batch");
    BATCH = data;
    if (!BATCH.predictions.some((p) => p.symbol === SELECTED)) SELECTED = (BATCH.predictions[0] || {}).symbol;
    renderBatch();
    setLiveStatus(data.market_open, data.as_of);
    scheduleLive(data.market_open);
  } catch (err) {
    el("live-status").textContent = err.message;
  } finally {
    body.classList.remove("is-loading");
  }
}

// Merge the latest quotes into the loaded series (replace today's point or append it).
function mergeQuotes(quotes) {
  const apply = (series, q) => {
    if (!series || !q) return;
    const date = q.time.slice(0, 10);
    const n = series.dates.length;
    if (n && series.dates[n - 1] === date) series.closes[n - 1] = q.price;
    else if (!n || date > series.dates[n - 1]) { series.dates.push(date); series.closes.push(q.price); }
  };
  for (const [sym, q] of Object.entries(quotes)) apply(sym === BATCH.benchmark.symbol ? BATCH.benchmark : BATCH.prices[sym], q);
}

async function refreshLive() {
  if (!BATCH) return;
  const symbols = Object.keys(BATCH.prices);
  if (!symbols.length) return;
  const body = el("batch-body");
  body.classList.add("is-loading");
  try {
    const res = await fetch(`/api/testing/live?symbols=${encodeURIComponent(symbols.join(","))}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Live prices unavailable");
    mergeQuotes(data.quotes);
    renderBatch();
    setLiveStatus(data.market_open, data.as_of);
    scheduleLive(data.market_open);
  } catch (err) {
    el("live-status").textContent = `Couldn't refresh prices: ${err.message}`;
  } finally {
    body.classList.remove("is-loading");
  }
}

function scheduleLive(marketOpen) {
  clearTimeout(liveTimer);
  if (marketOpen) liveTimer = setTimeout(refreshLive, LIVE_REFRESH_MS);
}

// ------------------------------------------------------------------ wiring
el("rig-model")?.addEventListener("change", renderModel);
el("rig-batch")?.addEventListener("change", loadBatch);
el("rig-horizon")?.addEventListener("change", renderBatch);
el("rig-refresh")?.addEventListener("click", refreshLive);
el("detail-range")?.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-range]");
  if (!btn) return;
  DETAIL_RANGE = btn.dataset.range;
  el("detail-range").querySelectorAll("button").forEach((b) => b.classList.toggle("is-active", b === btn));
  if (BATCH) renderDetail();
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    renderModel();
    renderBatch();
  }, 150);
});

renderAllModels();
renderModel();
loadBatch();
