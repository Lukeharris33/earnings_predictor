// Shared SVG chart kit for the testing and ranking pages: formatting,
// scales, tooltips, chart cards with a table view, and the chart types.
// Every chart has a table view, and every value a tooltip shows is also in
// that table.

const HORIZONS = ["1d", "1w", "1m", "1y"];
const SVG_NS = "http://www.w3.org/2000/svg";

function el(id) { return document.getElementById(id); }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

const C = {
  series: ["--series-1", "--series-2", "--series-3", "--series-4"].map(css),
  pos: css("--div-pos"),
  neg: css("--div-neg"),
  buy: css("--viz-buy"),
  sell: css("--viz-sell"),
  hold: css("--viz-hold"),
  muted: css("--viz-muted"),
  grid: css("--viz-grid"),
  axis: css("--viz-axis"),
  surface: css("--surface"),
};
const ACTION_COLOR = { BUY: C.buy, SELL: C.sell, HOLD: C.hold };

// ------------------------------------------------------------ formatting
function signed(v, d = 1) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
}
function pctOrDash(v, d = 1) { return Number.isFinite(v) ? `${signed(v, d)}%` : "—"; }
function pctPts(v, d = 1) { return v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`; }
function money(v) { return `$${v >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(2)}`; }
function shortDate(iso) {
  const d = new Date(`${iso}T12:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function addDays(iso, n) {
  const d = new Date(`${iso}T12:00:00`);
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}
function dayNum(iso) { return Date.parse(`${iso}T00:00:00Z`) / 86_400_000; }

// ----------------------------------------------------------- svg helpers
function svg(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function text(parent, x, y, str, attrs = {}) {
  const t = svg("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function scale(d0, d1, r0, r1) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}
function niceTicks(min, max, count = 5) {
  if (min === max) { min -= 1; max += 1; }
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  // Only ticks inside the domain; rounding outward would draw labels off the plot.
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
function extent(values, padFrac = 0.08, includeZero = true) {
  const vals = values.filter((v) => Number.isFinite(v));
  let lo = Math.min(...vals, includeZero ? 0 : Infinity);
  let hi = Math.max(...vals, includeZero ? 0 : -Infinity);
  if (!vals.length) { lo = -1; hi = 1; }
  const pad = (hi - lo || 1) * padFrac;
  return [lo - (lo < 0 || !includeZero ? pad : 0), hi + (hi > 0 || !includeZero ? pad : 0)];
}
// Column path with a 4px rounded data end and a square baseline end.
function columnPath(x, w, yBase, yEnd) {
  const r = Math.min(4, w / 2, Math.abs(yEnd - yBase));
  if (yEnd < yBase) {
    return `M${x},${yBase}V${yEnd + r}Q${x},${yEnd} ${x + r},${yEnd}H${x + w - r}Q${x + w},${yEnd} ${x + w},${yEnd + r}V${yBase}Z`;
  }
  return `M${x},${yBase}V${yEnd - r}Q${x},${yEnd} ${x + r},${yEnd}H${x + w - r}Q${x + w},${yEnd} ${x + w},${yEnd - r}V${yBase}Z`;
}
function marker(parent, shape, x, y, color, { r = 5, hollow = false } = {}) {
  const common = { fill: hollow ? C.surface : color, stroke: hollow ? color : C.surface, "stroke-width": 2 };
  if (shape === "up") return svg("path", { d: `M${x},${y - r - 1}L${x + r + 1},${y + r}L${x - r - 1},${y + r}Z`, ...common, "stroke-linejoin": "round" }, parent);
  if (shape === "down") return svg("path", { d: `M${x},${y + r + 1}L${x + r + 1},${y - r}L${x - r - 1},${y - r}Z`, ...common, "stroke-linejoin": "round" }, parent);
  return svg("circle", { cx: x, cy: y, r, ...common }, parent);
}
const ACTION_SHAPE = { BUY: "up", SELL: "down", HOLD: "dot" };

function yAxis(g, ticks, y, x0, x1, fmt) {
  for (const t of ticks) {
    svg("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), stroke: t === 0 ? C.axis : C.grid, "stroke-width": 1 }, g);
    text(g, x0 - 8, y(t) + 4, fmt(t), { class: "viz-tick", "text-anchor": "end" });
  }
}

// ---------------------------------------------------------------- tooltip
const tip = el("viz-tooltip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  const h = document.createElement("div");
  h.className = "viz-tip-title";
  h.textContent = title;
  tip.appendChild(h);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "viz-tip-row";
    if (r.color) {
      const key = document.createElement("span");
      key.className = "viz-tip-key";
      key.style.background = r.color;
      row.appendChild(key);
    }
    const v = document.createElement("strong");
    v.textContent = r.value;
    const l = document.createElement("span");
    l.textContent = r.label;
    row.append(v, l);
    tip.appendChild(row);
  }
  tip.classList.remove("hidden");
  const pad = 14;
  const { innerWidth: W, innerHeight: Hh } = window;
  const rect = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + rect.width > W - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > Hh - 8) y = evt.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.classList.add("hidden"); }
function bindTip(node, getTip) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", (e) => { const t = getTip(); showTip(e, t.title, t.rows); });
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    const t = getTip();
    showTip({ clientX: r.right, clientY: r.top }, t.title, t.rows);
  });
  node.addEventListener("blur", hideTip);
}

// ------------------------------------------------- chart card + table view
// Each figure gets a Chart/Table toggle; `table` = {headers, rows}.
function mountChart(figure, draw, table, legend) {
  const body = figure.querySelector(".chart-body");
  let toggle = figure.querySelector(".view-toggle");
  if (!toggle) {
    toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "btn-ghost view-toggle";
    toggle.textContent = "Table";
    figure.querySelector("figcaption").appendChild(toggle);
    toggle.addEventListener("click", () => {
      figure.dataset.view = figure.dataset.view === "table" ? "chart" : "table";
      toggle.textContent = figure.dataset.view === "table" ? "Chart" : "Table";
      figure._render();
    });
  }
  figure._render = () => {
    body.replaceChildren();
    if (figure.dataset.view === "table") {
      body.appendChild(buildTable(table.headers, table.rows));
      return;
    }
    if (legend && legend.length) body.appendChild(buildLegend(legend));
    const holder = document.createElement("div");
    holder.className = "chart-svg";
    body.appendChild(holder);
    draw(holder, Math.max(280, holder.clientWidth || body.clientWidth));
  };
  figure._render();
}
function buildTable(headers, rows) {
  const wrap = document.createElement("div");
  wrap.className = "table-scroll";
  const t = document.createElement("table");
  t.className = "data-table";
  const thead = t.createTHead().insertRow();
  headers.forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (i > 0) th.className = "num";
    thead.appendChild(th);
  });
  const tb = t.createTBody();
  for (const r of rows) {
    const tr = tb.insertRow();
    r.forEach((v, i) => {
      const td = tr.insertCell();
      td.textContent = v;
      td.className = i > 0 ? "mono num" : "";
    });
  }
  wrap.appendChild(t);
  return wrap;
}
// legend items: {label, color, kind: "line"|"rect"|"up"|"down"|"dot"|"hollow"}
function buildLegend(items) {
  const box = document.createElement("div");
  box.className = "viz-legend";
  for (const it of items) {
    const item = document.createElement("span");
    item.className = "viz-legend-item";
    const s = svg("svg", { width: 16, height: 12, "aria-hidden": "true" });
    if (it.kind === "line") svg("line", { x1: 1, x2: 15, y1: 6, y2: 6, stroke: it.color, "stroke-width": 2, "stroke-linecap": "round" }, s);
    else if (it.kind === "rect") svg("rect", { x: 3, y: 1, width: 10, height: 10, rx: 2, fill: it.color }, s);
    else if (it.kind === "hollow") svg("circle", { cx: 8, cy: 6, r: 4, fill: "none", stroke: it.color, "stroke-width": 2 }, s);
    else marker(s, it.kind, 8, 6, it.color, { r: 3.5 });
    const label = document.createElement("span");
    label.textContent = it.label;
    item.append(s, label);
    box.appendChild(item);
  }
  return box;
}

// ------------------------------------------------------------- chart types
// Vertical columns from a zero baseline, grouped per category. A series'
// color may be a function of the value (diverging: sign picks the pole).
function drawColumns(holder, width, { categories, series, fmt, tickFmt, height = 240, labelTips = true }) {
  const m = { t: 18, r: 12, b: 30, l: 48 };
  const s = svg("svg", { width, height, class: "viz" }, holder);
  const all = series.flatMap((se) => se.values);
  const [lo, hi] = extent(all, 0.12);
  const y = scale(lo, hi, height - m.b, m.t);
  const ticks = niceTicks(lo, hi, 4);
  const g = svg("g", {}, s);
  yAxis(g, ticks, y, m.l, width - m.r, tickFmt || fmt);
  const band = (width - m.l - m.r) / categories.length;
  const barW = Math.min(24, (band * 0.6 - 2 * (series.length - 1)) / series.length);
  const groupW = barW * series.length + 2 * (series.length - 1);
  categories.forEach((cat, ci) => {
    const gx = m.l + band * ci + (band - groupW) / 2;
    text(g, m.l + band * ci + band / 2, height - m.b + 18, cat, { class: "viz-tick", "text-anchor": "middle" });
    series.forEach((se, si) => {
      const v = se.values[ci];
      if (!Number.isFinite(v)) return;
      const x = gx + si * (barW + 2);
      const color = typeof se.color === "function" ? se.color(v) : se.color;
      const hit = svg("g", { class: "viz-hit" }, g);
      svg("rect", { x: x - 3, y: m.t, width: barW + 6, height: height - m.t - m.b, fill: "transparent" }, hit);
      svg("path", { d: columnPath(x, barW, y(0), y(v)), fill: color, class: "viz-mark" }, hit);
      if (series.length === 1 && labelTips) {
        text(g, x + barW / 2, v >= 0 ? y(v) - 6 : y(v) + 14, fmt(v), { class: "viz-label", "text-anchor": "middle" });
      }
      bindTip(hit, () => ({
        title: cat,
        rows: series.map((o) => ({ color: typeof o.color === "function" ? o.color(o.values[ci]) : o.color, label: o.name, value: fmt(o.values[ci]) })),
      }));
    });
  });
}

// Lines over categorical x; crosshair snaps to the nearest category.
function drawCategoryLines(holder, width, { categories, series, fmt, height = 240 }) {
  const m = { t: 16, r: 16, b: 30, l: 52 };
  const s = svg("svg", { width, height, class: "viz" }, holder);
  const [lo, hi] = extent(series.flatMap((se) => se.values), 0.15);
  const y = scale(lo, hi, height - m.b, m.t);
  const x = scale(0, Math.max(1, categories.length - 1), m.l + 16, width - m.r - 16);
  const g = svg("g", {}, s);
  yAxis(g, niceTicks(lo, hi, 4), y, m.l, width - m.r, fmt);
  categories.forEach((c, i) => text(g, x(i), height - m.b + 18, c, { class: "viz-tick", "text-anchor": "middle" }));
  for (const se of series) {
    let d = "";
    se.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      d += `${d ? "L" : "M"}${x(i)},${y(v)}`;
    });
    svg("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
    se.values.forEach((v, i) => { if (Number.isFinite(v)) marker(g, "dot", x(i), y(v), se.color, { r: 4 }); });
  }
  const cross = svg("line", { y1: m.t, y2: height - m.b, stroke: C.axis, "stroke-width": 1, visibility: "hidden" }, g);
  const overlay = svg("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, s);
  overlay.addEventListener("pointermove", (e) => {
    const px = e.clientX - s.getBoundingClientRect().left;
    let best = 0;
    categories.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
    cross.setAttribute("x1", x(best)); cross.setAttribute("x2", x(best)); cross.setAttribute("visibility", "visible");
    showTip(e, categories[best], series.map((se) => ({ color: se.color, label: se.name, value: fmt(se.values[best]) })));
  });
  overlay.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

// Scatter of predicted (x) vs actual (y) with a y=x reference.
function drawScatter(holder, width, { points, height = 320 }) {
  const m = { t: 16, r: 18, b: 42, l: 56 };
  const s = svg("svg", { width, height, class: "viz" }, holder);
  const [xlo, xhi] = extent(points.map((p) => p.x), 0.15);
  const [ylo, yhi] = extent(points.map((p) => p.y), 0.1);
  const x = scale(xlo, xhi, m.l, width - m.r);
  const y = scale(ylo, yhi, height - m.b, m.t);
  const g = svg("g", {}, s);
  yAxis(g, niceTicks(ylo, yhi, 5), y, m.l, width - m.r, (v) => `${signed(v, 0)}%`);
  for (const t of niceTicks(xlo, xhi, 6)) {
    svg("line", { x1: x(t), x2: x(t), y1: m.t, y2: height - m.b, stroke: t === 0 ? C.axis : C.grid, "stroke-width": 1 }, g);
    text(g, x(t), height - m.b + 16, `${signed(t, Math.abs(xhi - xlo) < 6 ? 1 : 0)}%`, { class: "viz-tick", "text-anchor": "middle" });
  }
  text(g, (m.l + width - m.r) / 2, height - 6, "Predicted move vs benchmark", { class: "viz-axis-title", "text-anchor": "middle" });
  text(g, 14, (m.t + height - m.b) / 2, "Actual move", { class: "viz-axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + height - m.b) / 2})` });
  // y = x: a perfect forecast sits on this line.
  const lo = Math.max(xlo, ylo), hi = Math.min(xhi, yhi);
  if (hi > lo) svg("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: C.muted, "stroke-width": 1, opacity: 0.6 }, g);
  for (const p of points) {
    const hit = svg("g", { class: "viz-hit" }, g);
    svg("circle", { cx: x(p.x), cy: y(p.y), r: 12, fill: "transparent" }, hit);
    marker(hit, ACTION_SHAPE[p.action], x(p.x), y(p.y), ACTION_COLOR[p.action], { r: 5, hollow: !p.closed });
    bindTip(hit, () => ({
      title: `${p.symbol} · ${p.action}${p.closed ? "" : " (window still open)"}`,
      rows: [
        { label: "predicted", value: `${signed(p.x, 2)}%` },
        { label: p.closed ? "actual" : "so far", value: `${signed(p.y, 2)}%` },
        { label: "window ends", value: p.windowEnd },
      ],
    }));
  }
}

// Time-series chart over real dates. `lines` = [{name, color, dates, values, width}],
// `markers` = [{date, value, action, label, closed}], `vlines` = [{date, label}].
function drawTimeChart(holder, width, { lines, markers, vlines, xDomain, fmt, height = 340 }) {
  const m = { t: 22, r: 60, b: 30, l: 60 };
  const s = svg("svg", { width, height, class: "viz" }, holder);
  const [d0, d1] = xDomain.map(dayNum);
  const x = scale(d0, d1, m.l, width - m.r);
  const visible = (d) => dayNum(d) >= d0 && dayNum(d) <= d1;
  const vals = lines.flatMap((l) => l.values.filter((_, i) => visible(l.dates[i])))
    .concat(markers.filter((mk) => visible(mk.date)).map((mk) => mk.value));
  const [lo, hi] = extent(vals, 0.08, false);
  const y = scale(lo, hi, height - m.b, m.t);
  const g = svg("g", {}, s);
  yAxis(g, niceTicks(lo, hi, 5), y, m.l, width - m.r, fmt);
  // Month ticks.
  const start = new Date(`${xDomain[0]}T12:00:00`);
  for (let d = new Date(start.getFullYear(), start.getMonth() + 1, 1); dayNum(d.toISOString().slice(0, 10)) <= d1; d.setMonth(d.getMonth() + 1)) {
    const iso = d.toISOString().slice(0, 10);
    text(g, x(dayNum(iso)), height - m.b + 18, d.toLocaleDateString(undefined, { month: "short" }), { class: "viz-tick", "text-anchor": "middle" });
  }
  const clipId = `clip-${Math.random().toString(36).slice(2)}`;
  const clip = svg("clipPath", { id: clipId }, svg("defs", {}, s));
  svg("rect", { x: m.l, y: 0, width: width - m.l - m.r, height }, clip);
  // Window-end rules; labels closer than ~40px merge ("release · 1d") instead of overprinting.
  const labels = [];
  for (const v of vlines.filter((v) => visible(v.date)).sort((a, b) => dayNum(a.date) - dayNum(b.date))) {
    const vx = x(dayNum(v.date));
    svg("line", { x1: vx, x2: vx, y1: m.t, y2: height - m.b, stroke: C.grid, "stroke-width": 1 }, g);
    const prev = labels[labels.length - 1];
    if (prev && vx - prev.x < 40) prev.text += ` · ${v.label}`;
    else labels.push({ x: vx, text: v.label });
  }
  for (const l of labels) text(g, l.x, m.t - 8, l.text, { class: "viz-tick", "text-anchor": "start" });
  const plot = svg("g", { "clip-path": `url(#${clipId})` }, g);
  for (const l of lines) {
    let d = "";
    l.dates.forEach((date, i) => {
      if (!Number.isFinite(l.values[i])) return;
      d += `${d ? "L" : "M"}${x(dayNum(date))},${y(l.values[i])}`;
    });
    svg("path", { d, fill: "none", stroke: l.color, "stroke-width": l.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, plot);
  }
  // End labels for the lines (value at the latest point).
  for (const l of lines.filter((l) => l.endLabel)) {
    const i = l.values.length - 1;
    if (i < 0 || !visible(l.dates[i])) continue;
    marker(g, "dot", x(dayNum(l.dates[i])), y(l.values[i]), l.color, { r: 4 });
    text(g, x(dayNum(l.dates[i])) + 8, y(l.values[i]) + 4, fmt(l.values[i]), { class: "viz-label" });
  }
  for (const mk of markers) {
    if (!visible(mk.date)) continue;
    const hit = svg("g", { class: "viz-hit" }, g);
    svg("circle", { cx: x(dayNum(mk.date)), cy: y(mk.value), r: 12, fill: "transparent" }, hit);
    marker(hit, ACTION_SHAPE[mk.action], x(dayNum(mk.date)), y(mk.value), ACTION_COLOR[mk.action], { r: 5, hollow: !mk.closed });
    bindTip(hit, () => ({ title: mk.label, rows: mk.tipRows }));
  }
  const cross = svg("line", { y1: m.t, y2: height - m.b, stroke: C.axis, "stroke-width": 1, visibility: "hidden" }, g);
  const overlay = svg("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, s);
  // Markers sit above the overlay so they keep their own tooltips.
  g.querySelectorAll(".viz-hit").forEach((n) => s.appendChild(n));
  const ref = lines[0];
  overlay.addEventListener("pointermove", (e) => {
    const px = e.clientX - s.getBoundingClientRect().left;
    let best = -1;
    ref.dates.forEach((d, i) => {
      if (!visible(d)) return;
      if (best < 0 || Math.abs(x(dayNum(d)) - px) < Math.abs(x(dayNum(ref.dates[best])) - px)) best = i;
    });
    if (best < 0) return;
    const date = ref.dates[best];
    cross.setAttribute("x1", x(dayNum(date))); cross.setAttribute("x2", x(dayNum(date))); cross.setAttribute("visibility", "visible");
    showTip(e, shortDate(date), lines.map((l) => {
      const i = l.dates.indexOf(date);
      return { color: l.color, label: l.name, value: i >= 0 ? fmt(l.values[i]) : "—" };
    }));
  });
  overlay.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

// Tiny line for small multiples: move vs benchmark since release, with call markers.
function drawSpark(holder, width, { dates, values, markers, height = 64 }) {
  const m = { t: 8, r: 6, b: 8, l: 6 };
  const s = svg("svg", { width, height, class: "viz", "aria-hidden": "true" }, holder);
  const [lo, hi] = extent(values.concat(markers.map((mk) => mk.value)), 0.1);
  const x = scale(0, Math.max(1, dates.length - 1), m.l, width - m.r);
  const y = scale(lo, hi, height - m.b, m.t);
  svg("line", { x1: m.l, x2: width - m.r, y1: y(0), y2: y(0), stroke: C.axis, "stroke-width": 1 }, s);
  let d = "";
  values.forEach((v, i) => { if (Number.isFinite(v)) d += `${d ? "L" : "M"}${x(i)},${y(v)}`; });
  svg("path", { d, fill: "none", stroke: C.series[0], "stroke-width": 2, "stroke-linejoin": "round" }, s);
  for (const mk of markers) {
    const i = dates.indexOf(mk.date);
    if (i >= 0) marker(s, ACTION_SHAPE[mk.action], x(i), y(mk.value), ACTION_COLOR[mk.action], { r: 3.5, hollow: !mk.closed });
  }
}

function kpiTile(label, value, sub, subClass = "") {
  const tile = document.createElement("div");
  tile.className = "kpi";
  const l = document.createElement("div"); l.className = "kpi-label"; l.textContent = label;
  const v = document.createElement("div"); v.className = "kpi-value"; v.textContent = value;
  const s = document.createElement("div"); s.className = `kpi-sub ${subClass}`; s.textContent = sub;
  tile.append(l, v, s);
  return tile;
}

// Horizontal interval chart ("forest plot"): one row per item, each series a
// dot at its value with a line across its 95% range. rows =
// [{label, series: [{name, value, lo, hi, color, tipRows}]}].
function drawForest(holder, width, { rows, fmt, height }) {
  const perSeries = Math.max(...rows.map((r) => r.series.length), 1);
  const rowH = 20 + 14 * perSeries;
  const labelW = Math.min(240, Math.max(120, width * 0.34));
  const m = { t: 8, r: 20, b: 30, l: labelW + 12 };
  const h = height || m.t + m.b + rowH * rows.length;
  const s = svg("svg", { width, height: h, class: "viz" }, holder);
  const all = rows.flatMap((r) => r.series.flatMap((se) => [se.lo, se.hi, se.value]));
  const [lo, hi] = extent(all, 0.08);
  const x = scale(lo, hi, m.l, width - m.r);
  const g = svg("g", {}, s);
  for (const t of niceTicks(lo, hi, 5)) {
    svg("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, stroke: t === 0 ? C.axis : C.grid, "stroke-width": 1 }, g);
    text(g, x(t), h - m.b + 16, fmt(t), { class: "viz-tick", "text-anchor": "middle" });
  }
  rows.forEach((row, ri) => {
    const y0 = m.t + ri * rowH;
    const label = text(g, 0, y0 + rowH / 2 + 4, row.label, { class: "viz-row-label" });
    // Truncate long names to the label column; the full name is in the tooltip and table.
    let str = row.label;
    while (label.getComputedTextLength && label.getComputedTextLength() > labelW && str.length > 4) {
      str = str.slice(0, -2);
      label.textContent = `${str.trimEnd()}…`;
    }
    row.series.forEach((se, si) => {
      const cy = y0 + 10 + 7 + si * 14 + (perSeries - row.series.length) * 7;
      const hit = svg("g", { class: "viz-hit" }, g);
      svg("rect", { x: m.l, y: cy - 7, width: width - m.l - m.r, height: 14, fill: "transparent" }, hit);
      if (Number.isFinite(se.lo) && Number.isFinite(se.hi)) {
        svg("line", { x1: x(se.lo), x2: x(se.hi), y1: cy, y2: cy, stroke: se.color, "stroke-width": 2, "stroke-linecap": "round" }, hit);
      }
      marker(hit, "dot", x(se.value), cy, se.color, { r: 5 });
      bindTip(hit, () => ({ title: `${row.label}${se.name ? ` · ${se.name}` : ""}`, rows: se.tipRows }));
    });
  });
}

// Horizontal bars from a zero line, grouped per row. rows = [{label, values: [..]}],
// series = [{name, color}] (one value per series per row).
function drawHBars(holder, width, { rows, series, fmt }) {
  const barH = Math.min(10, 22 / series.length);
  const rowH = barH * series.length + 2 * (series.length - 1) + 14;
  const labelW = Math.min(200, Math.max(110, width * 0.3));
  const m = { t: 6, r: 56, b: 28, l: labelW + 12 };
  const h = m.t + m.b + rowH * rows.length;
  const s = svg("svg", { width, height: h, class: "viz" }, holder);
  const [lo, hi] = extent(rows.flatMap((r) => r.values), 0.1);
  const x = scale(lo, hi, m.l, width - m.r);
  const g = svg("g", {}, s);
  for (const t of niceTicks(lo, hi, 4)) {
    svg("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, stroke: t === 0 ? C.axis : C.grid, "stroke-width": 1 }, g);
    text(g, x(t), h - m.b + 16, fmt(t), { class: "viz-tick", "text-anchor": "middle" });
  }
  rows.forEach((row, ri) => {
    const y0 = m.t + ri * rowH + 7;
    text(g, 0, y0 + (rowH - 14) / 2 + 4, row.label, { class: "viz-row-label" });
    const hit = svg("g", { class: "viz-hit" }, g);
    svg("rect", { x: 0, y: y0 - 7, width, height: rowH, fill: "transparent" }, hit);
    series.forEach((se, si) => {
      const v = row.values[si];
      if (!Number.isFinite(v)) return;
      const y = y0 + si * (barH + 2);
      const x0 = x(Math.min(0, v)), w = Math.max(1, Math.abs(x(v) - x(0)));
      const r = Math.min(4, barH / 2, w);
      // Rounded at the data end only.
      const d = v >= 0
        ? `M${x0},${y}H${x0 + w - r}Q${x0 + w},${y} ${x0 + w},${y + r}V${y + barH - r}Q${x0 + w},${y + barH} ${x0 + w - r},${y + barH}H${x0}Z`
        : `M${x0 + w},${y}H${x0 + r}Q${x0},${y} ${x0},${y + r}V${y + barH - r}Q${x0},${y + barH} ${x0 + r},${y + barH}H${x0 + w}Z`;
      svg("path", { d, fill: se.color, class: "viz-mark" }, hit);
      if (series.length === 1) text(g, v >= 0 ? x(v) + 6 : x(v) - 6, y + barH - 1, fmt(v), { class: "viz-label", "text-anchor": v >= 0 ? "start" : "end" });
    });
    bindTip(hit, () => ({ title: row.label, rows: series.map((se, si) => ({ color: se.color, label: se.name, value: fmt(row.values[si]) })) }));
  });
}

// ------------------------------------------------------------------- fetch
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request to ${url} failed`);
  return data;
}

// Polls `url` every `ms` until `onStatus` returns true; gives up after 5
// consecutive network failures.
function poll(url, ms, onStatus, onLost) {
  let failures = 0;
  const timer = setInterval(async () => {
    let status;
    try {
      status = await (await fetch(url)).json();
      failures = 0;
    } catch (err) {
      if (++failures >= 5) { clearInterval(timer); onLost(); }
      return;
    }
    if (onStatus(status)) clearInterval(timer);
  }, ms);
}
