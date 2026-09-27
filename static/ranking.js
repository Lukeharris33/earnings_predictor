// Model ranking: leaderboard by strength of evidence, what the better models
// share, and a head-to-head explainer. Chart helpers come from charts.js.

const DATA = JSON.parse(el("ranking-data").textContent || "{}");
const ENTRIES = Object.fromEntries((DATA.entries || []).map((e) => [e.id, e]));
const GROUP_LABELS = DATA.group_labels || {};
const COLOR_A = C.series[0];
const COLOR_B = C.series[1];

let FOCUS = "overall";
let TRACK = null;          // live record per model, loaded after first paint
let A = null, B = null;    // compared model ids
let IMP_H = null;          // horizon shown in the importance chart

function ranked(focus = FOCUS) {
  return (DATA.boards[focus] || []).map((id) => ENTRIES[id]).filter((e) => e && e.scores[focus]);
}
function focusLabel(f = FOCUS) { return f === "overall" ? "all horizons" : `the ${f} horizon`; }
function ic3(v) { return Number.isFinite(v) ? signed(v, 3) : "—"; }
function pct0(v) { return Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—"; }
function money(v) {
  if (!Number.isFinite(v)) return "—";
  return v >= 1e12 ? `$${(v / 1e12).toFixed(1)}T` : v >= 1e9 ? `$${(v / 1e9).toFixed(0)}B` : `$${(v / 1e6).toFixed(0)}M`;
}
function cell(tr, content, cls = "") {
  const td = tr.insertCell();
  if (cls) td.className = cls;
  if (content instanceof Node) td.appendChild(content); else td.textContent = content;
  return td;
}
function modelColor(id) { return id === A ? COLOR_A : id === B ? COLOR_B : C.hold; }

// ------------------------------------------------------------- leaderboard
function edgeFor(e) {
  const hs = FOCUS === "overall" ? Object.values(e.horizons) : [e.horizons[FOCUS]].filter(Boolean);
  const vals = hs.filter((h) => Number.isFinite(h.edge));
  if (!vals.length) return null;
  return {
    edge: vals.reduce((a, h) => a + h.edge, 0) / vals.length,
    acc: vals.reduce((a, h) => a + h.accuracy, 0) / vals.length,
    base: vals.reduce((a, h) => a + h.baseline, 0) / vals.length,
  };
}
function yearsFor(e) {
  const hs = FOCUS === "overall" ? Object.values(e.horizons) : [e.horizons[FOCUS]].filter(Boolean);
  const pos = hs.reduce((a, h) => a + h.years_positive, 0);
  const all = hs.reduce((a, h) => a + h.years, 0);
  return all ? `${pos} of ${all}` : "—";
}

function renderLeaderboard() {
  const list = ranked();
  el("forest-sub").textContent = `${FOCUS === "overall" ? "Average across horizons" : `${FOCUS} horizon`}. The score is where each line's left end sits. Model A and B from the comparison below are colored; the rest are gray.`;
  mountChart(el("chart-forest"), (holder, w) => drawForest(holder, w, {
    rows: list.map((e) => {
      const s = e.scores[FOCUS];
      return {
        label: e.name,
        series: [{
          value: s.ic, lo: s.lo, hi: s.hi, color: modelColor(e.id),
          tipRows: [
            { label: "IC", value: ic3(s.ic) },
            { label: "95% range", value: `${ic3(s.lo)} to ${ic3(s.hi)}` },
            { label: "score (low end)", value: ic3(s.score) },
            { label: "tests", value: s.n.toLocaleString() },
          ],
        }],
      };
    }),
    fmt: (v) => signed(v, 2),
  }), {
    headers: ["Model", "IC", "95% low", "95% high", "Tests"],
    rows: list.map((e) => { const s = e.scores[FOCUS]; return [e.name, ic3(s.ic), ic3(s.lo), ic3(s.hi), s.n.toLocaleString()]; }),
  });

  const table = el("leaderboard");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  ["#", "Model", "Score", "IC (95% range)", "Tests", "Years IC > 0", "Accuracy vs baseline", "Live record", "Diagnostics"].forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (i >= 2 && i <= 7) th.className = "num";
    head.appendChild(th);
  });
  const body = table.createTBody();
  list.forEach((e, i) => {
    const s = e.scores[FOCUS];
    const tr = body.insertRow();
    cell(tr, String(i + 1), "mono");
    const name = document.createElement("span");
    name.className = "model-name";
    const key = document.createElement("span");
    key.className = "model-key";
    key.style.background = modelColor(e.id);
    const label = document.createElement("span");
    label.textContent = e.name + (s.partial ? " (some horizons missing)" : "");
    name.append(key, label);
    cell(tr, name, "model-cell");
    cell(tr, ic3(s.score), `mono num ${s.score > 0 ? "is-good" : "is-bad"}`);
    cell(tr, `${ic3(s.ic)}  (${ic3(s.lo)} to ${ic3(s.hi)})`, "mono num");
    cell(tr, s.n.toLocaleString(), "mono num");
    cell(tr, yearsFor(e), "mono num");
    const edge = edgeFor(e);
    cell(tr, edge ? `${pct0(edge.acc)} vs ${pct0(edge.base)} (${signed(edge.edge * 100)} pts)` : "—", "mono num");
    const rec = TRACK && TRACK[e.id] && TRACK[e.id][FOCUS === "overall" ? "all" : FOCUS];
    cell(tr, TRACK === null ? "loading…" : rec && rec.n ? `${rec.hits}/${rec.n} right (${pct0(rec.hits / rec.n)})` : "no closed calls", "mono num");
    cell(tr, diagnosticsCell(e));
  });

  const un = (DATA.entries || []).filter((e) => !e.scores[FOCUS]);
  el("unrankable").textContent = un.length
    ? `Not ranked for ${focusLabel()}: ${un.map((e) => `${e.name} (${e.unrankable_reason || "no result at this horizon"})`).join("; ")}`
    : "";
}

function diagnosticsCell(e) {
  const wrap = document.createElement("span");
  if (e.importance) {
    wrap.className = "muted";
    wrap.textContent = e.diagnostics_method && e.diagnostics_method.startsWith("backfilled") ? "✓ backfilled" : "✓ at training";
    return wrap;
  }
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn-ghost btn-small";
  btn.textContent = "Run diagnostics";
  btn.addEventListener("click", () => runDiagnostics(e, btn, wrap));
  wrap.appendChild(btn);
  return wrap;
}

async function runDiagnostics(e, btn, wrap) {
  btn.disabled = true;
  const status = document.createElement("span");
  status.className = "muted diag-status";
  wrap.appendChild(status);
  try {
    const { job_id } = await postJSON(`/api/models/${encodeURIComponent(e.id)}/diagnostics`, {});
    poll(`/api/diagnostics/${encodeURIComponent(job_id)}`, 3000, (st) => {
      status.textContent = st.done ? st.message : `${Math.round((st.progress || 0) * 100)}% · ${st.message}`;
      if (!st.done) return false;
      if (!st.error) window.location.reload();
      else btn.disabled = false;
      return true;
    }, () => { status.textContent = "Lost contact with the server."; btn.disabled = false; });
  } catch (err) {
    status.textContent = err.message;
    btn.disabled = false;
  }
}

async function loadTrackRecord() {
  try {
    const res = await fetch("/api/ranking/track-record");
    const data = await res.json();
    TRACK = res.ok ? data : {};
  } catch (err) {
    TRACK = {};
  }
  renderLeaderboard();
}

// ------------------------------------------------------------ commonalities
function renderCommon() {
  const c = (DATA.commonalities || {})[FOCUS] || {};
  const box = el("common-body");
  box.replaceChildren();
  if (!c.enough) {
    const p = document.createElement("p");
    p.className = "field-hint";
    p.textContent = `Needs at least ${c.needed || 4} ranked models with diagnostics; there ${c.n === 1 ? "is" : "are"} ${c.n || 0}. Train more models or run diagnostics on existing ones and this fills in.`;
    box.appendChild(p);
  } else {
    const p = document.createElement("p");
    p.className = "field-hint";
    p.textContent = `Across ${c.n} models, ranked by ${focusLabel()}. Rank correlation from −1 to +1: positive means models with more of this scored higher. With ${c.n} models, anything below about ±0.8 could easily be chance.`;
    box.appendChild(p);
    const list = document.createElement("ul");
    list.className = "common-list";
    for (const r of c.correlations) {
      const li = document.createElement("li");
      const bar = document.createElement("span");
      bar.className = "common-bar";
      const fill = document.createElement("span");
      fill.className = `common-fill ${r.rho >= 0 ? "pos" : "neg"}`;
      fill.style.width = `${Math.abs(r.rho) * 50}%`;
      fill.style[r.rho >= 0 ? "left" : "right"] = "50%";
      bar.appendChild(fill);
      const label = document.createElement("span");
      label.textContent = r.label;
      const val = document.createElement("span");
      val.className = "mono";
      const strength = Math.abs(r.rho) >= 0.8 ? "strong" : Math.abs(r.rho) >= 0.5 ? "moderate" : "weak";
      val.textContent = `${signed(r.rho, 2)} · ${strength}, ${r.rho >= 0 ? "more → better" : "more → worse"}`;
      li.append(label, bar, val);
      list.appendChild(li);
    }
    box.appendChild(list);
  }
  renderProfiles();
}

function renderProfiles() {
  const list = ranked().concat((DATA.entries || []).filter((e) => !e.scores[FOCUS] && e.version === 2));
  const attrs = [
    ["Rank", (e, i) => (e.scores[FOCUS] ? String(i + 1) : "—")],
    ["Score", (e) => (e.scores[FOCUS] ? ic3(e.scores[FOCUS].score) : "—")],
    ["Tickers", (e) => (e.profile ? String(e.profile.tickers_used) : String(e.tickers))],
    ["Training reports", (e) => (e.profile ? e.profile.rows.toLocaleString() : "—")],
    ["Years of history", (e) => (e.profile ? String(e.profile.years_of_history) : "—")],
    ["Sectors covered", (e) => (e.profile ? String(e.profile.sectors_covered) : "—")],
    ["Largest sector", (e) => {
      if (!e.profile) return "—";
      const [s, v] = Object.entries(e.profile.sector_mix).sort((a, b) => b[1] - a[1])[0] || [];
      return s ? `${s.replace(/_/g, " ")} ${pct0(v)}` : "—";
    }],
    ["Loss-making quarters", (e) => (e.profile ? pct0(e.profile.loss_making_share) : "—")],
    ["Median market cap", (e) => (e.profile ? money(e.profile.median_market_cap) : "—")],
    ["Under $2B market cap", (e) => (e.profile ? pct0(e.profile.small_cap_share) : "—")],
    ["Thin tickers (< 8 reports)", (e) => (e.profile ? (e.profile.thin_tickers.join(", ") || "none") : "—")],
    ["Model type", (e) => e.config.model_type || "—"],
    ["Epochs / batch size", (e) => `${e.config.epochs ?? "—"} / ${e.config.batch_size ?? "—"}`],
  ];
  const rows = attrs.map(([label, fn]) => [label, ...list.map((e, i) => fn(e, i))]);
  el("profiles").replaceChildren(buildTable(["", ...list.map((e) => e.name)], rows));
}

// ----------------------------------------------------------- head to head
function fillCompareSelects() {
  const all = ranked().concat((DATA.entries || []).filter((e) => !e.scores[FOCUS]));
  for (const id of ["cmp-a", "cmp-b"]) {
    const sel = el(id);
    sel.replaceChildren(...all.map((e) => {
      const o = document.createElement("option");
      o.value = e.id;
      o.textContent = e.name;
      return o;
    }));
  }
  const list = ranked();
  if (!A || !ENTRIES[A]) A = list[0] && list[0].id;
  if (!B || !ENTRIES[B] || B === A) B = (list.find((e) => e.id !== A) || {}).id;
  el("cmp-a").value = A || "";
  el("cmp-b").value = B || "";
}

async function renderCompare() {
  const fl = el("findings"), ex = el("experiments");
  if (!A || !B || A === B) {
    fl.replaceChildren(Object.assign(document.createElement("li"), { textContent: "Pick two different models." }));
    ex.replaceChildren();
    return;
  }
  const a = ENTRIES[A], b = ENTRIES[B];
  renderCompareCharts(a, b);
  fl.classList.add("is-loading");
  try {
    const res = await fetch(`/api/ranking/compare?a=${encodeURIComponent(A)}&b=${encodeURIComponent(B)}&focus=${FOCUS}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Comparison failed");
    const icons = { headline: "→", significant: "✓", noise: "≈", evidence: "!", consistency: "↕", family: "◆", data: "▤", settings: "⚙", importance: "★", info: "i" };
    fl.replaceChildren(...data.findings.map((f) => {
      const li = document.createElement("li");
      li.className = `finding finding-${f.kind}`;
      const icon = document.createElement("span");
      icon.className = "finding-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = icons[f.kind] || "•";
      const t = document.createElement("span");
      t.textContent = f.text;
      li.append(icon, t);
      return li;
    }));
    ex.replaceChildren(...(data.experiments.length ? data.experiments : ["No obvious single difference to test; compare them on new batches over time."]).map((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      return li;
    }));
  } catch (err) {
    fl.replaceChildren(Object.assign(document.createElement("li"), { textContent: err.message }));
  } finally {
    fl.classList.remove("is-loading");
  }
}

function renderCompareCharts(a, b) {
  const legend = [
    { label: a.name, color: COLOR_A, kind: "dot" },
    { label: b.name, color: COLOR_B, kind: "dot" },
  ];
  const hs = DATA.horizons.filter((h) => a.horizons[h] || b.horizons[h]);
  const serie = (e, h, color) => {
    const x = e.horizons[h];
    return x ? {
      name: e.name, value: x.ic, lo: x.ic_lo, hi: x.ic_hi, color,
      tipRows: [
        { color, label: "IC", value: ic3(x.ic) },
        { label: "95% range", value: `${ic3(x.ic_lo)} to ${ic3(x.ic_hi)}` },
        { label: "tests", value: x.n.toLocaleString() },
      ],
    } : null;
  };
  mountChart(el("chart-cmp-ic"), (holder, w) => drawForest(holder, w, {
    rows: hs.map((h) => ({ label: h, series: [serie(a, h, COLOR_A), serie(b, h, COLOR_B)].filter(Boolean) })),
    fmt: (v) => signed(v, 2),
  }), {
    headers: ["Horizon", `${a.name} IC`, `${a.name} range`, `${b.name} IC`, `${b.name} range`],
    rows: hs.map((h) => [h, ...[a, b].flatMap((e) => {
      const x = e.horizons[h];
      return x ? [ic3(x.ic), `${ic3(x.ic_lo)} to ${ic3(x.ic_hi)}`] : ["—", "—"];
    })]),
  }, legend.map((l) => ({ ...l, kind: "line" })));

  // Importance.
  const impBox = el("importance-horizon");
  const impHs = DATA.horizons.filter((h) => (a.importance || {})[h] || (b.importance || {})[h]);
  const fig = el("chart-importance");
  if (!impHs.length) {
    impBox.replaceChildren();
    fig.querySelector(".chart-body").textContent = "Run diagnostics on both models (Leaderboard → Diagnostics) to see what each relies on.";
    fig.querySelector(".view-toggle")?.remove();
  } else {
    if (!impHs.includes(IMP_H)) IMP_H = impHs.includes(FOCUS) ? FOCUS : impHs[0];
    impBox.replaceChildren(...impHs.map((h) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = h;
      btn.className = h === IMP_H ? "is-active" : "";
      btn.addEventListener("click", () => { IMP_H = h; renderCompareCharts(a, b); });
      return btn;
    }));
    const ia = (a.importance || {})[IMP_H] || {}, ib = (b.importance || {})[IMP_H] || {};
    const groups = [...new Set([...Object.keys(ia), ...Object.keys(ib)])]
      .map((g) => ({ g, va: ia[g] ? ia[g].ic_drop : NaN, vb: ib[g] ? ib[g].ic_drop : NaN }))
      .sort((x, y) => Math.max(y.va || -1, y.vb || -1) - Math.max(x.va || -1, x.vb || -1));
    mountChart(fig, (holder, w) => drawHBars(holder, w, {
      rows: groups.map((r) => ({ label: GROUP_LABELS[r.g] || r.g, values: [r.va, r.vb] })),
      series: [{ name: a.name, color: COLOR_A }, { name: b.name, color: COLOR_B }],
      fmt: (v) => (Number.isFinite(v) ? signed(v, 3) : "—"),
    }), {
      headers: ["Input group", `${a.name} IC lost`, `${b.name} IC lost`],
      rows: groups.map((r) => [GROUP_LABELS[r.g] || r.g, ic3(r.va), ic3(r.vb)]),
    }, legend.map((l) => ({ ...l, kind: "rect" })));
  }

  // Sector mix.
  const sfig = el("chart-sectors");
  if (!a.profile || !b.profile) {
    sfig.querySelector(".chart-body").textContent = "Run diagnostics on both models to see their training mix.";
    sfig.querySelector(".view-toggle")?.remove();
  } else {
    const sectors = [...new Set([...Object.keys(a.profile.sector_mix), ...Object.keys(b.profile.sector_mix)])]
      .map((s) => ({ s, va: (a.profile.sector_mix[s] || 0) * 100, vb: (b.profile.sector_mix[s] || 0) * 100 }))
      .sort((x, y) => Math.max(y.va, y.vb) - Math.max(x.va, x.vb));
    mountChart(sfig, (holder, w) => drawHBars(holder, w, {
      rows: sectors.map((r) => ({ label: r.s.replace(/_/g, " "), values: [r.va, r.vb] })),
      series: [{ name: a.name, color: COLOR_A }, { name: b.name, color: COLOR_B }],
      fmt: (v) => `${v.toFixed(0)}%`,
    }), {
      headers: ["Sector", a.name, b.name],
      rows: sectors.map((r) => [r.s.replace(/_/g, " "), `${r.va.toFixed(1)}%`, `${r.vb.toFixed(1)}%`]),
    }, legend.map((l) => ({ ...l, kind: "rect" })));
  }
}

// ------------------------------------------------------------------ wiring
function renderAll() {
  fillCompareSelects();
  renderLeaderboard();
  renderCommon();
  renderCompare();
}

el("rank-focus").addEventListener("change", (e) => { FOCUS = e.target.value; renderAll(); });
el("cmp-a").addEventListener("change", (e) => { A = e.target.value; renderLeaderboard(); renderCompare(); });
el("cmp-b").addEventListener("change", (e) => { B = e.target.value; renderLeaderboard(); renderCompare(); });

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    renderLeaderboard();
    if (A && B && A !== B) renderCompareCharts(ENTRIES[A], ENTRIES[B]);
  }, 150);
});

if ((DATA.entries || []).length) {
  renderAll();
  loadTrackRecord();
} else {
  el("leaderboard-section").querySelector(".chart-body").textContent = "No models yet. Train one on the dashboard.";
}
