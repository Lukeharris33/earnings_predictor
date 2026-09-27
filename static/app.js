const HORIZON_ORDER = ["1d", "1w", "1m", "1y"];
const HORIZON_DESC = {
  "1d": "release-day reaction",
  "1w": "1 week after first close",
  "1m": "1 month after first close",
  "1y": "1 year after first close",
};

function el(id) { return document.getElementById(id); }

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function signed(v, digits = 2) {
  if (v === null || v === undefined) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

function todayISO() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// A horizon whose window has already passed is history, not a forecast.
function isClosed(d) {
  return Boolean(d && (d.closed || d.window_closed || (d.window_end && d.window_end <= todayISO())));
}

function actualText(d, vs = "") {
  return d.actual_pct_change === null || d.actual_pct_change === undefined
    ? "" : `actual ${signed(d.actual_pct_change)}%${vs}`;
}

function pct(v, digits = 1) {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;
}

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
      const res = await fetch(url);
      status = await res.json();
      failures = 0;
    } catch (err) {
      if (++failures >= 5) {
        clearInterval(timer);
        onLost();
      }
      return;
    }
    if (onStatus(status)) clearInterval(timer);
  }, ms);
}

// ------------------------------------------------------------ ticker lists
let universeCache = null;
async function loadUniverse() {
  if (!universeCache) {
    const res = await fetch("/api/universe");
    universeCache = await res.json();
  }
  return universeCache;
}

el("load-universe")?.addEventListener("click", async () => {
  const u = await loadUniverse();
  el("tickers").value = u.universe.join(", ");
});

el("load-holdout")?.addEventListener("click", async () => {
  const u = await loadUniverse();
  el("batch-tickers").value = u.holdout.join(", ");
});

// --------------------------------------------------------------- training
const trainForm = el("train-form");
if (trainForm) {
  trainForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const tickers = el("tickers").value.trim();
    if (!tickers) return;

    const statusBox = el("train-status");
    const fill = el("progress-fill");
    const msg = el("status-message");
    const metricsBox = el("status-metrics");

    statusBox.classList.remove("hidden");
    metricsBox.classList.add("hidden");
    msg.classList.remove("is-error", "is-done");
    fill.style.width = "2%";
    msg.textContent = "Starting...";

    let runId;
    try {
      const data = await postJSON("/api/train", {
        symbols: tickers,
        name: el("model-name").value.trim(),
        model_type: el("model-type").value,
        epochs: el("epochs").value,
        batch_size: el("batch-size").value,
      });
      runId = data.run_id;
    } catch (err) {
      msg.textContent = err.message;
      msg.classList.add("is-error");
      return;
    }

    poll(`/api/status/${encodeURIComponent(runId)}`, 2000, (status) => {
      if (status.error && !status.phase) {
        msg.textContent = status.error;
        msg.classList.add("is-error");
        return true;
      }
      fill.style.width = `${Math.round((status.progress || 0) * 100)}%`;
      msg.textContent = status.message || "Working...";
      if (!status.done) return false;
      if (status.error) {
        msg.classList.add("is-error");
      } else {
        msg.classList.add("is-done");
        msg.textContent = `${status.message} (${status.rows_used} rows, ${status.warnings || 0} logged errors)`;
        renderMetrics(status.metrics);
        refreshModelsDropdown();
      }
      return true;
    }, () => {
      msg.textContent = "Lost contact with the server while training.";
      msg.classList.add("is-error");
    });
  });
}

function renderMetrics(metrics) {
  const box = el("status-metrics");
  if (!metrics) return;
  const rows = HORIZON_ORDER.filter((h) => metrics[h]).map((h) => {
    const m = metrics[h];
    if (m.directional_accuracy === undefined) {
      return `<tr><td>${h}</td><td colspan="6" class="muted">${esc(m.reason || "not evaluated")}</td></tr>`;
    }
    const beats = m.directional_accuracy > m.baseline_majority_accuracy;
    return `
      <tr>
        <td>${h}</td>
        <td class="mono num ${beats ? "c-buy" : "c-sell"}">${pct(m.directional_accuracy)}</td>
        <td class="mono num muted">${pct(m.baseline_majority_accuracy)}</td>
        <td class="mono num">${signed(m.ic, 3)}</td>
        <td class="mono num">${m.top_minus_bottom_quintile === null ? "—" : signed(m.top_minus_bottom_quintile) + " pts"}</td>
        <td class="mono num">${m.mae_pct_points.toFixed(2)} <span class="muted">/ ${m.baseline_zero_mae.toFixed(2)}</span></td>
        <td class="mono num">${m.n}${m.folds ? ` <span class="muted">(${m.folds.length} ${m.evaluation === "walk_forward" ? "yrs" : "split"})</span>` : ""}</td>
      </tr>`;
  }).join("");

  let ablation = "";
  const a = metrics["1d"] && metrics["1d"].ablation_vs_trend;
  if (a) {
    ablation = `<p class="field-hint">1d trend-feature test (boosted trees, same folds): IC ${signed(a.with.ic, 3)} with the vs-own-trend features, ${signed(a.without.ic, 3)} without; accuracy ${pct(a.with.directional_accuracy)} vs ${pct(a.without.directional_accuracy)}.</p>`;
  }

  box.innerHTML = `
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>Horizon</th><th class="num">Accuracy</th><th class="num">Baseline</th><th class="num">IC</th><th class="num">Top&minus;bottom 20%</th><th class="num">MAE / zero-guess</th><th class="num">Test rows</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="field-hint">All numbers are out-of-sample. Baseline = always guessing the more common direction. IC = rank correlation between predicted and actual moves; top&minus;bottom = how much the model's top-fifth picks beat its bottom-fifth.</p>
    ${ablation}`;
  box.classList.remove("hidden");
}

// -------------------------------------------------------------- prediction
const predictForm = el("predict-form");
if (predictForm) {
  predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const modelId = el("model-select").value;
    const symbol = el("predict-symbol").value.trim();
    if (!modelId || !symbol) return;

    const resultBox = el("predict-result");
    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `<p class="field-hint">Fetching latest report for ${esc(symbol)} and scoring it...</p>`;

    try {
      const data = await postJSON("/api/predict", {
        model_id: modelId, symbol, name: el("predict-name").value.trim(),
      });
      renderDecisions(resultBox, data);
    } catch (err) {
      resultBox.innerHTML = `<p class="status-message is-error">${esc(err.message)}</p>`;
    }
  });
}

function decisionCards(data) {
  const vs = data.benchmark ? ` vs ${esc(data.benchmark)}` : "";
  return HORIZON_ORDER.filter((h) => data.decisions[h]).map((h) => {
    const d = data.decisions[h];
    const weightPct = Math.round(d.confidence_weight * 100);
    const sub = data.model_version >= 2 ? HORIZON_DESC[h] : `from ${esc(data.earnings_date || "latest report")}`;
    if (isClosed(d)) {
      const actual = actualText(d, vs);
      return `
      <div class="decision-card is-closed" title="This window has already passed, so the call is no longer actionable.">
        <div class="decision-horizon">${h} &middot; ${sub}</div>
        <div class="decision-action">Closed</div>
        <div class="decision-pct">window ended ${esc(d.window_end || "")}</div>
        <div class="decision-weight">predicted ${esc(d.action)} ${signed(d.predicted_pct_change)}%${actual ? ` &middot; ${actual}` : ""}</div>
      </div>`;
    }
    return `
      <div class="decision-card action-${esc(d.action)}">
        <div class="decision-horizon">${h} &middot; ${sub}</div>
        <div class="decision-action">${esc(d.action)}</div>
        <div class="decision-pct">${signed(d.predicted_pct_change)}%${vs}</div>
        <div class="decision-weight">confidence weight ${weightPct}%${d.hold_band_pct !== undefined ? ` &middot; HOLD within &plusmn;${d.hold_band_pct.toFixed(2)}%` : ""}</div>
        <div class="decision-weight">window closes ${esc(d.window_end || "")}</div>
        <div class="weight-track"><div class="weight-fill" style="width:${weightPct}%"></div></div>
      </div>`;
  }).join("");
}

function renderDecisions(box, data) {
  const title = data.name ? `${esc(data.symbol)} <span class="muted">&middot; ${esc(data.name)}</span>` : esc(data.symbol);
  box.innerHTML = `<h3>${title}</h3>`
    + (data.note ? `<p class="field-hint">${esc(data.note)}</p>` : "")
    + `<div class="decision-grid">${decisionCards(data)}</div>`
    + renderBreakdown(data.source, data.inputs);
}

// ------------------------------------------------ what the model was fed
// kind: "pct" = ratio shown as %, "pctpts" = already in % units,
// "log" = log10 of dollars, "days", "x" = plain multiple.
const FEATURE_INFO = {
  log_revenue: ["Revenue (log10 $)", "log"],
  log_assets: ["Total assets (log10 $)", "log"],
  gross_margin: ["Gross margin", "pct"],
  operating_margin: ["Operating margin", "pct"],
  net_margin: ["Net margin", "pct"],
  rnd_to_revenue: ["R&D / revenue", "pct"],
  sga_to_revenue: ["SG&A / revenue", "pct"],
  sbc_to_revenue: ["Stock comp / revenue", "pct"],
  ocf_margin: ["Operating cash flow margin", "pct"],
  fcf_margin: ["Free cash flow margin", "pct"],
  capital_return_to_revenue: ["Buybacks + dividends / revenue", "pct"],
  revenue_growth_qoq: ["Revenue growth QoQ", "pct"],
  revenue_growth_yoy: ["Revenue growth YoY", "pct"],
  operating_income_growth_yoy: ["Operating income growth YoY", "pct"],
  net_income_growth_yoy: ["Net income growth YoY", "pct"],
  ocf_growth_yoy: ["Operating cash flow growth YoY", "pct"],
  gross_margin_change_yoy: ["Gross margin change YoY", "pct"],
  operating_margin_change_yoy: ["Operating margin change YoY", "pct"],
  current_ratio: ["Current ratio", "x"],
  liabilities_to_assets: ["Liabilities / assets", "pct"],
  cash_to_assets: ["Cash / assets", "pct"],
  long_term_debt_to_assets: ["Long-term debt / assets", "pct"],
  inventory_to_assets: ["Inventory / assets", "pct"],
  receivables_to_revenue: ["Receivables / revenue", "pct"],
  return_on_assets: ["Return on assets (quarter)", "pct"],
  log_market_cap: ["Market cap", "log"],
  sales_yield: ["Sales / market cap (TTM)", "pct"],
  earnings_yield: ["Earnings yield (TTM)", "pct"],
  fcf_yield: ["Free cash flow yield (TTM)", "pct"],
  book_to_market: ["Book / market", "x"],
  revenue_growth_accel: ["Revenue growth acceleration", "pct"],
  revenue_growth_vs_trend: ["Revenue growth vs 4-qtr trend", "pct"],
  gross_margin_vs_trend: ["Gross margin vs 4-qtr trend", "pct"],
  operating_margin_vs_trend: ["Operating margin vs 4-qtr trend", "pct"],
  revenue_growth_volatility_8q: ["Revenue growth volatility (8 qtrs)", "pct"],
  positive_growth_share_8q: ["Quarters with YoY growth (of 8)", "pct"],
  operating_margin_slope_8q: ["Operating margin trend per qtr (8 qtrs)", "pct"],
  revenue_cagr_2y: ["Revenue CAGR, 2 years (TTM)", "pct"],
  accruals_ttm: ["Accruals: (net income − cash flow) / assets", "pct"],
  report_lag_days: ["Days from quarter end to release", "days"],
  pre_excess_return_21d: ["Return vs market, 21 days before", "pctpts"],
  pre_excess_return_63d: ["Return vs market, 63 days before", "pctpts"],
  pre_excess_return_252d: ["Return vs market, 12 months before", "pctpts"],
  pre_volatility_21d: ["Daily volatility, 21 days before", "pctpts"],
  day1_excess_return: ["First-day reaction vs market", "pctpts"],
  pre_return_21d: ["Share price return, 21 days before", "pctpts"],
  pre_return_63d: ["Share price return, 63 days before", "pctpts"],
};

function fmtFeature(value, kind) {
  if (value === null || value === undefined) return "—";
  switch (kind) {
    case "pct": return `${(value * 100).toFixed(1)}%`;
    case "pctpts": return `${value.toFixed(1)}%`;
    case "log": return fmtUsd(10 ** value);
    case "days": return `${Math.round(value)} d`;
    default: return `${value.toFixed(2)}×`;
  }
}

function fmtUsd(value) {
  if (value === null || value === undefined) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

function fmtCount(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return value.toLocaleString();
}

function sectorName(s) {
  return String(s || "").replace(/_/g, " ");
}

function renderBreakdown(source, inputs) {
  if (!source && !inputs) return "";
  let html = `<div class="breakdown"><h3>What the model saw</h3>`;

  if (source) {
    const filing = source.filing_url
      ? `<a href="${esc(source.filing_url)}" target="_blank" rel="noopener">${esc(source.form)} &middot; ${esc(source.accession_number)}</a>`
      : esc(source.form || "—");
    html += `
      <dl class="filing-facts">
        ${source.company ? `<dt>Company</dt><dd>${esc(source.company)}</dd>` : ""}
        ${source.sector ? `<dt>Sector</dt><dd>${esc(sectorName(source.sector))} <span class="field-hint">(SIC ${esc(source.sic)} &middot; ${esc(source.sic_description)})</span></dd>` : ""}
        <dt>Quarter ended</dt><dd class="mono">${esc(source.period_end)}</dd>
        <dt>Earnings released</dt><dd class="mono">${esc(source.announced_at)} <span class="field-hint">(${esc(source.announced_via)})</span></dd>
        <dt>Financials filed</dt><dd class="mono">${esc(source.filed_at || source.filed_date)}</dd>
        <dt>Filing</dt><dd class="mono">${filing}</dd>
        ${source.market_cap ? `<dt>Market cap</dt><dd class="mono">${fmtUsd(source.market_cap)} <span class="field-hint">(close ${esc(source.baseline_price_date)})</span></dd>` : ""}
        ${source.baseline_price_date ? `<dt>Price basis</dt><dd class="mono">${esc(source.baseline_price_date)} close &rarr; ${esc(source.day1_date || "not closed yet")}</dd>` : ""}
      </dl>`;

    if (source.reported && source.reported.length) {
      const rows = source.reported.map((r) => {
        const fmt = r.kind === "shares" ? fmtCount : fmtUsd;
        return `
        <tr>
          <td>${esc(r.label)}${r.kind === "instant" ? ' <span class="field-hint">(balance sheet)</span>' : ""}</td>
          <td class="mono num">${fmt(r.value)}</td>
          <td class="mono num">${fmt(r.prior_year_value)}</td>
        </tr>`;
      }).join("");
      html += `
        <h4>Figures reported in XBRL</h4>
        <p class="field-hint">Quarterly values from the original filing (restatements ignored). Year-ago column is the quarter ended ${esc(source.prior_year_period_end || "—")}.</p>
        <div class="table-scroll"><table class="data-table">
          <thead><tr><th>Line item</th><th class="num">This quarter</th><th class="num">Year ago</th></tr></thead>
          <tbody>${rows}</tbody>
        </table></div>`;
    }
  }

  if (inputs && inputs.length) {
    const allHorizons = HORIZON_ORDER.filter((h) => inputs.some((f) => (f.used_by || []).includes(h)));
    const hasPeers = inputs.some((f) => f.peer_percentile !== null && f.peer_percentile !== undefined);
    const rows = inputs.map((f) => {
      const [label, kind] = FEATURE_INFO[f.feature] || [f.feature, "x"];
      const z = f.z_score;
      const usedBy = (f.used_by || []).length === allHorizons.length ? "all" : (f.used_by || []).join(" ");
      const tags = (f.imputed ? `<span class="imputed-tag">missing</span> ` : "")
        + (f.clipped ? `<span class="imputed-tag">capped</span> ` : "");
      return `
        <tr class="${f.imputed ? "is-imputed" : ""}">
          <td>${esc(label)}</td>
          <td class="mono num">${tags}${fmtFeature(f.imputed ? f.training_median : f.value, kind)}</td>
          <td class="mono num muted">${fmtFeature(f.training_median, kind)}</td>
          ${hasPeers ? `<td class="mono num">${f.peer_percentile === null || f.peer_percentile === undefined ? "" : Math.round(f.peer_percentile * 100)}</td>` : ""}
          <td class="mono num ${Math.abs(z) >= 2 ? "z-unusual" : ""}">${signed(z)}</td>
          <td class="mono muted">${esc(usedBy)}</td>
        </tr>`;
    }).join("");
    html += `
      <h4>Model inputs</h4>
      <p class="field-hint">Built from the figures above plus the price history around the release. Missing values were filled with the training median; values beyond the training data's 1st&ndash;99th percentile were capped. Peer %ile ranks the value against companies reporting the same quarter. The z-score is how far the value sits from the training data, in standard deviations; 2 or more away is highlighted.</p>
      <div class="table-scroll"><table class="data-table">
        <thead><tr><th>Input</th><th class="num">Value</th><th class="num">Training median</th>${hasPeers ? '<th class="num">Peer %ile</th>' : ""}<th class="num">z-score</th><th>Used by</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
  }

  return html + `</div>`;
}

// ---------------------------------------------------------------- batches
const batchForm = el("batch-form");
if (batchForm) {
  batchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = el("batch-name").value.trim();
    const tickers = el("batch-tickers").value.trim();
    const statusBox = el("batch-status");
    const fill = el("batch-progress-fill");
    const msg = el("batch-message");
    const resultBox = el("batch-result");
    if (!name || !tickers) {
      statusBox.classList.remove("hidden");
      msg.textContent = "Give the batch a name and at least one ticker.";
      msg.classList.add("is-error");
      return;
    }

    statusBox.classList.remove("hidden");
    msg.classList.remove("is-error", "is-done");
    msg.textContent = "Starting...";
    fill.style.width = "2%";

    let jobId;
    try {
      jobId = (await postJSON("/api/batch", { name, model_id: el("batch-model").value, symbols: tickers })).job_id;
    } catch (err) {
      msg.textContent = err.message;
      msg.classList.add("is-error");
      return;
    }

    poll(`/api/batch/status/${encodeURIComponent(jobId)}`, 2000, (status) => {
      fill.style.width = `${Math.round((status.progress || 0) * 100)}%`;
      msg.textContent = status.message || "Working...";
      renderBatch(resultBox, { name, predictions: status.results || [], summary: status.summary, failures: status.failures });
      if (!status.done) return false;
      msg.classList.add(status.error ? "is-error" : "is-done");
      return true;
    }, () => {
      msg.textContent = "Lost contact with the server during the batch.";
      msg.classList.add("is-error");
    });
  });
}

document.querySelectorAll(".view-batch").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const box = el("batch-result");
    box.classList.remove("hidden");
    box.innerHTML = `<p class="field-hint">Loading batch...</p>`;
    try {
      const res = await fetch(`/api/batches/${encodeURIComponent(btn.dataset.batchId)}`);
      const b = await res.json();
      if (!res.ok) throw new Error(b.error || "Could not load batch");
      renderBatch(box, {
        name: b.name, model: b.models && b.models.name, created: b.created_at,
        predictions: b.predictions, summary: b.summary, failures: (b.summary || {}).failures,
      });
      box.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      box.innerHTML = `<p class="status-message is-error">${esc(err.message)}</p>`;
    }
  });
});

function renderBatch(box, b) {
  box.classList.remove("hidden");
  const preds = b.predictions || [];
  const rows = preds.map((p) => `
    <tr>
      <td class="mono">${esc(p.symbol)}</td>
      <td class="mono">${esc(p.earnings_date || "")}</td>
      ${HORIZON_ORDER.map((h) => {
        const d = (p.decisions || {})[h];
        if (!d) return `<td class="mono">—</td>`;
        const call = `<span class="c-${d.action.toLowerCase()}">${d.action}</span> ${signed(d.predicted_pct_change, 1)}%`;
        if (!isClosed(d)) return `<td class="mono">${call}</td>`;
        const actual = actualText(d);
        return `<td class="mono is-closed" title="Window closed ${esc(d.window_end || "")}">${call}${actual ? `<div class="metric-sub">${actual}</div>` : ""}</td>`;
      }).join("")}
    </tr>`).join("");

  let summary = "";
  if (b.summary && b.summary.counts) {
    summary = `<div class="batch-summary">` + HORIZON_ORDER.filter((h) => b.summary.counts[h]).map((h) => {
      const c = b.summary.counts[h];
      const open = c.BUY + c.SELL + c.HOLD;
      if (!open && c.CLOSED) {
        return `<div class="metric-card is-closed"><div class="metric-label">${h}</div>
          <div class="metric-value">closed</div>
          <div class="metric-sub">window had passed for all ${c.CLOSED}</div></div>`;
      }
      const avg = b.summary.mean_predicted_pct[h];
      return `<div class="metric-card"><div class="metric-label">${h}</div>
        <div class="metric-value"><span class="c-buy">${c.BUY}</span> / <span class="c-sell">${c.SELL}</span> / <span class="c-hold">${c.HOLD}</span></div>
        <div class="metric-sub">buy / sell / hold${avg === undefined ? "" : ` &middot; avg ${signed(avg, 2)}%`}${c.CLOSED ? ` &middot; ${c.CLOSED} closed` : ""}</div></div>`;
    }).join("") + `</div>`;
  }

  const failures = (b.failures || []).length
    ? `<p class="field-hint">Failed: ${(b.failures || []).map((f) => `<span class="mono">${esc(f.symbol)}</span> (${esc(f.error)})`).join("; ")}</p>`
    : "";
  const meta = [b.model && `model ${esc(b.model)}`, b.created && esc(b.created.slice(0, 16).replace("T", " "))].filter(Boolean).join(" &middot; ");

  box.innerHTML = `
    <h3>${esc(b.name)}${meta ? ` <span class="muted">&middot; ${meta}</span>` : ""}</h3>
    ${summary}
    ${preds.length ? `<div class="table-scroll"><table class="data-table">
      <thead><tr><th>Ticker</th><th>Report</th>${HORIZON_ORDER.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody></table></div>` : ""}
    ${failures}`;
}

async function refreshModelsDropdown() {
  try {
    const res = await fetch("/api/models");
    const models = await res.json();
    if (!Array.isArray(models)) return;
    const options = models.map((m) => {
      const version = m.target_scale && m.target_scale.version ? `v${m.target_scale.version}` : "v1";
      const symbols = (m.symbols || []).length > 6 ? `${m.symbols.length} tickers` : (m.symbols || []).join(", ");
      return `<option value="${esc(m.id)}">${esc(m.name)} · ${version} · ${esc(symbols)}</option>`;
    }).join("");
    document.querySelectorAll(".model-select").forEach((s) => { s.innerHTML = options; });
    document.querySelectorAll("#predict-form button[type=submit]").forEach((b) => (b.disabled = false));
  } catch (err) {
    // Non-fatal: the dropdowns just won't refresh until the next page load.
  }
}
