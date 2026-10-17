// Sepsis Digital Twin dashboard. Plain JS + SVG; all data text goes in via textContent.
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const state = { clock: 24, selected: null, playing: null, meta: null, wTok: 0, pTok: 0, cohort: "physionet_b" };
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else e.setAttribute(k, v);
  }
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = String(text);
  return e;
}
const pct = (p, nd = 1) => (p === null || p === undefined ? "–" : (100 * p).toFixed(nd) + "%");
const fmt = (v, nd = 0) => (v === null || v === undefined ? "–" : Number(v).toFixed(nd));
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}
function statusChip(alert, inIcu = true) {
  if (!inIcu) return el("span", { class: "chip out" }, "Left ICU");
  const c = el("span", { class: "chip " + (alert ? "alert" : "ok") });
  c.append(el("span", { class: alert ? "tri" : "dot", "aria-hidden": "true" }));
  c.append(document.createTextNode(alert ? "Alert" : "Below threshold"));
  return c;
}

// ------------------------------------------------------------------ tooltip
const tip = $("tooltip");
function showTip(evt, head, rows) {
  tip.replaceChildren(el("div", { class: "t-head" }, head));
  for (const r of rows) {
    const row = el("div", { class: "t-row" });
    if (r.color) { const k = el("span", { class: "t-key" }); k.style.background = r.color; row.append(k); }
    row.append(el("b", {}, r.value), el("span", {}, r.label));
    tip.append(row);
  }
  tip.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

// ------------------------------------------------------------------ charts
// Line chart over ICU hours with crosshair. series: [{label, color, values}]
function lineChart({ series, height = 220, yFmt, yMin, yMax, threshold, alertMask, onset, dots = false, unit = "", compact = false,
                     xFmt = (i) => `h${i}`, tipHead = (i) => `ICU hour ${i}`, width }) {
  // compact: small multiples drawn at a narrow viewBox so their text stays legible when scaled down
  // compact: tiny multiples; width: a mid-size chart for half-width columns
  const W = width ?? (compact ? 300 : 760), H = height;
  const m = compact ? { l: 34, r: 10, t: 8, b: 18 } : width ? { l: 40, r: 48, t: 10, b: 22 } : { l: 48, r: 92, t: 12, b: 26 };
  const n = Math.max(...series.map((s) => s.values.length));
  const all = series.flatMap((s) => [...s.values, ...(s.low || []), ...(s.high || [])]).filter((v) => v !== null && v !== undefined);
  let lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
  if (threshold !== undefined) hi = Math.max(hi, threshold * 1.4);
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi === lo) { hi = lo + 1; }
  const pad = yMin === undefined ? (hi - lo) * 0.12 : 0;
  lo -= pad; hi += (yMax === undefined ? (hi - lo) * 0.08 : 0);
  const X = (i) => m.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
  const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" });

  // alert bands behind everything
  if (alertMask) {
    alertMask.forEach((a, i) => {
      if (!a) return;
      const x0 = X(Math.max(0, i - 0.5)), x1 = X(Math.min(n - 1, i + 0.5));
      root.append(svg("rect", { x: x0, y: m.t, width: Math.max(1, x1 - x0), height: H - m.t - m.b, fill: "var(--critical-wash)" }));
    });
  }
  for (const t of niceTicks(lo, hi, compact ? 2 : 4)) {
    root.append(svg("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    root.append(svg("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, yFmt(t)));
  }
  root.append(svg("line", { class: "base", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  for (const t of niceTicks(0, n - 1, compact ? 3 : 6).filter((t) => Number.isInteger(t))) {
    root.append(svg("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, xFmt(t)));
  }
  if (threshold !== undefined) {
    root.append(svg("line", { class: "thr", x1: m.l, x2: W - m.r, y1: Y(threshold), y2: Y(threshold) }));
    root.append(svg("text", { class: "thr-label", x: W - m.r + 6, y: Y(threshold) + 4 }, "Alert threshold"));
  }
  if (onset !== undefined && onset !== null) {
    const inView = onset <= n - 1;
    const ox = inView ? X(onset) : W - m.r;
    root.append(svg("line", { class: "onset", x1: ox, x2: ox, y1: m.t, y2: H - m.b }));
    root.append(svg("text", { class: "thr-label", x: ox - 4, y: m.t + 10, "text-anchor": "end" },
      inView ? `Recorded onset h${onset}` : `Onset at h${onset} (future)`));
  }
  // uncertainty bands (series hue at ~12% opacity) under the lines
  for (const s of series) {
    if (!s.low || !s.high) continue;
    let d = "";
    s.low.forEach((v, i) => { d += (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1) + " "; });
    for (let i = s.high.length - 1; i >= 0; i--) d += "L" + X(i).toFixed(1) + " " + Y(s.high[i]).toFixed(1) + " ";
    root.append(svg("path", { d: d + "Z", fill: s.color, "fill-opacity": 0.12, stroke: "none" }));
  }
  const ends = [];
  for (const s of series) {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (v === null) { pen = false; return; }
      d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1) + " "; pen = true;
    });
    root.append(svg("path", { class: "line", d, stroke: s.color }));
    if (dots) {
      s.values.forEach((v, i) => {
        if (v !== null && s.measured?.[i]) root.append(svg("circle", { cx: X(i), cy: Y(v), r: 3, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }));
      });
    }
    const li = s.values.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0).pop();
    if (li !== undefined && li >= 0) {
      root.append(svg("circle", { cx: X(li), cy: Y(s.values[li]), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }));
      ends.push({ y: Y(s.values[li]), text: s.endLabel ?? yFmt(s.values[li]) });
    }
  }
  // end labels; spread apart when they collide
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  for (const e of ends) root.append(svg("text", { class: "end-label", x: W - m.r + 8, y: e.y + 4 }, e.text));

  // crosshair + tooltip
  const cross = svg("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 });
  const at = (evt, i) => {
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    showTip(evt, tipHead(i), series.map((s) => ({
      color: s.color, label: s.label,
      value: s.values[i] === null ? "not measured" : yFmt(s.values[i]) + unit +
        (s.low ? ` (${yFmt(s.low[i])}–${yFmt(s.high[i])})` : "") })));
  };
  hit.addEventListener("pointermove", (evt) => {
    const r = root.getBoundingClientRect();
    const x = ((evt.clientX - r.left) / r.width) * W;
    at(evt, Math.max(0, Math.min(n - 1, Math.round(((x - m.l) / (W - m.l - m.r)) * (n - 1)))));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("focus", () => {
    const r = root.getBoundingClientRect();
    at({ clientX: r.right - 120, clientY: r.top }, n - 1);
  });
  hit.addEventListener("blur", hideTip);
  root.append(hit);
  return root;
}

function sparkline(values, threshold) {
  const W = 96, H = 24;
  const vals = values.filter((v) => v !== null);
  const hi = Math.max(threshold * 1.4, ...vals), lo = 0;
  const X = (i) => (values.length <= 1 ? W : (i / (values.length - 1)) * (W - 4) + 2);
  const Y = (v) => 2 + (1 - (v - lo) / (hi - lo)) * (H - 4);
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, "aria-hidden": "true" });
  s.append(svg("line", { class: "thr", x1: 0, x2: W, y1: Y(threshold), y2: Y(threshold) }));
  let d = "";
  values.forEach((v, i) => { if (v !== null) d += (d ? "L" : "M") + X(i) + " " + Y(v) + " "; });
  s.append(svg("path", { class: "line", d, stroke: "var(--series-1)", "stroke-width": 1.5 }));
  return s;
}

// ------------------------------------------------------------------ ward list
async function loadWard() {
  const tok = ++state.wTok;
  const rows = await api(`/api/ward?clock=${state.clock}&cohort=${state.cohort}`);
  if (tok !== state.wTok) return;            // a newer request is in flight
  const onlyAlerts = $("only-alerts").checked;
  const body = $("ward-body");
  body.replaceChildren();
  const inIcu = rows.filter((r) => r.in_icu);
  $("ward-count").textContent = `${inIcu.filter((r) => r.alert).length} alerts · ${inIcu.length} in ICU`;
  for (const r of rows) {
    if (onlyAlerts && !(r.alert && r.in_icu)) continue;
    const tr = el("tr", { tabindex: 0, class: (r.stay_id === state.selected ? "selected " : "") + (r.in_icu ? "" : "discharged") });
    const who = el("td");
    who.append(el("div", { class: "pid" }, `#${r.stay_id}`), el("div", { class: "pmeta" }, `${r.age} ${r.sex} · ${r.unit}`));
    const risk = el("td", { class: "num" });
    risk.append(el("div", {}, pct(r.risk_6h)), statusChip(r.alert, r.in_icu));
    const tr2 = el("td"); tr2.append(sparkline(r.trend, state.meta.threshold["6h"]));
    tr.append(who, el("td", { class: "num" }, r.in_icu ? `h${r.hour}` : "–"), risk, tr2);
    const open = () => { state.selected = r.stay_id; loadWard(); loadPatient(); };
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    body.append(tr);
  }
  if (state.selected === null && rows.length) { state.selected = rows[0].stay_id; loadPatient(); loadWard(); }
}

// ------------------------------------------------------------------ patient view
function card(title, note) {
  const c = el("section", { class: "card" });
  const head = el("div", { class: "card-head" });
  head.append(el("h2", {}, title));
  c.append(head);
  if (note) c.append(el("p", { class: "note" }, note));
  return [c, head];
}
function tile(label, value, extra, hero = false) {
  const t = el("div", { class: "tile" + (hero ? " hero" : "") });
  t.append(el("div", { class: "label" }, label), el("div", { class: "value" }, value));
  if (extra) t.append(extra);
  return t;
}

async function loadPatient() {
  if (state.selected === null) return;
  const showOutcome = $("show-outcome").checked;
  const tok = ++state.pTok;
  const p = await api(`/api/patients/${state.selected}?clock=${state.clock}&show_outcome=${showOutcome}&cohort=${state.cohort}`);
  if (tok !== state.pTok) return;
  const thr = p.threshold;
  const r6 = p.risk["6h"], r12 = p.risk["12h"];
  const now6 = r6[r6.length - 1], now12 = r12[r12.length - 1];
  const d = $("detail");
  d.replaceChildren();

  // header
  const [hc] = card(`Patient #${p.stay_id}`);
  const meta = el("div", { class: "patient-head" });
  meta.append(el("span", { class: "meta" },
    `${p.age} y · ${p.sex === "F" ? "female" : "male"} · ${p.unit} · ICU hour ${p.hour}` +
    (p.hospital_admission_h_before_icu ? ` · in hospital ${fmt(p.hospital_admission_h_before_icu)} h before ICU` : "")),
    statusChip(p.alert, p.in_icu));
  hc.append(meta);
  if (!p.in_icu) hc.append(el("p", { class: "note" }, `Record ends at ICU hour ${p.last_hour}; showing the last available hour.`));
  d.append(hc);

  // tiles
  const tiles = el("div", { class: "tiles" });
  const prev6 = r6.length > 6 ? r6[r6.length - 7] : null;
  const delta = prev6 === null ? "No earlier value yet" :
    `${now6 >= prev6 ? "Up" : "Down"} ${pct(Math.abs(now6 - prev6))} vs 6 h ago`;
  tiles.append(
    tile("Risk of sepsis within 6 h", pct(now6), el("div", { class: "delta" }, delta), true),
    tile("Risk within 12 h", pct(now12), el("div", { class: "delta" }, `Alert threshold ${pct(thr["12h"])}`)),
    tile("Vital signs measured", `${fmt(p.data_quality.vitals_measured_pct)}%`, el("div", { class: "delta" }, "of hours so far")),
    tile("Lab results so far", fmt(p.data_quality.labs_drawn), el("div", { class: "delta" }, "values drawn")));
  d.append(tiles);

  // risk trajectory
  const [rc, rh] = card("Sepsis risk over this stay",
    "Each point uses only data available at that hour. Shaded hours were above the alert threshold.");
  const lg = el("div", { class: "legend" });
  for (const [label, color] of [["Within 6 h", "var(--series-1)"], ["Within 12 h", "var(--series-2)"]]) {
    const k = el("span"); const key = el("span", { class: "key" }); key.style.background = color;
    k.append(key, document.createTextNode(label)); lg.append(k);
  }
  const band = el("span"); band.append(el("span", { class: "band" }), document.createTextNode("Alert (6 h)")); lg.append(band);
  rh.append(lg);
  rc.append(lineChart({
    series: [{ label: "within 6 h", color: "var(--series-1)", values: r6, endLabel: `6 h ${pct(now6)}` },
             { label: "within 12 h", color: "var(--series-2)", values: r12, endLabel: `12 h ${pct(now12)}` }],
    yFmt: (v) => pct(v, v < 0.1 ? 1 : 0), yMin: 0, threshold: thr["6h"],
    alertMask: r6.map((v) => v !== null && v >= thr["6h"]), onset: p.recorded_onset_hour,
  }));
  d.append(rc);

  // why + what-if
  const g2 = el("div", { class: "grid-2" });
  const [wc] = card("What the model is weighing now",
    "Share of the model's attention given to each data stream at this hour. It shows where the estimate comes from; it is not a causal explanation.");
  const bars = el("div", { class: "bars" });
  for (const s of p.streams) {
    const row = el("div", { class: "row" });
    const track = el("div", { class: "track" }); const fill = el("div", { class: "fill" });
    fill.style.width = `${Math.round(100 * s.weight)}%`; track.append(fill);
    row.append(el("span", {}, s.name), track, el("span", { class: "num" }, `${Math.round(100 * s.weight)}%`));
    bars.append(row);
  }
  wc.append(bars);
  const wi = await api(`/api/patients/${p.stay_id}/whatif?clock=${state.clock}&cohort=${state.cohort}`);
  if (tok !== state.pTok) return;
  if (wi.available) {
    d.append(whatIfCard(p, wi));
    g2.append(wc, treatmentsCard(p));
    d.append(g2);
  } else {
    const [xc, xh] = card("What-if simulator");
    xc.classList.add("disabled-card");
    xh.querySelector("h2").prepend(el("span", { class: "lock", "aria-hidden": "true" }, "×"));
    xc.append(el("p", { class: "note" }, wi.reason));
    g2.append(wc, xc);
    d.append(g2);
  }

  // vitals small multiples
  const [vc] = card("Vital signs", "Dots are measured values; the line joins them. Gaps mean nothing was charted.");
  const vg = el("div", { class: "vitals" });
  for (const [, v] of Object.entries(p.vitals)) {
    const box = el("div", { class: "vital" });
    const last = v.values.filter((x) => x !== null).pop();
    const head = el("div"); head.append(el("div", { class: "vname" }, v.label));
    const val = el("div", { class: "vval" }, last === undefined ? "–" : fmt(last, v.unit === "degC" ? 1 : 0));
    val.append(el("span", { class: "vunit" }, v.unit === "degC" ? "°C" : v.unit));
    head.append(val); box.append(head);
    // carry values forward for the line, mark measured points with dots
    box.append(lineChart({
      series: [{ label: v.label, color: "var(--series-1)", values: v.values, measured: v.values.map((x) => x !== null), endLabel: "" }],
      yFmt: (t) => fmt(t, v.unit === "degC" ? 1 : 0), height: 110, dots: true, compact: true,
      unit: " " + (v.unit === "degC" ? "°C" : v.unit),
    }));
    vg.append(box);
  }
  vc.append(vg);
  d.append(vc);

  // labs + derived
  const g3 = el("div", { class: "grid-2" });
  const [lc] = card("Latest laboratory results", "Extreme = outside the typical range set in the quality gate; the value is real and kept.");
  const lt = el("table", { class: "data" });
  const lth = el("tr"); for (const h of ["Test", "Value", "Unit", "Drawn", "Flag"]) lth.append(el("th", {}, h));
  lt.append(lth);
  for (const l of p.labs) {
    const tr = el("tr");
    const flag = el("td");
    if (l.flag === "extreme") { const c = el("span", { class: "chip alert" }); c.append(el("span", { class: "tri" }), document.createTextNode("Extreme")); flag.append(c); }
    tr.append(el("td", { class: l.flag ? "flag-ext" : "" }, l.label), el("td", { class: "num" }, l.value === null ? "–" : String(l.value)),
      el("td", { class: "muted" }, l.unit), el("td", { class: "muted" }, l.hours_ago === 0 ? "this hour" : `${l.hours_ago} h ago`), flag);
    lt.append(tr);
  }
  if (!p.labs.length) lc.append(el("p", { class: "note" }, "No laboratory results yet."));
  else lc.append(lt);
  const [dc] = card("Derived scores", p.data_quality.missing_by_design + ", so some scores are partial.");
  const dt = el("table", { class: "data" });
  for (const x of p.derived) {
    const tr = el("tr");
    tr.append(el("td", {}, x.label), el("td", { class: "num" }, x.value === null ? "–" : String(x.value)),
      el("td", { class: "muted" }, x.unit), el("td", { class: "muted" }, x.hours_ago === 0 ? "now" : `${x.hours_ago} h ago`));
    dt.append(tr);
  }
  dc.append(dt);
  g3.append(lc, dc);
  d.append(g3);
}

// ------------------------------------------------------------------ treatments + what-if
function treatmentsCard(p) {
  const [c] = card("Treatments recorded", "Norepinephrine-equivalent dose and IV fluid volume per ICU hour, as charted.");
  const g = el("div", { class: "vitals two" });
  const ne = p.treatments.ne_dose, fl = p.treatments.fluid_ml;
  for (const [label, vals, unit, nd] of [["Vasopressor (NE-equivalent)", ne, "mcg/kg/min", 2], ["IV fluid", fl, "mL/h", 0]]) {
    const box = el("div", { class: "vital" });
    const last = vals[vals.length - 1];
    box.append(el("div", { class: "vname" }, label));
    const v = el("div", { class: "vval" }, fmt(last, nd)); v.append(el("span", { class: "vunit" }, unit)); box.append(v);
    box.append(lineChart({ series: [{ label, color: "var(--series-1)", values: vals, endLabel: "" }],
      yFmt: (t) => fmt(t, nd), yMin: 0, height: 110, compact: true, unit: " " + unit }));
    g.append(box);
  }
  c.append(g);
  return c;
}

function whatIfCard(p, setup) {
  const [c, head] = card("What-if simulator",
    "Compares a plan with usual care over the next 6 hours. Usual-care course and its 90% range come from " +
    "eICU and MIMIC-IV records; the plan's added effect comes from published studies (see model card).");
  head.querySelector("h2").append(el("span", { class: "chip ok", style: "margin-left:8px" }, "Not validated for clinical use"));
  const form = el("form", { class: "whatif-form" });
  const fluid = el("select", { id: "wi-fluid", "aria-label": "Fluid bolus in the next hour" });
  for (const v of setup.fluid_options) fluid.append(el("option", { value: v }, v === 0 ? "No bolus" : `${v} mL bolus`));
  const ne = el("input", { id: "wi-ne", type: "number", min: 0, max: setup.ne_max, step: "any", value: setup.ne_now ?? 0,
    "aria-label": "Norepinephrine-equivalent dose" });
  const l1 = el("label"); l1.append(el("span", {}, "Fluid in the next hour"), fluid);
  const l2 = el("label"); l2.append(el("span", {}, `Norepinephrine-equivalent dose (now ${fmt(setup.ne_now, 2)} mcg/kg/min)`), ne);
  const go = el("button", { type: "submit", class: "primary" }, "Simulate plan");
  form.append(l1, l2, go);
  const result = el("div", { class: "whatif-result" });
  result.append(el("p", { class: "note" }, "Choose a plan and select Simulate plan."));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    go.disabled = true; result.style.opacity = 0.5;
    try {
      const r = await api(`/api/patients/${p.stay_id}/whatif?clock=${state.clock}&cohort=${state.cohort}` +
                          `&fluid_ml=${fluid.value}&ne_dose=${ne.value || 0}`);
      renderWhatIf(result, r);
    } catch (err) {
      result.replaceChildren(el("p", { class: "note" }, "Simulation failed: " + err.message));
    } finally { go.disabled = false; result.style.opacity = 1; }
  });
  c.append(form, result);
  if (state.autoPlan) {                       // plan passed in the link: run it once
    fluid.value = state.autoPlan.fluid; if (state.autoPlan.ne !== null) ne.value = state.autoPlan.ne;
    state.autoPlan = null;
    setTimeout(() => form.requestSubmit(), 0);
  }
  return c;
}

function renderWhatIf(box, r) {
  box.replaceChildren();
  if (!r.answered) {
    const n = el("div", { class: "refusal" });
    const chip = el("span", { class: "chip alert" }); chip.append(el("span", { class: "tri" }), document.createTextNode("Not answered"));
    n.append(chip, el("p", {}, "The simulator will not extrapolate beyond what the data and published evidence cover:"));
    const ul = el("ul"); for (const x of r.reasons) ul.append(el("li", {}, x)); n.append(ul);
    box.append(n);
    return;
  }
  const t = r.trajectories;
  const m = t.map, hr = t.hr;
  const sum = el("p", { class: "whatif-summary" });
  sum.textContent = `MAP 1 h after the plan: ${fmt(m.plan[0])} mmHg (90% range ${fmt(m.plan_low[0])}–${fmt(m.plan_high[0])}) ` +
    `vs ${fmt(m.current_care[0])} with usual care. Added effect at 1 h: ${m.difference[0] >= 0 ? "+" : ""}${fmt(m.difference[0], 1)} mmHg ` +
    `(${fmt(m.difference_low[0], 1)} to ${fmt(m.difference_high[0], 1)}); at 6 h: ${m.difference[5] >= 0 ? "+" : ""}${fmt(m.difference[5], 1)} mmHg.`;
  box.append(sum);
  const lg = el("div", { class: "legend" });
  for (const [label, color] of [["Usual care", "var(--series-1)"], ["Plan", "var(--series-2)"]]) {
    const k = el("span"); const key = el("span", { class: "key" }); key.style.background = color;
    k.append(key, document.createTextNode(label)); lg.append(k);
  }
  lg.append(el("span", {}, "Shaded: 90% range"));
  box.append(lg);
  const g = el("div", { class: "grid-2 even" });
  for (const [name, tr, unit] of [["Mean arterial pressure", m, "mmHg"], ["Heart rate", hr, "bpm"]]) {
    const w = el("div"); w.append(el("h3", {}, name));
    w.append(lineChart({
      series: [{ label: "usual care", color: "var(--series-1)", values: tr.current_care, low: tr.current_low, high: tr.current_high, endLabel: "Usual" },
               { label: "plan", color: "var(--series-2)", values: tr.plan, low: tr.plan_low, high: tr.plan_high, endLabel: "Plan" }],
      yFmt: (v) => fmt(v), height: 240, width: 420, unit: " " + unit,
      xFmt: (i) => `+${i + 1} h`, tipHead: (i) => `${i + 1} h from now` }));
    g.append(w);
  }
  box.append(g);
  box.append(el("p", { class: "note" }, r.assumption));
}

// ------------------------------------------------------------------ model card
async function openCard() {
  const c = await api(`/api/model-card?cohort=${state.cohort}`);
  const b = $("card-body");
  b.replaceChildren(el("h2", {}, c.name), el("p", {}, c.intended_use), el("p", { class: "note" }, "Not for: " + c.not_for),
    el("h3", {}, "Training data"), el("p", {}, c.training_data), el("h3", {}, "Validation (6 h ahead)"));
  const t = el("table", { class: "data" });
  const hr = el("tr"); for (const h of ["Test set", "AUROC (95% CI)", "Sensitivity", "PPV", "Alert-hours / patient-day"]) hr.append(el("th", {}, h));
  t.append(hr);
  for (const v of c.validation) {
    const tr = el("tr");
    tr.append(el("td", {}, v.set), el("td", { class: "num" }, `${v.auroc} (${v.auroc_ci95[0]}–${v.auroc_ci95[1]})`),
      el("td", { class: "num" }, pct(v.sensitivity, 0)), el("td", { class: "num" }, pct(v.ppv, 1)), el("td", { class: "num" }, v.alerts_per_patient_day));
    t.append(tr);
  }
  b.append(t, el("p", { class: "note" }, c.operating_point), el("h3", {}, "Limitations"));
  const ul = el("ul"); for (const l of c.limitations) ul.append(el("li", {}, l));
  b.append(ul);
  if (c.whatif) {
    b.append(el("h3", {}, "What-if simulator"), el("p", {}, c.whatif.method));
    const et = el("table", { class: "data" });
    const eh = el("tr"); for (const h of ["Outcome", "Treatment", "Effect used", "Source"]) eh.append(el("th", {}, h));
    et.append(eh);
    for (const e of c.whatif.effects) {
      const tr = el("tr");
      const unit = e.treatment === "ne" ? "per +0.1 mcg/kg/min" : "per 500 mL (bolus hour)";
      tr.append(el("td", {}, e.outcome), el("td", {}, e.treatment === "ne" ? "Norepinephrine" : "Fluid bolus"),
        el("td", { class: "num" }, `${e.effect >= 0 ? "+" : ""}${e.effect} ± ${e.sd} ${unit}`), el("td", { class: "muted" }, e.source));
      et.append(tr);
    }
    b.append(et);
    const v = c.whatif.validation.map_treatment_changed, v0 = c.whatif.validation.map_no_change;
    if (v && v0) b.append(el("p", { class: "note" },
      `Usual-care MAP forecast on held-out patients: mean error ${v0.mae_baseline_only} mmHg (carry-forward ${v0.mae_carry_forward}); ` +
      `90% range covered ${Math.round(100 * v0.coverage_90)}% of outcomes. The added plan effects come from published studies and ` +
      `cannot be checked against these records, because clinicians titrate vasopressors to a blood-pressure target.`));
  }
  $("card-dialog").showModal();
}

// ------------------------------------------------------------------ controls
function setClock(h) {
  state.clock = Math.max(0, Math.min(+$("clock").max, h));
  $("clock").value = state.clock;
  $("clock-out").textContent = `ICU hour ${state.clock}`;
  loadWard(); loadPatient();
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("sdt-theme"); } catch (e) { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("btn-theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("sdt-theme", document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
  });
}

async function loadCohort() {
  state.meta = await api(`/api/meta?cohort=${state.cohort}`);
  $("cohort-note").textContent = state.meta.note;
  $("clock").max = Math.min(state.meta.max_hour, 120);
  state.selected = null;
}

async function main() {
  initTheme();
  const cohorts = await api("/api/cohorts");
  for (const c of cohorts) $("cohort").append(el("option", { value: c.id }, c.label));
  $("cohort").addEventListener("change", async (e) => {
    state.cohort = e.target.value;
    $("detail").replaceChildren(el("div", { class: "empty" }, "Loading cohort…"));
    await loadCohort(); setClock(state.clock);
  });
  // a link can open a cohort, patient, hour and plan directly
  const qs = new URLSearchParams(location.search);
  if (qs.get("cohort")) { state.cohort = qs.get("cohort"); $("cohort").value = state.cohort; }
  await loadCohort();
  if (qs.get("patient")) state.selected = +qs.get("patient");
  if (qs.get("fluid") || qs.get("ne")) state.autoPlan = { fluid: qs.get("fluid") || "0", ne: qs.get("ne") };
  if (qs.get("clock")) state.clock = +qs.get("clock");
  $("clock").addEventListener("input", (e) => setClock(+e.target.value));
  $("btn-back").addEventListener("click", () => setClock(state.clock - 1));
  $("btn-fwd").addEventListener("click", () => setClock(state.clock + 1));
  $("btn-play").addEventListener("click", () => {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("btn-play").textContent = "Play"; return; }
    $("btn-play").textContent = "Pause";
    state.playing = setInterval(() => setClock(state.clock + 1), 1200);
  });
  $("only-alerts").addEventListener("change", loadWard);
  $("show-outcome").addEventListener("change", loadPatient);
  $("btn-card").addEventListener("click", openCard);
  setClock(state.clock);
}
main().catch((e) => {
  $("detail").replaceChildren(el("div", { class: "empty" }, "Could not load the twin: " + e.message));
});
