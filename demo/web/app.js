const COLORS = {
  reference: "var(--series-reference)",
  upwind: "var(--series-upwind)",
  weno5: "var(--series-weno5)",
  nn: "var(--series-nn)",
};
const LABELS = { reference: "Reference (fine WENO5)", upwind: "Upwind", weno5: "WENO5", nn: "ConservativeFluxNet" };

const SVG_NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};

let scenarios = [];
let paretoData = {};
let current = null; // current scenario object
let stepIndex = 0;
let playing = false;
let playTimer = null;

async function main() {
  const [scenarioPayload, paretoPayload] = await Promise.all([
    fetch("data/scenarios.json").then((r) => r.json()),
    fetch("data/pareto.json").then((r) => r.json()).catch(() => ({})),
  ]);
  scenarios = scenarioPayload.scenarios;
  paretoData = paretoPayload;

  const select = document.getElementById("scenario-select");
  scenarios.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = s.label;
    select.appendChild(opt);
  });
  select.addEventListener("change", () => selectScenario(Number(select.value)));

  document.getElementById("time-slider").addEventListener("input", (e) => {
    stepIndex = Number(e.target.value);
    render();
  });
  document.getElementById("play-button").addEventListener("click", togglePlay);

  selectScenario(0);
}

function selectScenario(index) {
  stop();
  current = scenarios[index];
  stepIndex = 0;
  const slider = document.getElementById("time-slider");
  slider.max = current.step_times.length - 1;
  slider.value = 0;
  renderErrorTable();
  renderParetoChart();
  render();
}

function togglePlay() {
  playing ? stop() : play();
}

function play() {
  playing = true;
  document.getElementById("play-button").textContent = "⏸ Pause";
  playTimer = setInterval(() => {
    stepIndex = (stepIndex + 1) % current.step_times.length;
    document.getElementById("time-slider").value = stepIndex;
    render();
  }, 180);
}

function stop() {
  playing = false;
  document.getElementById("play-button").textContent = "▶ Play";
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
}

function render() {
  document.getElementById("time-readout").textContent = `t = ${current.step_times[stepIndex].toFixed(2)}`;
  renderSolutionChart();
}

function linearScale(value, domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMax === domainMin) return (rangeMin + rangeMax) / 2;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function renderSolutionChart() {
  const svg = document.getElementById("solution-chart");
  svg.innerHTML = "";
  const [W, H, M] = [820, 420, { left: 44, right: 20, top: 20, bottom: 34 }];
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const x = current.x_coarse;
  const seriesKeys = ["reference", "upwind", "weno5", "nn"];
  let yMin = Infinity, yMax = -Infinity;
  for (const key of seriesKeys) {
    for (const step of current.series[key]) {
      for (const v of step) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    }
  }
  const pad = (yMax - yMin) * 0.1 || 1;
  yMin -= pad; yMax += pad;
  const xMin = Math.min(...x), xMax = Math.max(...x);

  const sx = (v) => M.left + linearScale(v, xMin, xMax, 0, plotW);
  const sy = (v) => M.top + linearScale(v, yMax, yMin, 0, plotH); // inverted (svg y grows downward)

  // gridlines
  for (let i = 0; i <= 4; i++) {
    const gy = M.top + (plotH * i) / 4;
    svg.appendChild(el("line", { x1: M.left, x2: M.left + plotW, y1: gy, y2: gy, class: "grid-line" }));
    const val = yMax - ((yMax - yMin) * i) / 4;
    const label = el("text", { x: M.left - 8, y: gy + 3, class: "axis-label", "text-anchor": "end" });
    label.textContent = val.toFixed(2);
    svg.appendChild(label);
  }
  svg.appendChild(el("line", { x1: M.left, x2: M.left, y1: M.top, y2: M.top + plotH, class: "axis-line" }));
  svg.appendChild(el("line", { x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH, class: "axis-line" }));

  const title = el("text", { x: M.left, y: 14, class: "chart-title" });
  title.textContent = `${current.label} — u(x)`;
  svg.appendChild(title);

  for (const key of seriesKeys) {
    const values = current.series[key][stepIndex];
    const points = x.map((xi, i) => `${sx(xi)},${sy(values[i])}`).join(" ");
    const color = COLORS[key];
    svg.appendChild(el("polyline", { points, fill: "none", stroke: color, "stroke-width": 2 }));
    if (key === "nn") {
      x.forEach((xi, i) => {
        if (i % 4 !== 0) return;
        svg.appendChild(el("circle", { cx: sx(xi), cy: sy(values[i]), r: 4, fill: color }));
      });
    }
  }

  const legend = document.getElementById("solution-legend");
  legend.innerHTML = "";
  for (const key of seriesKeys) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(--series-${key})`;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(LABELS[key]));
    legend.appendChild(item);
  }
}

function renderErrorTable() {
  const tbody = document.getElementById("error-table");
  tbody.innerHTML = "";
  for (const key of ["nn", "weno5", "upwind"]) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = LABELS[key];
    const value = document.createElement("td");
    value.className = "value";
    value.textContent = `${(current.final_errors[key] * 100).toFixed(2)}%`;
    row.appendChild(name);
    row.appendChild(value);
    tbody.appendChild(row);
  }
}

function renderParetoChart() {
  const svg = document.getElementById("pareto-chart");
  svg.innerHTML = "";
  const data = paretoData[current.pde];
  if (!data) return;

  const [W, H, M] = [820, 380, { left: 60, right: 130, top: 30, bottom: 40 }];
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const allTimes = [data.nn_point.time, ...Object.values(data.classical_sweep).flatMap((s) => s.times)];
  const allErrors = [data.nn_point.error, ...Object.values(data.classical_sweep).flatMap((s) => s.errors)];
  const logMinT = Math.log10(Math.min(...allTimes)), logMaxT = Math.log10(Math.max(...allTimes));
  const logMinE = Math.log10(Math.min(...allErrors)), logMaxE = Math.log10(Math.max(...allErrors));

  const sx = (t) => M.left + linearScale(Math.log10(t), logMinT, logMaxT, 0, plotW);
  const sy = (e) => M.top + linearScale(Math.log10(e), logMaxE, logMinE, 0, plotH);

  for (let i = 0; i <= 4; i++) {
    const gy = M.top + (plotH * i) / 4;
    svg.appendChild(el("line", { x1: M.left, x2: M.left + plotW, y1: gy, y2: gy, class: "grid-line" }));
  }
  svg.appendChild(el("line", { x1: M.left, x2: M.left, y1: M.top, y2: M.top + plotH, class: "axis-line" }));
  svg.appendChild(el("line", { x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH, class: "axis-line" }));

  const title = el("text", { x: M.left, y: 16, class: "chart-title" });
  title.textContent = `Accuracy vs. cost — ${current.pde}`;
  svg.appendChild(title);
  const xlabel = el("text", { x: M.left + plotW / 2, y: H - 6, class: "axis-label", "text-anchor": "middle" });
  xlabel.textContent = "Wall-clock time per rollout (s, log scale)";
  svg.appendChild(xlabel);
  const ylabel = el("text", {
    x: 14, y: M.top + plotH / 2, class: "axis-label", "text-anchor": "middle",
    transform: `rotate(-90 14 ${M.top + plotH / 2})`,
  });
  ylabel.textContent = "Relative L2 error (log scale)";
  svg.appendChild(ylabel);

  const schemeColors = { upwind: "var(--series-upwind)", muscl: "#eb6834", weno5: "var(--series-weno5)" };
  for (const [scheme, series] of Object.entries(data.classical_sweep)) {
    const order = series.times.map((_, i) => i).sort((a, b) => series.times[a] - series.times[b]);
    const points = order.map((i) => `${sx(series.times[i])},${sy(series.errors[i])}`).join(" ");
    const color = schemeColors[scheme] || "#898781";
    svg.appendChild(el("polyline", { points, fill: "none", stroke: color, "stroke-width": 2 }));
    order.forEach((i) => svg.appendChild(el("circle", { cx: sx(series.times[i]), cy: sy(series.errors[i]), r: 4, fill: color })));
    const lastI = order[order.length - 1];
    const label = el("text", {
      x: sx(series.times[lastI]) + 8, y: sy(series.errors[lastI]) + 3, class: "axis-label",
    });
    label.textContent = scheme.toUpperCase();
    svg.appendChild(label);
  }

  svg.appendChild(el("path", {
    d: starPath(sx(data.nn_point.time), sy(data.nn_point.error), 10),
    fill: "var(--series-nn)",
  }));
  const nnLabel = el("text", { x: sx(data.nn_point.time) + 10, y: sy(data.nn_point.error) - 8, class: "axis-label" });
  nnLabel.setAttribute("fill", "var(--ink-primary)");
  nnLabel.style.fontWeight = "bold";
  nnLabel.textContent = "ConservativeFluxNet";
  svg.appendChild(nnLabel);
}

function starPath(cx, cy, r) {
  const points = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? r : r * 0.45;
    const angle = (Math.PI / 5) * i - Math.PI / 2;
    points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return `M${points.join("L")}Z`;
}

main();
