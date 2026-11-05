"""Interactive viewer of the grids written by pusht_cost_surface.py: one HTML page with the
grids inlined and plotly.js from cdnjs, so it opens straight from disk. With several JSON
files (e.g. the plain cost and its LSE smoothing) a toggle switches between them on one
color scale, keeping the camera.

Example:
    python make_surface_viewer.py   # plain + every pusht_cost_surface_lse*.json -> pusht_cost_surface.html
"""

import argparse
from pathlib import Path

here = Path(__file__).parent
parser = argparse.ArgumentParser(description="HTML viewer of the PushT cost surfaces")
parser.add_argument("--json", nargs="+",
                    default=[str(here / "pusht_cost_surface.json")]
                    + sorted(str(p) for p in here.glob("pusht_cost_surface_lse*.json")))
parser.add_argument("--out", default=str(here / "pusht_cost_surface.html"))
args = parser.parse_args()

PAGE = r"""<title>PushT Cost Surfaces</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: instrument panel — controls strip, 3-D view beside the top-down map, readouts below */
:root {
  --bg: #f3f5f4; --panel: #ffffff; --ink: #18211f; --muted: #5d6b67; --line: #d5dcd9;
  --accent: #1f7a6d; --accent-ink: #ffffff; --star: #f2c230;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --font-body: "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121816; --panel: #19211e; --ink: #e3ebe8; --muted: #94a5a0; --line: #2c3834;
  --accent: #4fc0ad; --accent-ink: #0d1412; --star: #f2c230; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121816; --panel: #19211e; --ink: #e3ebe8; --muted: #94a5a0; --line: #2c3834;
  --accent: #4fc0ad; --accent-ink: #0d1412; --star: #f2c230; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 28px 40px; }
.wrap { max-width: 1240px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
h1 { font-family: var(--font-display); font-weight: 600; font-size: 1.9rem; margin: 0; text-wrap: balance; }
.lede { color: var(--muted); max-width: 70ch; margin: 4px 0 0; }
.math { font-family: var(--font-mono); font-size: 0.92em; color: var(--ink); }
.controls { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: end; }
.group { display: flex; flex-direction: column; gap: 6px; }
.group > span { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); flex-wrap: wrap; }
.seg button { font: 500 0.9rem var(--font-body); color: var(--ink); background: transparent; border: 0; padding: 7px 14px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.views { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 14px; }
.view { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px 6px; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.view h2 { font-family: var(--font-display); font-weight: 500; font-size: 1.02rem; margin: 0; }
.view p { margin: 0; color: var(--muted); font-size: 0.85rem; }
.plot { width: 100%; height: 540px; }
.readouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.readout { border-top: 2px solid var(--line); padding-top: 6px; display: flex; flex-direction: column; }
.readout span { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.readout b { font-family: var(--font-mono); font-weight: 500; font-size: 1.25rem; font-variant-numeric: tabular-nums; }
.notes { color: var(--muted); font-size: 0.88rem; max-width: 80ch; display: flex; flex-direction: column; gap: 6px; }
.notes p { margin: 0; }
.swatch { display: inline-block; width: 0.8em; height: 0.8em; border-radius: 2px; vertical-align: -0.05em; margin-right: 4px; }
@media (max-width: 900px) { .views { grid-template-columns: minmax(0, 1fr); } .plot { height: 400px; } }
</style>

<div class="wrap">
  <header>
    <h1>PushT cost surfaces</h1>
    <p class="lede">The 12-D planning cost <span class="math">J(x)</span> (one 1 s rollout of 6 knots × 2 controls, <span class="math">x ∈ [−1, 1]¹²</span>) on random 2-D planes <span class="math">x = c + a·d1 + b·d2</span>. Drag the 3-D view to rotate it, scroll to zoom, and hover for the cost at a point.</p>
  </header>

  <div class="controls">
    <div class="group" id="costGroup"><span>Cost</span><div class="seg" id="costSeg" role="group" aria-label="Cost"></div></div>
    <div class="group"><span>Center c</span><div class="seg" id="centerSeg" role="group" aria-label="Center"></div></div>
    <div class="group"><span>Random plane</span><div class="seg" id="planeSeg" role="group" aria-label="Random plane"></div></div>
    <div class="group"><span>Height</span><div class="seg" id="heightSeg" role="group" aria-label="Height">
      <button id="hLin" aria-pressed="true">cost</button><button id="hLog" aria-pressed="false">log cost</button></div></div>
  </div>

  <div class="views">
    <section class="view"><h2>3-D surface</h2><p id="sub3d"></p><div class="plot" id="p3d"></div></section>
    <section class="view"><h2>Seen from above</h2><p>blue = low cost, red = high · the flat plateau is the pusher missing the block · blank = outside the box · <span style="color:var(--star)">★</span> center</p><div class="plot" id="p2d"></div></section>
  </div>

  <div class="readouts">
    <div class="readout"><span>center cost</span><b id="rCenter"></b></div>
    <div class="readout"><span>lowest cost on plane</span><b id="rMin"></b></div>
    <div class="readout"><span>on the plateau</span><b id="rPlat"></b></div>
    <div class="readout"><span>outside the box</span><b id="rOut"></b></div>
  </div>

  <div class="notes" id="notes"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<script>
const ALL = /*DATA*/null;
let D = ALL[0];
// ParaView's cool to warm (matplotlib coolwarm), the colors of Li et al. (2018) and SAM, Figure 1
const COOLWARM = ["#3b4cc0", "#5977e3", "#7b9ff9", "#9ebeff", "#c0d4f5", "#dddcdc", "#f2cbb7", "#f7ac8e", "#ee8468", "#d65244", "#b40426"];
const centers = [...new Set(D.planes.map(p => p.center))];
const state = { data: 0, center: 0, plane: 0, log: false };

// one color scale for every plane: log10 cost
const onPlateau = v => v !== null && Math.abs(v - D.plateau) < 6e-3;
let lo = Infinity, hi = -Infinity;
for (const d of ALL) for (const p of d.planes) for (const row of p.J) for (const v of row) if (v !== null) {
  const l = Math.log10(v); if (l < lo) lo = l; if (l > hi) hi = l; }
const cmin = lo, cmax = hi;
const scale = COOLWARM.map((c, i) => [i / (COOLWARM.length - 1), c]);
const ticks = [10, 15, 20, 30, 50, 80, 120, 200].filter(t => Math.log10(t) >= lo - 1e-9 && Math.log10(t) <= hi + 1e-9);

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function buttons(el, labels, key) {
  el.innerHTML = "";
  labels.forEach((label, i) => {
    const b = document.createElement("button");
    b.textContent = label; b.setAttribute("aria-pressed", String(state[key] === i));
    b.addEventListener("click", () => { state[key] = i; buttons(el, labels, key); draw(); });
    el.appendChild(b);
  });
}

function current() { D = ALL[state.data]; return D.planes.filter(p => p.center === centers[state.center])[state.plane]; }

function draw() {
  const p = current(), n = D.grid, ax = Array.from({ length: n }, (_, i) => -p.half + 2 * p.half * i / (n - 1));
  const color = p.J.map(row => row.map(v => v === null ? null : Math.log10(v)));
  const height = state.log ? p.J.map(row => row.map(v => v === null ? null : Math.log10(v))) : p.J;
  const ink = css("--ink"), muted = css("--muted"), line = css("--line"), star = css("--star");
  const font = { family: "IBM Plex Sans, system-ui, sans-serif", color: ink, size: 12 };
  const hover = "a = %{x:.3f}<br>b = %{y:.3f}<br>cost = %{customdata:.2f}<extra></extra>";
  const cbar = { title: { text: "cost", side: "right" }, tickvals: ticks.map(Math.log10), ticktext: ticks.map(String),
                 outlinewidth: 0, thickness: 12, len: 0.8, tickfont: { color: ink } };
  const axis3 = t => ({ title: t, color: muted, gridcolor: line, zerolinecolor: line, backgroundcolor: "rgba(0,0,0,0)" });
  const zc = state.log ? Math.log10(p.center_cost) : p.center_cost;

  Plotly.react("p3d", [
    { type: "surface", x: ax, y: ax, z: height, surfacecolor: color, customdata: p.J, cmin, cmax,
      colorscale: scale, colorbar: cbar, hovertemplate: hover, connectgaps: false,
      lighting: { ambient: 0.45, diffuse: 0.85, specular: 0.25, roughness: 0.55, fresnel: 0.2 },
      lightposition: { x: -1e5, y: -1e5, z: 2e5 } },
    { type: "scatter3d", mode: "lines+markers", x: [0, 0], y: [0, 0], z: [state.log ? 1 : 0, zc],
      line: { color: ink, width: 6 }, marker: { size: [0, 5], color: ink }, hoverinfo: "skip", showlegend: false },
  ], {
    margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font,
    scene: { xaxis: axis3("a (d1)"), yaxis: axis3("b (d2)"), zaxis: axis3(state.log ? "log10 cost" : "cost"),
             aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.7 },
             camera: { eye: { x: -1.5, y: -1.5, z: 1.1 } } },
    uirevision: `${state.center}-${state.plane}`,
  }, { responsive: true, displaylogo: false });

  Plotly.react("p2d", [
    { type: "heatmap", x: ax, y: ax, z: color, customdata: p.J, zmin: cmin, zmax: cmax, colorscale: scale,
      showscale: false, hovertemplate: hover, hoverongaps: false },
    { type: "contour", x: ax, y: ax, z: color, showscale: false, hoverinfo: "skip",
      contours: { coloring: "none", start: lo, end: hi, size: (hi - lo) / 12 }, line: { color: "rgba(0,0,0,0.35)", width: 0.6 } },
    { type: "scatter", mode: "markers", x: [0], y: [0], marker: { symbol: "star", size: 16, color: star, line: { color: "#000", width: 1 } },
      hoverinfo: "skip", showlegend: false },
  ], {
    margin: { l: 48, r: 10, t: 6, b: 42 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font,
    xaxis: { title: "a (d1)", color: muted, gridcolor: line, zeroline: false, constrain: "domain" },
    yaxis: { title: "b (d2)", color: muted, gridcolor: line, zeroline: false, scaleanchor: "x" },
  }, { responsive: true, displaylogo: false });

  const vals = p.J.flat(), inside = vals.filter(v => v !== null);
  document.getElementById("sub3d").textContent = `${D.label || "plain cost"} · ${p.center}, random plane ${p.plane}, |a|, |b| ≤ ${p.half}`;
  document.getElementById("rCenter").textContent = p.center_cost.toFixed(2);
  document.getElementById("rMin").textContent = Math.min(...inside).toFixed(2);
  document.getElementById("rPlat").textContent = `${(100 * inside.filter(onPlateau).length / vals.length).toFixed(0)}%`;
  document.getElementById("rOut").textContent = `${(100 * (1 - inside.length / vals.length)).toFixed(0)}%`;
}

// short button labels ("plain", "σ 0.1", "σ 0.3 · λ 1"); the full setting goes in the 3-D subtitle
const lamMain = ALL.map(d => (d.label || "").match(/λ = ([\d.]+)/)).filter(Boolean).map(m => m[1])
  .reduce((a, l, _, arr) => arr.filter(x => x === l).length > arr.filter(x => x === a).length ? l : a, null);
const short = d => { const s = (d.label || "").match(/σ = ([\d.]+)/), l = (d.label || "").match(/λ = ([\d.]+)/);
  return s ? `σ ${s[1]}` + (l && l[1] !== lamMain ? ` · λ ${l[1]}` : "") : "plain"; };
buttons(document.getElementById("costSeg"), ALL.map(short), "data");
document.getElementById("costGroup").hidden = ALL.length < 2;
buttons(document.getElementById("centerSeg"), centers.map(c => c.startsWith("x = 0") ? "box center x = 0" : "best interior sample"), "center");
buttons(document.getElementById("planeSeg"), [...new Set(D.planes.map(p => p.plane))].map(String), "plane");
for (const [id, v] of [["hLin", false], ["hLog", true]]) document.getElementById(id).addEventListener("click", () => {
  state.log = v; document.getElementById("hLin").setAttribute("aria-pressed", String(!v));
  document.getElementById("hLog").setAttribute("aria-pressed", String(v)); draw(); });

document.getElementById("notes").innerHTML = `
  <p><b>Reading it.</b> c is the center point, d1 and d2 are two random orthonormal directions in the 12-D control space, and a, b are the distances along them. Every grid point (${D.grid} × ${D.grid} per plane) is one MuJoCo rollout from the start state block (${D.block.join(", ")}), pusher (${D.pusher.join(", ")}).</p>
  <p><b>Centers.</b> “Box center” is x = 0 (the pusher holds still), planes of half-width 2. “Best interior sample” is the cheapest of 1000 uniform samples in [−0.6, 0.6]¹², planes of half-width 0.5, so they stay inside the box.</p>
  <p><b>LSE smoothing.</b> J<sub>lse</sub>(x) = −λ log mean<sub>l</sub> exp(−J(clip(x + σ ε<sub>l</sub>)) / λ), as GlobalMPPI's LSE rollout, with the same perturbations ε<sub>l</sub> at every grid point, so the surface has no sampling noise from point to point. Same centers and planes as the plain cost; switching keeps the camera.</p>
  <p><b>Plateau.</b> Cost ${D.plateau.toFixed(2)}: the pusher never touches the block, so the cost does not change and there is no gradient. Points outside [−1, 1]¹² are left blank, not clipped.</p>`;

draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script>
"""

data = "[" + ",".join(Path(p).read_text() for p in args.json) + "]"
Path(args.out).write_text(PAGE.replace("/*DATA*/null", data))
print("RESULT saved", args.out)
