import { useEffect, useMemo, useState } from "react";
import type {
  CurvePoint,
  Episode,
  HeldoutField,
  MethodBlock,
  MethodCurve,
  Payload,
} from "./types";
import "./App.css";

const DATA_URL = "/data/demo_payload.json";
const VW = 1040;
const VH = 560;
const PAD = 36;
const BUDGET_MIN = 10;
const BUDGET_MAX = 100;
const BUDGET_DEFAULT = 30;
const CURVE_W = 300;
const CURVE_H = 148;
const CURVE_PAD = { l: 26, r: 8, t: 8, b: 24 };
function budgetCount(n: number, fraction: number): number {
  return Math.max(1, Math.min(n, Math.round(n * fraction)));
}

function fmt(n: number, digits = 3): string {
  return n.toFixed(digits);
}

function shortName(name: string): string {
  if (name === "Dedup-only") return "Dedup";
  if (name === "Diversity-only") return "Diversity";
  return name;
}

function episodeCode(ep: Episode): string {
  return `E${String(ep.rank).padStart(3, "0")}`;
}

function whyLines(reason: string): string[] {
  return reason
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean);
}

function similarPct(value: number | null): string | null {
  if (value == null) return null;
  return `${Math.round(value * 100)}% similar`;
}

function methodKey(name: string): string {
  if (name === "EgoSelect") return "ego";
  if (name === "Dedup-only") return "dedup";
  if (name === "Diversity-only") return "div";
  return "rand";
}

function pointAt(curve: MethodCurve, k: number) {
  return curve.points.find((row) => row.k === k) ?? curve.points[k - 1];
}

function curveX(fraction: number): number {
  return (
    CURVE_PAD.l + fraction * (CURVE_W - CURVE_PAD.l - CURVE_PAD.r)
  );
}

function curveY(coverage: number): number {
  return (
    CURVE_PAD.t + (1 - coverage) * (CURVE_H - CURVE_PAD.t - CURVE_PAD.b)
  );
}

function wins(values: number[], better: "max" | "min"): boolean[] {
  const best = better === "max" ? Math.max(...values) : Math.min(...values);
  const hit = values.map((v) => Math.abs(v - best) < 1e-6);
  return hit.every(Boolean) ? values.map(() => false) : hit;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function HeldOut({
  fields,
  point,
}: {
  fields: HeldoutField[];
  point: CurvePoint;
}) {
  const rows = fields.filter((field) => point[field.key] != null);
  if (!rows.length) return null;
  return (
    <div className="heldout">
      <p className="kicker">Held-out validation</p>
      <p className="note">not used by selector</p>
      <dl>
        {rows.map((field) => (
          <div key={field.key}>
            <dt>{field.label}</dt>
            <dd>{pct(point[field.key] as number)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function CoverageChart({
  series,
  percent,
}: {
  series: MethodCurve[];
  percent: number;
}) {
  const nowX = curveX(percent / 100);
  const ticks = [10, 30, 50, 100];
  const ego = series.find((curve) => curve.name === "EgoSelect");
  const live = ego
    ? pointAt(ego, budgetCount(ego.points.length, percent / 100))
    : undefined;
  return (
    <div className="curve-block">
      <div className="curve-head">
        <p className="k">Coverage vs training budget</p>
        {live ? (
          <p className="curve-read">
            <b>{fmt(live.coverage, 2)}</b>
            <span>at {percent}%</span>
          </p>
        ) : null}
      </div>
      <svg
        viewBox={`0 0 ${CURVE_W} ${CURVE_H}`}
        role="img"
        aria-label="Region coverage versus training budget for Random, Dedup, Diversity, and EgoSelect"
      >
        <line
          className="curve-grid"
          x1={CURVE_PAD.l}
          y1={curveY(1)}
          x2={CURVE_W - CURVE_PAD.r}
          y2={curveY(1)}
        />
        <line
          className="curve-grid"
          x1={CURVE_PAD.l}
          y1={curveY(0.5)}
          x2={CURVE_W - CURVE_PAD.r}
          y2={curveY(0.5)}
        />
        <text className="curve-tick" x={2} y={curveY(1) + 3}>
          1.0
        </text>
        <text className="curve-tick" x={2} y={curveY(0.5) + 3}>
          0.5
        </text>
        <line
          className="curve-now"
          x1={nowX}
          y1={CURVE_PAD.t}
          x2={nowX}
          y2={CURVE_H - CURVE_PAD.b}
        />
        {series.map((curve) => {
          const d = curve.points
            .map((row) => `${curveX(row.fraction)},${curveY(row.coverage)}`)
            .join(" ");
          return (
            <polyline
              key={curve.name}
              className={`curve-line ${methodKey(curve.name)}`}
              points={d}
            />
          );
        })}
        {ticks.map((pct) => (
          <text
            key={pct}
            className="curve-tick"
            x={curveX(pct / 100)}
            y={CURVE_H - 6}
            textAnchor="middle"
          >
            {pct}
          </text>
        ))}
      </svg>
      <div className="curve-legend">
        {series.map((curve) => (
          <span key={curve.name} className={methodKey(curve.name)}>
            {shortName(curve.name)}
          </span>
        ))}
      </div>
    </div>
  );
}

function project(
  episodes: Episode[],
): (x: number, y: number) => { cx: number; cy: number } {
  const xs = episodes.map((e) => e.x);
  const ys = episodes.map((e) => e.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const dx = xmax - xmin || 1;
  const dy = ymax - ymin || 1;
  return (x, y) => ({
    cx: PAD + ((x - xmin) / dx) * (VW - 2 * PAD),
    cy: PAD + (1 - (y - ymin) / dy) * (VH - 2 * PAD),
  });
}

function Missing({ reason }: { reason: string }) {
  return (
    <div className="missing">
      <p>{reason}</p>
      <p>Generate the payload, then reload.</p>
      <code>python scripts/export_demo.py</code>
    </div>
  );
}

export default function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [percent, setPercent] = useState(BUDGET_DEFAULT);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [stress, setStress] = useState(false);
  const [scoreOpen, setScoreOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((res) => {
        if (!res.ok) {
          throw new Error("missing");
        }
        return res.json() as Promise<Payload>;
      })
      .then((data) => {
        if (cancelled) return;
        if (
          !data.episodes?.length ||
          !data.retention_curve?.length ||
          !data.method_curves?.length
        ) {
          setLoadError("demo_payload.json is incomplete.");
          return;
        }
        setPayload(data);
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError("demo_payload.json was not found.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const n = payload?.meta.n_episodes ?? 0;
  const k = budgetCount(n, percent / 100);
  const toXY = useMemo(
    () => (payload ? project(payload.episodes) : null),
    [payload],
  );
  const indexed = payload?.retention_curve[k - 1];
  const curve: CurvePoint | undefined =
    indexed?.k === k
      ? indexed
      : payload?.retention_curve.find((row) => row.k === k);
  const selected = payload?.episodes.find((e) => e.id === selectedId);
  const neighbor = selected?.nearest
    ? payload?.episodes.find((e) => e.id === selected.nearest)
    : undefined;
  const methodCurves = payload?.method_curves;

  if (loadError) return <Missing reason={loadError} />;
  if (!payload) {
    return <div className="shell" aria-busy="true" />;
  }
  if (!toXY || !curve || !methodCurves || methodCurves.length < 4) {
    return <Missing reason="demo_payload.json is incomplete." />;
  }

  const keep = (ep: Episode) => ep.rank <= k;
  const selectedKept = selected ? keep(selected) : false;
  const neighborKept = Boolean(neighbor && keep(neighbor));
  const showLink = Boolean(
    selected && !selectedKept && neighbor && neighborKept && toXY,
  );
  const selectedXY = selected && toXY ? toXY(selected.x, selected.y) : null;
  const neighborXY = neighbor && toXY ? toXY(neighbor.x, neighbor.y) : null;
  const reasons = selected ? whyLines(selected.reason) : [];
  const similar = selected ? similarPct(selected.nearest_similarity) : null;
  const compared: MethodBlock[] = stress
    ? payload.stress.methods
    : methodCurves.map((series) => {
        const row = pointAt(series, k);
        return {
          name: series.name,
          coverage: row?.coverage ?? 0,
          quality: row?.quality ?? 0,
          redundancy: row?.redundancy ?? 0,
          visual_coverage: 0,
          stationary: 0,
        };
      });
  const bestCov = wins(
    compared.map((m) => m.coverage),
    "max",
  );
  const bestQual = wins(
    compared.map((m) => m.quality),
    "max",
  );
  const bestRed = wins(
    compared.map((m) => m.redundancy),
    "min",
  );
  const bestInj = stress
    ? wins(
        compared.map((m) => m.corrupt_retained ?? Number.POSITIVE_INFINITY),
        "min",
      )
    : compared.map(() => false);
  return (
    <div className={scoreOpen ? "shell score-open" : "shell"}>
      <header className="top">
        <div className="brand">
          <h1>EgoSelect</h1>
          <button
            type="button"
            className={scoreOpen ? "score-toggle on" : "score-toggle"}
            aria-expanded={scoreOpen}
            aria-controls="score-layer"
            onClick={() => setScoreOpen((open) => !open)}
          >
            Training Value Score
          </button>
        </div>
        <div className="slider-block">
          <div className="slider-head">
            <label htmlFor="budget">Training budget</label>
            <span className="slider-value">
              {percent}%
              <em>
                {k}/{n}
              </em>
            </span>
          </div>
          <div className="slider-track">
            <span>{BUDGET_MIN}%</span>
            <input
              id="budget"
              type="range"
              min={BUDGET_MIN}
              max={BUDGET_MAX}
              step={1}
              value={percent}
              aria-valuemin={BUDGET_MIN}
              aria-valuemax={BUDGET_MAX}
              aria-valuenow={percent}
              aria-label="Training budget"
              onChange={(ev) => setPercent(Number(ev.target.value))}
            />
            <span>{BUDGET_MAX}%</span>
          </div>
        </div>
      </header>

      {scoreOpen ? (
        <div id="score-layer" className="score-layer">
          <p className="score-formula">{payload.meta.formula}</p>
          <dl>
            <div>
              <dt>Quality</dt>
              <dd>{payload.meta.weights.alpha.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Coverage gain</dt>
              <dd>{payload.meta.weights.beta.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Redundancy</dt>
              <dd>−{payload.meta.weights.gamma.toFixed(2)}</dd>
            </div>
          </dl>
        </div>
      ) : null}

      <div className="stage">
        <div className="field">
          <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label="Behavior space">
            <text className="axis" x={PAD} y={VH - 12}>
              PCA-1
            </text>
            <text className="axis" x={VW - 56} y={PAD - 10}>
              PCA-2
            </text>
            <text className="axis" x={PAD} y={18}>
              behavior space · {n} episodes · {payload.meta.n_regions} regions
            </text>
            {showLink && selectedXY && neighborXY ? (
              <line
                className="link"
                x1={selectedXY.cx}
                y1={selectedXY.cy}
                x2={neighborXY.cx}
                y2={neighborXY.cy}
              />
            ) : null}
            {payload.episodes.map((ep) => {
              const { cx, cy } = toXY(ep.x, ep.y);
              const retained = keep(ep);
              const active = selectedId === ep.id;
              const mate = showLink && neighbor?.id === ep.id;
              return (
                <g key={ep.id}>
                  {ep.role ? (
                    <circle
                      className={retained ? "mark" : "mark drop"}
                      cx={cx}
                      cy={cy}
                      r={11}
                    />
                  ) : null}
                  <circle
                    className={[
                      "ep",
                      retained ? "kept" : "drop",
                      active ? "active" : "",
                      mate ? "mate" : "",
                    ].join(" ")}
                    data-id={ep.id}
                    data-rank={ep.rank}
                    cx={cx}
                    cy={cy}
                    r={active || mate ? 7.5 : 6}
                    onClick={() =>
                      setSelectedId((cur) => (cur === ep.id ? null : ep.id))
                    }
                  />
                </g>
              );
            })}
          </svg>
        </div>

        <aside className="inspector">
          {selected ? (
            <>
              <p className="kicker" title={selected.id}>
                Episode {episodeCode(selected)}
              </p>
              <p className={selectedKept ? "verdict" : "verdict drop"}>
                {selectedKept ? "KEEP" : "DROP"}
              </p>
              {selectedKept ? (
                <dl className="stats">
                  <div>
                    <dt>Training value</dt>
                    <dd>{fmt(selected.value, 2)}</dd>
                  </div>
                  <div>
                    <dt>Quality</dt>
                    <dd>{fmt(selected.quality, 2)}</dd>
                  </div>
                  <div>
                    <dt>Coverage gain</dt>
                    <dd>{fmt(selected.coverage_gain, 2)}</dd>
                  </div>
                  <div>
                    <dt>Redundancy</dt>
                    <dd>{fmt(selected.redundancy, 2)}</dd>
                  </div>
                </dl>
              ) : neighbor && neighborKept ? (
                <div className="nearest">
                  <p className="k">Nearest retained</p>
                  <p className="v" title={neighbor.id}>
                    {episodeCode(neighbor)}
                    {similar ? <span> · {similar}</span> : null}
                  </p>
                </div>
              ) : null}
              {reasons.length ? (
                <>
                  <p className="why-label">Why</p>
                  <ul className="why">
                    {reasons.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </>
              ) : null}
            </>
          ) : (
            <HeldOut fields={payload.meta.heldout ?? []} point={curve} />
          )}
        </aside>
      </div>

      <footer className="bottom">
        <div className="compare-row">
          <div>
            <p className="compare-kicker">
              {stress
                ? `Equal keep · ${payload.stress.n_keep} · ${payload.stress.n_injected} injected`
                : `Equal keep · ${percent}%`}
            </p>
            <div className="methods">
              {compared.map((m, i) => (
                <div
                  key={m.name}
                  className={m.name === "EgoSelect" ? "method ego" : "method"}
                >
                  <div className="name">{shortName(m.name)}</div>
                  <div className={stress ? "nums kinds" : "nums"}>
                    {stress ? (
                      <>
                        <div>
                          <span>corrupted</span>
                          <b className={bestInj[i] ? "best" : ""}>
                            {m.corrupt_retained}/{m.corrupt_pool}
                          </b>
                        </div>
                        <div>
                          <span>duplicate</span>
                          <b>{m.dup_retained}</b>
                        </div>
                        <div>
                          <span>idle</span>
                          <b>{m.idle_retained}</b>
                        </div>
                        <div>
                          <span>over-region</span>
                          <b>{m.over_retained}</b>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <span>coverage</span>
                          <b className={bestCov[i] ? "best" : ""}>
                            {fmt(m.coverage)}
                          </b>
                        </div>
                        <div>
                          <span>quality</span>
                          <b className={bestQual[i] ? "best" : ""}>
                            {fmt(m.quality)}
                          </b>
                        </div>
                        <div>
                          <span>redundancy</span>
                          <b className={bestRed[i] ? "best" : ""}>
                            {fmt(m.redundancy)}
                          </b>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="toggle">
            <button
              type="button"
              className={stress ? "" : "on"}
              onClick={() => {
                setStress(false);
              }}
            >
              Original
            </button>
            <button
              type="button"
              className={stress ? "on" : ""}
              onClick={() => {
                setStress(true);
              }}
            >
              Stress test
            </button>
          </div>
        </div>
        <CoverageChart series={methodCurves} percent={percent} />
      </footer>
    </div>
  );
}
