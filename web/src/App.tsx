import { useEffect, useMemo, useState } from "react";
import type { CurvePoint, Episode, Payload } from "./types";
import "./App.css";

const DATA_URL = "/data/demo_payload.json";
const VW = 1040;
const VH = 560;
const PAD = 36;
const BUDGET_MIN = 10;
const BUDGET_MAX = 100;
const BUDGET_DEFAULT = 30;

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
        if (!data.episodes?.length || !data.retention_curve?.length) {
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
  const methods = stress
    ? payload?.stress.methods
    : payload?.benchmark.methods;

  if (loadError) return <Missing reason={loadError} />;
  if (!payload) {
    return <div className="shell" aria-busy="true" />;
  }
  if (!toXY || !curve || !methods) {
    return <Missing reason="demo_payload.json is incomplete." />;
  }

  const keep = (ep: Episode) => ep.rank <= k;

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <h1>EgoSelect</h1>
          <p>{payload.meta.formula}</p>
        </div>
        <div className="slider-block">
          <div className="slider-head">
            <label htmlFor="budget">Training budget</label>
            <span className="slider-value">{percent}%</span>
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
            {payload.episodes.map((ep) => {
              const { cx, cy } = toXY(ep.x, ep.y);
              const retained = keep(ep);
              const active = selectedId === ep.id;
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
                    ].join(" ")}
                    cx={cx}
                    cy={cy}
                    r={active ? 7.5 : 6}
                    onClick={() => setSelectedId(ep.id)}
                  />
                </g>
              );
            })}
          </svg>
        </div>

        <aside className="inspector">
          {selected ? (
            <>
              <div className="id" title={selected.id}>
                {selected.id.slice(0, 16)}
              </div>
              <div className={keep(selected) ? "verdict" : "verdict drop"}>
                {keep(selected) ? "KEEP" : "DROP"}
              </div>
              <dl className="stats">
                <div>
                  <dt>training value</dt>
                  <dd>{fmt(selected.value)}</dd>
                </div>
                <div>
                  <dt>quality</dt>
                  <dd>{fmt(selected.quality)}</dd>
                </div>
                <div>
                  <dt>coverage gain</dt>
                  <dd>{fmt(selected.coverage_gain)}</dd>
                </div>
                <div>
                  <dt>redundancy</dt>
                  <dd>{fmt(selected.redundancy)}</dd>
                </div>
              </dl>
              <p className="why-label">Why</p>
              <p className="why">{selected.reason}</p>
            </>
          ) : (
            <p className="idle">Select an episode.</p>
          )}
        </aside>
      </div>

      <footer className="bottom">
        <div className="live">
          <div>
            <div className="k">retained</div>
            <div className="v">
              {k}/{n}
            </div>
          </div>
          <div>
            <div className="k">coverage</div>
            <div className="v">{fmt(curve.coverage)}</div>
          </div>
          <div>
            <div className="k">quality</div>
            <div className="v">{fmt(curve.quality)}</div>
          </div>
          <div>
            <div className="k">redundancy</div>
            <div className="v">{fmt(curve.redundancy)}</div>
          </div>
        </div>

        <div className="compare-row">
          <div className="methods">
            {methods.map((m) => (
              <div
                key={m.name}
                className={m.name === "EgoSelect" ? "method ego" : "method"}
              >
                <div className="name">{shortName(m.name)}</div>
                <div className="nums">
                  {stress ? (
                    <>
                      <div>
                        <span>injected</span>
                        <b>
                          {m.corrupt_retained}/{m.corrupt_pool}
                        </b>
                      </div>
                      <div>
                        <span>cov</span>
                        <b>{fmt(m.coverage)}</b>
                      </div>
                      <div>
                        <span>q</span>
                        <b>{fmt(m.quality)}</b>
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <span>cov</span>
                        <b>{fmt(m.coverage)}</b>
                      </div>
                      <div>
                        <span>q</span>
                        <b>{fmt(m.quality)}</b>
                      </div>
                      <div>
                        <span>red</span>
                        <b>{fmt(m.redundancy)}</b>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="toggle">
            <button
              type="button"
              className={stress ? "" : "on"}
              onClick={() => {
                setStress(false);
              }}
            >
              Normal
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
      </footer>
    </div>
  );
}
