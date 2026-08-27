"use client";

import { useEffect, useMemo, useState } from "react";
import CostCurve from "@/components/CostCurve";
import QueueTable from "@/components/QueueTable";
import ScorePanel from "@/components/ScorePanel";
import StatTile from "@/components/StatTile";
import {
  API, decide, getCurve, getPolicy, getQueue, money, pct,
  type Curve, type Policy, type Queue,
} from "@/lib/api";

type Filter = "all" | "block" | "review" | "allow";

export default function Page() {
  const [curve, setCurve] = useState<Curve | null>(null);
  const [queue, setQueue] = useState<Queue | null>(null);
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The slider binds to an INDEX, never to a float threshold. Every stored tau
  // is reachable and nothing in between is, which makes it impossible to land
  // between two curve rows and have to search for the nearest — the same
  // snapping bug that was fixed in the Python sweep.
  const [index, setIndex] = useState(0);
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    Promise.all([getCurve(), getQueue(), getPolicy()])
      .then(([c, q, p]) => {
        setCurve(c); setQueue(q); setPolicy(p);
        let best = 0, d = Infinity;
        c.points.forEach((pt, i) => {
          const dd = Math.abs(pt.tau - c.tau_recommended);
          if (dd < d) { d = dd; best = i; }
        });
        setIndex(best);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const point = curve?.points[index];

  const visible = useMemo(() => {
    if (!queue || !point) return [];
    if (filter === "all") return queue.items;
    return queue.items.filter((it) => decide(it.probability, point.tau) === filter);
  }, [queue, point, filter]);

  if (error) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-20">
        <h1 className="text-xl font-semibold">Cannot reach the scoring service</h1>
        <p className="mt-3 text-sm text-muted">{error}</p>
        <pre className="mt-4 overflow-x-auto rounded border border-hair bg-surface-2 p-4 text-xs">
{`# start the API, then reload
.venv/bin/uvicorn api.main:app --port 8000

# if /curve or /queue 503s, generate them first
.venv/bin/python scripts/03_cost_model.py
.venv/bin/python scripts/05_demo_queue.py`}
        </pre>
        <p className="mt-3 text-xs text-faint">Currently pointing at {API}</p>
      </main>
    );
  }

  if (!curve || !queue || !policy || !point) {
    return <main className="mx-auto max-w-2xl px-6 py-20 text-sm text-muted">Loading…</main>;
  }

  const saved = curve.cost_do_nothing - point.cost;
  const atShip = Math.abs(point.tau - curve.tau_recommended) < 1e-12;
  const overCeiling = point.good_declined_rate > curve.max_decline_rate;
  const counts = { block: 0, review: 0, allow: 0 } as Record<string, number>;
  queue.items.forEach((it) => { counts[decide(it.probability, point.tau)] += 1; });

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-10">
      <header className="flex flex-col gap-2">
        <p className="text-[11px] uppercase tracking-[0.13em] text-[var(--accent)] font-[family-name:var(--font-mono)]">
          Track 2 · AI Risk Manager
        </p>
        <h1 className="text-3xl font-semibold tracking-tight">Cutline</h1>
        <p className="max-w-2xl text-[15px] text-muted">
          A payment risk scorer whose decline threshold comes from cost, not
          accuracy. Trained on {policy.trained_on}, {queue.test_rows.toLocaleString()}{" "}
          held-out transactions at {pct(queue.fraud_rate, 2)} fraud.
        </p>
      </header>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-lg font-semibold">Threshold</h2>
          <p className="text-xs text-faint">
            {curve.points.length} positions — every distinct value isotonic
            calibration can produce. A finer slider would misrepresent the
            model&apos;s resolution.
          </p>
        </div>

        <div className="rounded border border-hair bg-surface px-5 py-4">
          <div className="flex flex-wrap items-center gap-4">
            <span className="tabular font-[family-name:var(--font-mono)] text-2xl font-semibold text-[var(--accent)]">
              τ = {point.tau.toFixed(4)}
            </span>
            {atShip ? (
              <span className="rounded border border-[var(--ok)]/35 bg-[var(--ok)]/12 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-[var(--ok)]">
                shipped policy
              </span>
            ) : null}
            {overCeiling ? (
              <span className="rounded border border-[var(--risk)]/35 bg-[var(--risk)]/12 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-[var(--risk)]">
                over the {pct(curve.max_decline_rate, 0)} decline ceiling
              </span>
            ) : null}
            <button
              onClick={() => {
                let best = 0, d = Infinity;
                curve.points.forEach((pt, i) => {
                  const dd = Math.abs(pt.tau - curve.tau_recommended);
                  if (dd < d) { d = dd; best = i; }
                });
                setIndex(best);
              }}
              className="ml-auto rounded border border-hair px-3 py-1 text-xs text-muted hover:text-ink hover:border-[var(--accent)] transition-colors"
            >
              reset to shipped
            </button>
          </div>

          <input
            type="range"
            min={0}
            max={curve.points.length - 1}
            step={1}
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            aria-label="decline threshold"
            className="mt-4 w-full"
          />
          <div className="mt-1 flex justify-between text-[11px] text-faint font-[family-name:var(--font-mono)]">
            <span>block nothing</span>
            <span>block everything</span>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile
            label={`total cost (${curve.currency})`}
            value={money(point.cost, curve.currency)}
            sub={`vs ${money(curve.cost_do_nothing, curve.currency)} approving everything`}
            tone="ink"
          />
          <StatTile
            label="saved"
            value={money(saved, curve.currency)}
            sub={`${money((saved / queue.test_rows) * 10000, curve.currency)} per 10,000 transactions`}
            tone={saved > 0 ? "ok" : "risk"}
          />
          <StatTile
            label="fraud caught by value"
            value={pct(point.fraud_value_caught)}
            sub={`${pct(point.fraud_count_caught)} by transaction count`}
            tone="accent"
          />
          <StatTile
            label="good customers declined"
            value={pct(point.good_declined_rate, 2)}
            sub={`ceiling is ${pct(curve.max_decline_rate, 0)}`}
            tone={overCeiling ? "risk" : "ok"}
          />
        </div>
        <p className="text-xs text-faint">
          These four figures are read from the cost curve at τ, which covers all{" "}
          {queue.test_rows.toLocaleString()} test transactions — not derived from
          the {queue.n} rows in the queue below. One source, so the dashboard and
          the report never disagree.
        </p>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">Cost against threshold</h2>
        <div className="rounded border border-hair bg-surface px-5 py-4">
          <CostCurve curve={curve} index={index} onPick={setIndex} />
        </div>
      </section>

      <ScorePanel tau={point.tau} currency={curve.currency} />

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-lg font-semibold">Review queue</h2>
          <div className="flex gap-1.5">
            {(["all", "block", "review", "allow"] as Filter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded border px-2.5 py-1 text-xs transition-colors ${
                  filter === f
                    ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                    : "border-hair text-muted hover:text-ink"
                }`}
              >
                {f}
                {f !== "all" ? ` ${counts[f]}` : ` ${queue.items.length}`}
              </button>
            ))}
          </div>
        </div>
        <QueueTable items={visible} tau={point.tau} currency={curve.currency} />
        <p className="text-xs text-faint">
          A {queue.n}-transaction sample of the held-out slice, sampled across the
          whole risk range so the review band and the correct allows are both
          visible. Reasons come from SHAP over the raw model output; isotonic
          sits downstream and is monotone, so their order and direction hold
          exactly.
        </p>
      </section>

      <footer className="border-t border-hair pt-5 text-xs text-faint">
        <p>
          Constants behind the threshold — dispute fee{" "}
          {money(policy.constants.dispute_fee, curve.currency)}, margin{" "}
          {pct(policy.constants.gross_margin, 0)}, support{" "}
          {money(policy.constants.support_cost, curve.currency)}, churn{" "}
          {money(policy.constants.churn_cost ?? 0, curve.currency)}, review{" "}
          {money(policy.constants.analyst_cost, curve.currency)}. Every one is an
          assumption, on screen so it can be argued with. Pure cost minimisation
          would sit at τ={curve.tau_star.toFixed(4)}; the shipped
          threshold respects the decline ceiling instead.
        </p>
      </footer>
    </main>
  );
}
