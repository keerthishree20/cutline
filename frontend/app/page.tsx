"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import CostCurve from "@/components/CostCurve";
import DecisionBar from "@/components/DecisionBar";
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

  // The slider binds to an INDEX, never a float threshold. Every stored tau is
  // reachable and nothing between two of them is, so landing between curve rows
  // is impossible rather than merely avoided.
  const [index, setIndex] = useState(0);
  const [filter, setFilter] = useState<Filter>("all");

  const nearestTo = useCallback((c: Curve, tau: number) => {
    let best = 0, d = Infinity;
    c.points.forEach((pt, i) => { const dd = Math.abs(pt.tau - tau); if (dd < d) { d = dd; best = i; } });
    return best;
  }, []);

  useEffect(() => {
    Promise.all([getCurve(), getQueue(), getPolicy()])
      .then(([c, q, p]) => {
        setCurve(c); setQueue(q); setPolicy(p);
        setIndex(nearestTo(c, c.tau_recommended));
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [nearestTo]);

  const point = curve?.points[index];

  const visible = useMemo(() => {
    if (!queue || !point) return [];
    const rows = filter === "all"
      ? queue.items
      : queue.items.filter((it) => decide(it.probability, point.tau) === filter);
    // Order by expected loss, not raw probability. Sorted by score alone the
    // top of the queue is a dozen identical 1.0000 rows and reads as broken;
    // a real analyst works the biggest money at risk first, which also happens
    // to be far more legible.
    return [...rows].sort(
      (a, b) => b.probability * b.TransactionAmt - a.probability * a.TransactionAmt,
    );
  }, [queue, point, filter]);

  if (error) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-24">
        <p className="label mb-2">Service unreachable</p>
        <h1 className="text-2xl font-semibold tracking-tight">Cannot reach the scorer</h1>
        <p className="mt-3 text-sm text-muted">{error}</p>
        <pre className="mono mt-5 overflow-x-auto rounded-md border border-hair bg-surface-2 p-4 text-[12px] leading-relaxed">
{`.venv/bin/uvicorn api.main:app --port 8000

# if /curve or /queue returns 503, generate them first
.venv/bin/python scripts/03_cost_model.py
.venv/bin/python scripts/05_demo_queue.py`}
        </pre>
        <p className="mt-3 text-xs text-faint">Pointing at {API}</p>
      </main>
    );
  }

  if (!curve || !queue || !policy || !point) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-24">
        <div className="h-2 w-28 animate-pulse rounded bg-surface-2" />
        <div className="mt-4 h-8 w-52 animate-pulse rounded bg-surface-2" />
      </main>
    );
  }

  const saved = curve.cost_do_nothing - point.cost;
  const atShip = index === nearestTo(curve, curve.tau_recommended);
  const overCeiling = point.good_declined_rate > curve.max_decline_rate;
  const counts = { block: 0, review: 0, allow: 0 } as Record<string, number>;
  queue.items.forEach((it) => { counts[decide(it.probability, point.tau)] += 1; });

  return (
    <div className="min-h-screen">
      <header className="border-b border-hair bg-surface">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="flex flex-col gap-1.5">
              <p className="label text-[var(--accent)]">Track 2 · AI Risk Manager</p>
              <h1 className="text-[34px] font-semibold leading-none tracking-tight">Cutline</h1>
            </div>
            <dl className="flex flex-wrap gap-x-7 gap-y-2">
              {[
                ["model", policy.trained_on],
                ["held out", queue.test_rows.toLocaleString()],
                ["fraud rate", pct(queue.fraud_rate, 2)],
                ["ceiling", pct(curve.max_decline_rate, 0)],
              ].map(([k, v]) => (
                <div key={k} className="flex flex-col gap-0.5">
                  <dt className="label">{k}</dt>
                  <dd className="mono tabular text-[13px] font-medium">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
          <p className="max-w-2xl text-[14.5px] leading-relaxed text-muted">
            The decline threshold comes from cost, not accuracy. A missed
            chargeback and a wrongly declined sale are not worth the same, so
            0.5 is almost never the right cut.
          </p>
        </div>
      </header>

      <main className="mx-auto flex max-w-6xl flex-col gap-11 px-6 py-10">
        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 className="text-[17px] font-semibold">Threshold</h2>
            <p className="text-[11.5px] text-faint">
              {curve.points.length} positions — every value isotonic calibration
              can produce. A finer slider would overstate the model&apos;s resolution.
            </p>
          </div>

          <div className="card px-5 py-5">
            <div className="flex flex-wrap items-center gap-3">
              <span className="mono tabular text-[30px] font-semibold leading-none text-[var(--accent)]">
                τ {point.tau.toFixed(4)}
              </span>
              {atShip ? (
                <span className="rounded border border-[var(--allow)]/40 bg-[var(--allow)]/10 px-2 py-[3px] text-[10.5px] font-semibold uppercase tracking-wider text-[var(--allow)]">
                  shipped policy
                </span>
              ) : null}
              {overCeiling ? (
                <span className="rounded border border-[var(--block)]/40 bg-[var(--block)]/10 px-2 py-[3px] text-[10.5px] font-semibold uppercase tracking-wider text-[var(--block)]">
                  over the {pct(curve.max_decline_rate, 0)} ceiling
                </span>
              ) : null}
              <button
                onClick={() => setIndex(nearestTo(curve, curve.tau_recommended))}
                className="ml-auto rounded border border-hair px-3 py-1.5 text-[12px] text-muted transition-colors hover:border-[var(--accent)] hover:text-ink"
              >
                reset to shipped
              </button>
            </div>

            <input
              type="range" min={0} max={curve.points.length - 1} step={1} value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
              aria-label="decline threshold"
              aria-valuetext={`tau ${point.tau.toFixed(4)}, ${pct(point.good_declined_rate, 2)} of good customers declined`}
              className="mt-5"
            />
            <div className="mono mt-1.5 flex justify-between text-[10.5px] text-faint">
              <span>block nothing</span>
              <span>block everything</span>
            </div>

            <div className="mt-6 border-t border-hair pt-5">
              <p className="label mb-3">how the queue splits here</p>
              <DecisionBar items={queue.items} tau={point.tau} />
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile
              label={`total cost · ${curve.currency}`}
              value={money(point.cost, curve.currency)}
              sub={`${money(curve.cost_do_nothing, curve.currency)} with no model at all`}
            />
            <StatTile
              accent
              label="saved"
              value={money(saved, curve.currency)}
              sub={`${money((saved / queue.test_rows) * 10000, curve.currency)} per 10,000 transactions`}
              tone={saved > 0 ? "allow" : "block"}
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
              sub={`ceiling ${pct(curve.max_decline_rate, 0)}`}
              tone={overCeiling ? "block" : "allow"}
            />
          </div>
          <p className="text-[11.5px] leading-relaxed text-faint">
            Read from the cost curve at τ, which covers all{" "}
            {queue.test_rows.toLocaleString()} held-out transactions — not derived
            from the {queue.n} rows below. One source, so the dashboard and the
            report can never disagree.
          </p>
        </section>

        <section className="flex flex-col gap-4">
          <h2 className="text-[17px] font-semibold">Cost against threshold</h2>
          <div className="card px-5 py-5">
            <CostCurve curve={curve} index={index} onPick={setIndex} />
          </div>
        </section>

        <ScorePanel tau={point.tau} currency={curve.currency} />

        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 className="text-[17px] font-semibold">Review queue</h2>
            <div className="flex gap-1.5">
              {(["all", "block", "review", "allow"] as Filter[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`rounded border px-2.5 py-1 text-[12px] transition-colors ${
                    filter === f
                      ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                      : "border-hair text-muted hover:border-hair-strong hover:text-ink"
                  }`}
                >
                  {f}{" "}
                  <span className="mono tabular opacity-70">
                    {f === "all" ? queue.items.length : counts[f]}
                  </span>
                </button>
              ))}
            </div>
          </div>
          <QueueTable items={visible} tau={point.tau} currency={curve.currency} />
          <p className="text-[11.5px] leading-relaxed text-faint">
            Ordered by expected loss — probability times amount — which is how an
            analyst actually works a queue. A {queue.n}-transaction sample of the held-out slice, drawn across the
            whole risk range so the review band and the correct allows are both
            visible. Reasons come from SHAP over the raw model output; isotonic
            sits downstream and is monotone, so their order and direction hold
            exactly.
          </p>
        </section>

        <footer className="border-t border-hair pt-6 text-[11.5px] leading-relaxed text-faint">
          <p className="max-w-3xl">
            Constants behind the threshold — dispute fee{" "}
            {money(policy.constants.dispute_fee, curve.currency)}, margin{" "}
            {pct(policy.constants.gross_margin, 0)}, support{" "}
            {money(policy.constants.support_cost, curve.currency)}, churn{" "}
            {money(policy.constants.churn_cost ?? 0, curve.currency)}, manual review{" "}
            {money(policy.constants.analyst_cost, curve.currency)}. Every one is an
            assumption, on screen so it can be argued with. Pure cost minimisation
            would sit at τ={curve.tau_star.toFixed(4)} and decline far more; the
            shipped threshold respects the ceiling instead.
          </p>
        </footer>
      </main>
    </div>
  );
}
