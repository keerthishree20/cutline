"use client";

import type { Curve } from "@/lib/api";

// Hand-authored SVG rather than a charting library: the shape is four
// polylines and three markers, and a dependency would buy nothing but weight.
// x is the curve INDEX, which is exactly what the slider binds to — so the
// marker sits where the slider sits, with no coordinate translation to get
// wrong.
export default function CostCurve({
  curve,
  index,
  onPick,
}: {
  curve: Curve;
  index: number;
  onPick: (i: number) => void;
}) {
  const pts = curve.points;
  const W = 720;
  const H = 260;
  const pad = { l: 62, r: 16, t: 16, b: 34 };

  const costs = pts.map((p) => p.cost);
  const lo = Math.min(...costs, curve.cost_do_nothing);
  const hi = Math.max(...costs, curve.cost_do_nothing);
  const span = hi - lo || 1;

  const x = (i: number) =>
    pad.l + (i / Math.max(pts.length - 1, 1)) * (W - pad.l - pad.r);
  const y = (c: number) =>
    pad.t + (1 - (c - lo) / span) * (H - pad.t - pad.b);

  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.cost)}`).join(" ");

  const nearest = (tau: number) => {
    let best = 0;
    let d = Infinity;
    pts.forEach((p, i) => {
      const dd = Math.abs(p.tau - tau);
      if (dd < d) { d = dd; best = i; }
    });
    return best;
  };
  const iStar = nearest(curve.tau_star);
  const iShip = nearest(curve.tau_recommended);
  const cur = pts[index];

  const ticks = [0, Math.floor(pts.length / 3), Math.floor((2 * pts.length) / 3), pts.length - 1];

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Total cost against decline threshold. Cost is U-shaped; the minimum sits at tau ${curve.tau_star.toFixed(3)} and the shipped threshold at ${curve.tau_recommended.toFixed(3)}.`}
        className="w-full h-auto"
        style={{ color: "var(--ink)" }}
      >
        <line x1={pad.l} y1={y(curve.cost_do_nothing)} x2={W - pad.r} y2={y(curve.cost_do_nothing)}
          stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.35" />
        <text x={W - pad.r} y={y(curve.cost_do_nothing) - 5} textAnchor="end"
          fontSize="10" fill="currentColor" opacity="0.5" fontFamily="var(--font-mono)">
          approve everything
        </text>

        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="currentColor" opacity="0.25" />
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="currentColor" opacity="0.25" />

        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" />

        <line x1={x(iStar)} y1={pad.t} x2={x(iStar)} y2={H - pad.b}
          stroke="var(--risk)" strokeWidth="1.2" strokeDasharray="4 3" />
        <text x={x(iStar) + 5} y={pad.t + 11} fontSize="10" fill="var(--risk)" fontFamily="var(--font-mono)">
          τ* {curve.tau_star.toFixed(3)}
        </text>

        <line x1={x(iShip)} y1={pad.t} x2={x(iShip)} y2={H - pad.b}
          stroke="var(--ok)" strokeWidth="1.2" strokeDasharray="6 3" />
        <text x={x(iShip) + 5} y={pad.t + 24} fontSize="10" fill="var(--ok)" fontFamily="var(--font-mono)">
          ship {curve.tau_recommended.toFixed(3)}
        </text>

        <circle cx={x(index)} cy={y(cur.cost)} r="5" fill="var(--accent)"
          stroke="var(--surface)" strokeWidth="2" />

        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - pad.b + 15} textAnchor="middle" fontSize="10"
            fill="currentColor" opacity="0.55" fontFamily="var(--font-mono)">
            {pts[t].tau.toFixed(3)}
          </text>
        ))}
        <text x={(W - pad.l) / 2 + pad.l} y={H - 3} textAnchor="middle" fontSize="10"
          fill="currentColor" opacity="0.5">
          threshold τ — lower blocks more
        </text>
        {[lo, lo + span / 2, hi].map((c, i) => (
          <text key={i} x={pad.l - 8} y={y(c) + 3} textAnchor="end" fontSize="10"
            fill="currentColor" opacity="0.55" fontFamily="var(--font-mono)">
            {Math.round(c / 1000)}k
          </text>
        ))}

        {pts.map((_, i) => (
          <rect key={i} x={x(i) - 3} y={pad.t} width="6" height={H - pad.t - pad.b}
            fill="transparent" style={{ cursor: "pointer" }} onClick={() => onPick(i)} />
        ))}
      </svg>
      <figcaption className="mt-2 text-xs text-muted">
        Cost is U-shaped in τ. The dashed red line is pure cost minimisation; the
        green line is the threshold actually shipped, which respects the{" "}
        {(curve.max_decline_rate * 100).toFixed(0)}% decline ceiling. Click the
        chart to move the threshold.
      </figcaption>
    </figure>
  );
}
