"use client";

import { useState } from "react";
import type { Curve } from "@/lib/api";

// One series, so no legend box — the title names it. Area fill under the line
// gives the U its weight; the hover crosshair is not optional on a line chart.
// x is the curve INDEX, exactly what the slider binds to, so the marker sits
// where the slider sits with no coordinate translation to get wrong.
export default function CostCurve({
  curve,
  index,
  onPick,
}: {
  curve: Curve;
  index: number;
  onPick: (i: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = curve.points;
  const W = 760;
  const H = 300;
  const pad = { l: 64, r: 18, t: 22, b: 40 };

  const costs = pts.map((p) => p.cost);
  const lo = Math.min(...costs) * 0.985;
  // Clamp the top of the scale. Blocking EVERY transaction costs about six
  // times what doing nothing does, so an honest full-range axis squashes the
  // entire interesting band — the U between the minimum and the do-nothing
  // line — into the bottom sixth of the plot, where it reads as a flat line.
  // The curve is clipped rather than rescaled, and the axis says where it was
  // cut, so nothing is hidden: the region above the cut is uniformly terrible
  // and carries no information a viewer needs.
  const ceilingCost = curve.cost_do_nothing * 1.12;
  const trueMax = Math.max(...costs);
  const clipped = trueMax > ceilingCost;
  const hi = clipped ? ceilingCost : trueMax * 1.02;
  const span = hi - lo || 1;

  const x = (i: number) => pad.l + (i / Math.max(pts.length - 1, 1)) * (W - pad.l - pad.r);
  const y = (c: number) => pad.t + (1 - (c - lo) / span) * (H - pad.t - pad.b);

  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.cost)}`).join(" ");
  const area = `${line} L${x(pts.length - 1)},${H - pad.b} L${x(0)},${H - pad.b} Z`;

  const nearest = (tau: number) => {
    let best = 0, d = Infinity;
    pts.forEach((p, i) => { const dd = Math.abs(p.tau - tau); if (dd < d) { d = dd; best = i; } });
    return best;
  };
  const iStar = nearest(curve.tau_star);
  const iShip = nearest(curve.tau_recommended);
  const active = hover ?? index;
  const cur = pts[active];

  const yTicks = [lo + span * 0.08, lo + span * 0.42, lo + span * 0.78];
  const xTicks = [0, Math.floor(pts.length / 3), Math.floor((2 * pts.length) / 3), pts.length - 1];

  function fromEvent(e: React.MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const frac = (px - pad.l) / (W - pad.l - pad.r);
    return Math.max(0, Math.min(pts.length - 1, Math.round(frac * (pts.length - 1))));
  }

  const tipRight = x(active) > W * 0.6;

  return (
    <figure className="m-0 flex flex-col gap-3">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Total cost against decline threshold. Cost is U-shaped; pure cost minimisation sits at tau ${curve.tau_star.toFixed(3)} and the shipped threshold at ${curve.tau_recommended.toFixed(3)}.`}
        className="h-auto w-full cursor-crosshair select-none"
        style={{ color: "var(--ink)" }}
        onMouseMove={(e) => setHover(fromEvent(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onPick(fromEvent(e))}
      >
        <defs>
          <linearGradient id="cc-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.02" />
          </linearGradient>
          <clipPath id="cc-clip">
            <rect x={pad.l} y={pad.t - 8} width={W - pad.l - pad.r} height={H - pad.t - pad.b + 8} />
          </clipPath>
        </defs>

        {yTicks.map((c, i) => (
          <line key={i} x1={pad.l} y1={y(c)} x2={W - pad.r} y2={y(c)}
            stroke="currentColor" strokeWidth="1" opacity="0.08" />
        ))}

        <line x1={pad.l} y1={y(curve.cost_do_nothing)} x2={W - pad.r} y2={y(curve.cost_do_nothing)}
          stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 4" opacity="0.4" />
        <text x={pad.l + 8} y={y(curve.cost_do_nothing) - 7}
          className="mono" fontSize="10.5" fill="currentColor" opacity="0.55">
          no model at all
        </text>

        <g clipPath="url(#cc-clip)">
          <path d={area} fill="url(#cc-fill)" />
          <path d={line} fill="none" stroke="var(--accent)" strokeWidth="2"
            strokeLinejoin="round" strokeLinecap="round" />
        </g>
        {clipped ? (
          <text x={pad.l} y={pad.t - 8} className="mono"
            fontSize="10" fill="currentColor" opacity="0.45">
            axis cut — blocking everything costs ${Math.round(trueMax / 1000)}k
          </text>
        ) : null}

        {[
          // The shipped threshold always sits at a higher tau (further left)
          // than the pure minimum, so labelling ship to the LEFT of its line
          // and tau* to the RIGHT of its own keeps them apart at any spacing.
          { i: iShip, c: "var(--allow)", t: `ship ${curve.tau_recommended.toFixed(3)}`, anchor: "end" as const, dx: -7 },
          { i: iStar, c: "var(--block)", t: `τ* ${curve.tau_star.toFixed(3)}`, anchor: "start" as const, dx: 7 },
        ].map((m) => (
          <g key={m.t}>
            <line x1={x(m.i)} y1={pad.t - 6} x2={x(m.i)} y2={H - pad.b}
              stroke={m.c} strokeWidth="1.5" strokeDasharray="5 4" opacity="0.85" />
            <text x={x(m.i) + m.dx} y={pad.t + 4} textAnchor={m.anchor} className="mono"
              fontSize="10.5" fill={m.c} fontWeight="500">{m.t}</text>
          </g>
        ))}

        <line x1={x(active)} y1={pad.t - 6} x2={x(active)} y2={H - pad.b}
          stroke="currentColor" strokeWidth="1" opacity="0.3" />
        <circle cx={x(active)} cy={y(cur.cost)} r="5.5" fill="var(--accent)"
          stroke="var(--surface)" strokeWidth="2.5" />

        {hover !== null ? (
        <g transform={`translate(${tipRight ? x(active) - 158 : x(active) + 12}, ${pad.t + 26})`}>
          <rect width="146" height="58" rx="4" fill="var(--surface)"
            stroke="var(--hair-strong)" strokeWidth="1" opacity="0.97" />
          <text x="10" y="18" className="mono" fontSize="10.5" fill="currentColor" opacity="0.6">
            τ {cur.tau.toFixed(4)}
          </text>
          <text x="10" y="34" className="mono" fontSize="12.5" fill="currentColor" fontWeight="600">
            ${Math.round(cur.cost).toLocaleString()}
          </text>
          <text x="10" y="49" className="mono" fontSize="10.5" fill="currentColor" opacity="0.6">
            {(cur.fraud_value_caught * 100).toFixed(0)}% caught ·{" "}
            {(cur.good_declined_rate * 100).toFixed(2)}% declined
          </text>
        </g>
        ) : null}

        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b}
          stroke="currentColor" opacity="0.2" />
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={H - pad.b + 17} textAnchor="middle" className="mono"
            fontSize="10.5" fill="currentColor" opacity="0.5">
            {pts[t].tau.toFixed(3)}
          </text>
        ))}
        <text x={pad.l + (W - pad.l - pad.r) / 2} y={H - 6} textAnchor="middle"
          fontSize="10.5" fill="currentColor" opacity="0.45">
          threshold τ — lower blocks more
        </text>
        {yTicks.map((c, i) => (
          <text key={i} x={pad.l - 10} y={y(c) + 3.5} textAnchor="end" className="mono"
            fontSize="10.5" fill="currentColor" opacity="0.5">
            ${Math.round(c / 1000)}k
          </text>
        ))}
      </svg>
      <figcaption className="text-xs leading-relaxed text-muted">
        Cost is U-shaped in τ. The red line is pure cost minimisation; the green
        one is what actually ships, respecting the{" "}
        {(curve.max_decline_rate * 100).toFixed(0)}% decline ceiling. Hover to
        read any threshold, click to move there.
      </figcaption>
    </figure>
  );
}
