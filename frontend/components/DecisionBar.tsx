"use client";

import { decide, type QueueItem } from "@/lib/api";

// A single stacked bar showing how the queue splits at the current threshold.
// This is the mark that makes the slider feel like it does something: drag it
// and the three bands visibly trade against each other.
//
// Status colours, so they carry a fixed meaning and are never reused for a
// series. Every segment gets a direct label — the amber fails contrast against
// the light surface, and the palette check obliges relief rather than allowing
// colour to be the only signal. 2px surface gaps separate adjacent fills.
export default function DecisionBar({
  items,
  tau,
}: {
  items: QueueItem[];
  tau: number;
}) {
  const counts = { block: 0, review: 0, allow: 0 };
  items.forEach((it) => { counts[decide(it.probability, tau)] += 1; });
  const total = items.length || 1;

  const bands = [
    { key: "block", label: "Block", n: counts.block, color: "var(--block)" },
    { key: "review", label: "Review", n: counts.review, color: "var(--review)" },
    { key: "allow", label: "Allow", n: counts.allow, color: "var(--allow)" },
  ];

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex h-9 w-full gap-[2px] overflow-hidden rounded">
        {bands.map((b) => {
          const share = b.n / total;
          if (b.n === 0) return null;
          return (
            <div
              key={b.key}
              className="flex items-center justify-center overflow-hidden transition-[flex-grow] duration-200 ease-out"
              style={{ flexGrow: b.n, flexBasis: 0, background: b.color }}
              title={`${b.label}: ${b.n} of ${total}`}
            >
              {share > 0.09 ? (
                <span className="mono text-[11px] font-medium text-white tabular">
                  {Math.round(share * 100)}%
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1.5">
        {bands.map((b) => (
          <span key={b.key} className="flex items-center gap-1.5 text-[12px]">
            <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: b.color }} aria-hidden />
            <span className="text-muted">{b.label}</span>
            <span className="mono tabular font-medium">{b.n}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
