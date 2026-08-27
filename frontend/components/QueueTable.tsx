"use client";

import { decide, money, type Decision, type QueueItem } from "@/lib/api";

const TONE: Record<Decision, { chip: string; stripe: string }> = {
  block: {
    chip: "border-[var(--block)]/40 bg-[var(--block)]/10 text-[var(--block)]",
    stripe: "var(--block)",
  },
  review: {
    chip: "border-[var(--review)]/40 bg-[var(--review)]/10 text-[var(--review)]",
    stripe: "var(--review)",
  },
  allow: {
    chip: "border-[var(--allow)]/40 bg-[var(--allow)]/10 text-[var(--allow)]",
    stripe: "var(--allow)",
  },
};

// The queue is a held-out slice, so the truth is known. Showing whether each
// call was right is the difference between a list of scores and a
// demonstration that the threshold does something.
function outcome(isFraud: number, d: Decision) {
  const stopped = d !== "allow";
  if (isFraud === 1 && stopped) return { label: "caught", cls: "text-[var(--allow)]" };
  if (isFraud === 1) return { label: "missed", cls: "text-[var(--block)]" };
  if (d === "block") return { label: "false decline", cls: "text-[var(--block)]" };
  if (d === "review") return { label: "needless review", cls: "text-[var(--review)]" };
  return { label: "correctly allowed", cls: "text-faint" };
}

export default function QueueTable({
  items,
  tau,
  currency,
}: {
  items: QueueItem[];
  tau: number;
  currency: string;
}) {
  if (!items.length) {
    return (
      <p className="rounded-md border border-dashed border-hair-strong bg-surface px-4 py-8 text-center text-sm text-muted">
        No transactions in this band at τ = {tau.toFixed(4)}.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-hair bg-surface">
      <table className="w-full min-w-[760px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-hair bg-surface-2">
            {["", "risk", "decision", "amount", "why it scored", "actual"].map((h, i) => (
              <th key={i}
                className="label px-4 py-2.5 text-left font-medium first:w-1 first:px-0">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((it, i) => {
            const d = decide(it.probability, tau);
            const o = outcome(it.is_fraud, d);
            return (
              <tr key={`${it.TransactionID ?? i}-${i}`}
                className="border-b border-hair align-top last:border-b-0 transition-colors hover:bg-surface-2/60">
                <td className="w-1 p-0">
                  <span className="block h-full min-h-[52px] w-[3px]"
                    style={{ background: TONE[d].stripe }} aria-hidden />
                </td>
                <td className="mono tabular px-4 py-3 text-[13px]">
                  {it.probability.toFixed(4)}
                </td>
                <td className="px-4 py-3">
                  <span className={`inline-block rounded border px-2 py-[3px] text-[10.5px] font-semibold uppercase tracking-wider ${TONE[d].chip}`}>
                    {d}
                  </span>
                </td>
                <td className="mono tabular whitespace-nowrap px-4 py-3 text-[13px]">
                  {money(it.TransactionAmt, currency)}
                </td>
                <td className="max-w-[400px] px-4 py-3">
                  {it.reasons.length ? (
                    <ul className="space-y-[3px]">
                      {it.reasons.map((r, j) => (
                        <li key={j} className="text-[12.5px] leading-snug text-muted">{r}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-[12.5px] text-faint">nothing pushed toward fraud</span>
                  )}
                  {it.context?.length ? (
                    <p className="mono mt-2 text-[10.5px] leading-relaxed text-faint">
                      {it.context.join("  ·  ")}
                    </p>
                  ) : null}
                </td>
                <td className="whitespace-nowrap px-4 py-3">
                  <span className="text-[12.5px] font-medium">
                    {it.is_fraud === 1 ? "fraud" : "legitimate"}
                  </span>
                  <span className={`mt-0.5 block text-[10.5px] ${o.cls}`}>{o.label}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
