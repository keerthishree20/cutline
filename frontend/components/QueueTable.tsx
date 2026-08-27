"use client";

import { decide, money, type QueueItem } from "@/lib/api";

const CHIP: Record<string, string> = {
  block: "bg-[var(--risk)]/12 text-[var(--risk)] border-[var(--risk)]/35",
  review: "bg-[var(--warn)]/12 text-[var(--warn)] border-[var(--warn)]/35",
  allow: "bg-[var(--ok)]/12 text-[var(--ok)] border-[var(--ok)]/35",
};

// Whether the policy got this one right. The queue comes from a held-out test
// slice, so the truth is known — and showing it is the difference between a
// list of scores and a demonstration that the threshold does something.
function outcome(isFraud: number, d: string) {
  const stopped = d !== "allow";
  if (isFraud === 1 && stopped) return { label: "caught", tone: "text-[var(--ok)]" };
  if (isFraud === 1 && !stopped) return { label: "missed", tone: "text-[var(--risk)]" };
  if (isFraud === 0 && d === "block") return { label: "false decline", tone: "text-[var(--risk)]" };
  if (isFraud === 0 && d === "review") return { label: "needless review", tone: "text-[var(--warn)]" };
  return { label: "correctly allowed", tone: "text-faint" };
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
  return (
    <div className="overflow-x-auto rounded border border-hair bg-surface">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="bg-surface-2 text-left">
            {["risk", "decision", "amount", "why", "actual"].map((h) => (
              <th key={h} className="px-4 py-2.5 font-medium text-[11px] uppercase tracking-wider text-faint font-[family-name:var(--font-mono)]">
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
              <tr key={`${it.TransactionID ?? i}-${i}`} className="border-t border-hair align-top">
                <td className="px-4 py-2.5 tabular font-[family-name:var(--font-mono)]">
                  {it.probability.toFixed(4)}
                </td>
                <td className="px-4 py-2.5">
                  <span className={`inline-block rounded border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${CHIP[d]}`}>
                    {d}
                  </span>
                </td>
                <td className="px-4 py-2.5 tabular font-[family-name:var(--font-mono)] whitespace-nowrap">
                  {money(it.TransactionAmt, currency)}
                </td>
                <td className="px-4 py-2.5 text-muted max-w-[380px]">
                  {it.reasons.length ? (
                    <ul className="space-y-0.5">
                      {it.reasons.map((r, j) => (
                        <li key={j} className="text-[13px] leading-snug">{r}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-faint text-[13px]">nothing pushed toward fraud</span>
                  )}
                  {it.context?.length ? (
                    <p className="mt-1.5 text-[11px] text-faint font-[family-name:var(--font-mono)]">
                      {it.context.join(" · ")}
                    </p>
                  ) : null}
                </td>
                <td className={`px-4 py-2.5 whitespace-nowrap text-[13px] ${o.tone}`}>
                  {it.is_fraud === 1 ? "fraud" : "legitimate"}
                  <span className="block text-[11px] opacity-70">{o.label}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
