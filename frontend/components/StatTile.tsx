export default function StatTile({
  label,
  value,
  sub,
  tone = "ink",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "ok" | "risk" | "accent";
}) {
  const color = {
    ink: "text-ink",
    ok: "text-[var(--ok)]",
    risk: "text-[var(--risk)]",
    accent: "text-[var(--accent)]",
  }[tone];

  return (
    <div className="flex flex-col gap-1 rounded border border-hair bg-surface px-4 py-3">
      <span className="text-[11px] uppercase tracking-wider text-faint font-[family-name:var(--font-mono)]">
        {label}
      </span>
      <span className={`text-2xl font-semibold tabular font-[family-name:var(--font-mono)] ${color}`}>
        {value}
      </span>
      {sub ? <span className="text-xs text-muted leading-snug">{sub}</span> : null}
    </div>
  );
}
