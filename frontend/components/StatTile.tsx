export default function StatTile({
  label,
  value,
  sub,
  tone = "ink",
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "allow" | "block" | "accent";
  accent?: boolean;
}) {
  const color = {
    ink: "text-ink",
    allow: "text-[var(--allow)]",
    block: "text-[var(--block)]",
    accent: "text-[var(--accent)]",
  }[tone];

  return (
    <div
      className={`relative flex flex-col gap-1.5 overflow-hidden rounded-md border bg-surface px-4 py-3.5 ${
        accent ? "border-[var(--accent)]/45" : "border-hair"
      }`}
    >
      {accent ? (
        <span className="absolute inset-x-0 top-0 h-[2px] bg-[var(--accent)]" aria-hidden />
      ) : null}
      <span className="label">{label}</span>
      <span className={`mono tabular metric text-[26px] leading-none font-semibold ${color}`}>
        {value}
      </span>
      {sub ? <span className="text-[11.5px] leading-snug text-muted">{sub}</span> : null}
    </div>
  );
}
