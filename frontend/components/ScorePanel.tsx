"use client";

import { useRef, useState } from "react";
import { API, decide, money, pct, type Decision } from "@/lib/api";

type Scored = {
  transaction_id: number | null;
  probability: number;
  amount: number;
  decision: Decision;
  latency_ms: number;
  sparse: boolean;
  fields_supplied: number;
  fields_absent: number;
  history_seen: number;
  reasons?: { reason: string }[];
};

const NUMERIC = new Set([
  "TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2", "card3",
  "card5", "addr1", "dist1", "C1", "C13", "C14", "D1", "D15", "id_01", "id_02",
  "has_identity",
]);

// A small CSV reader rather than a dependency. It handles quoted fields, which
// is the only part of CSV that actually bites, and nothing else — anything more
// exotic than that belongs in the pipeline, not in a demo upload box.
function parseCSV(text: string): Record<string, unknown>[] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (ch !== "\r") field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  if (rows.length < 2) return [];

  const header = rows[0].map((h) => h.trim());
  return rows.slice(1)
    .filter((r) => r.some((c) => c.trim() !== ""))
    .map((r) => {
      const obj: Record<string, unknown> = {};
      header.forEach((key, j) => {
        const raw = (r[j] ?? "").trim();
        if (raw === "") return;          // absent is missing DATA, not an error
        obj[key] = NUMERIC.has(key) ? Number(raw) : raw;
      });
      return obj;
    });
}

const CHIP: Record<Decision, string> = {
  block: "bg-[var(--risk)]/12 text-[var(--risk)] border-[var(--risk)]/35",
  review: "bg-[var(--warn)]/12 text-[var(--warn)] border-[var(--warn)]/35",
  allow: "bg-[var(--ok)]/12 text-[var(--ok)] border-[var(--ok)]/35",
};

export default function ScorePanel({
  tau,
  currency,
}: {
  tau: number;
  currency: string;
}) {
  const [results, setResults] = useState<Scored[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function send(rows: Record<string, unknown>[]) {
    if (!rows.length) { setErr("No rows found in that file."); return; }
    setBusy(true); setErr(null);
    try {
      const res = await fetch(`${API}/replay?explain=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rows.slice(0, 500)),
      });
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 300)}`);
      const data = await res.json();
      setResults(data.results.sort((a: Scored, b: Scored) => b.probability - a.probability));
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file: File) {
    send(parseCSV(await file.text()));
  }

  // Real held-out transactions rather than an invented "suspicious" row. A
  // hand-made row scores low because this model leans on anonymised counting
  // fields no one can fabricate sensibly, so a fake would understate it.
  async function onSample() {
    setBusy(true); setErr(null);
    try {
      const res = await fetch(`${API}/sample`, { cache: "no-store" });
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
      const data = await res.json();
      await send(data.rows);
    } catch (e) {
      setErr(String((e as Error).message ?? e));
      setBusy(false);
    }
  }

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-lg font-semibold">Score your own</h2>
        <p className="text-xs text-faint">
          Live through <span className="font-[family-name:var(--font-mono)]">POST /replay</span>{" "}
          — the model, not a lookup.
        </p>
      </div>

      <div className="rounded border border-hair bg-surface px-5 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy}
            className="rounded border border-[var(--accent)] bg-[var(--accent-soft)] px-3.5 py-1.5 text-sm font-medium text-[var(--accent)] transition-opacity hover:opacity-80 disabled:opacity-50"
          >
            {busy ? "Scoring…" : "Upload a CSV"}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
              e.target.value = "";
            }}
          />
          <button
            onClick={onSample}
            disabled={busy}
            className="rounded border border-hair px-3.5 py-1.5 text-sm text-muted transition-colors hover:border-[var(--accent)] hover:text-ink disabled:opacity-50"
          >
            Score 25 held-out transactions
          </button>
          {results.length > 0 ? (
            <button
              onClick={() => setResults([])}
              className="ml-auto text-xs text-faint hover:text-ink"
            >
              clear
            </button>
          ) : null}
        </div>

        <p className="mt-3 text-xs text-muted">
          Needs <span className="font-[family-name:var(--font-mono)]">TransactionDT</span>,{" "}
          <span className="font-[family-name:var(--font-mono)]">TransactionAmt</span> and{" "}
          <span className="font-[family-name:var(--font-mono)]">card1</span>; every other column is
          optional, because in this dataset an absent field is missing data rather than an error.
          The button scores the same 25 rows as{" "}
          <span className="font-[family-name:var(--font-mono)]">reports/sample_upload.csv</span>:
          held-out transactions with the labels stripped, half of them drawn from
          the high-risk tail. That weighting is deliberate — a random 25 rows at
          a 3.4% fraud rate flags nothing and shows nothing — and it is stated
          here rather than passed off as a random draw.
        </p>

        {err ? (
          <p className="mt-3 rounded border border-[var(--risk)]/35 bg-[var(--risk)]/10 px-3 py-2 text-xs text-[var(--risk)]">
            {err}
          </p>
        ) : null}

        {results.length > 0 ? (
          <div className="mt-4 flex flex-col gap-2">
            <p className="text-xs text-faint">
              {results.length} scored ·{" "}
              {results.filter((r) => decide(r.probability, tau) === "block").length} blocked at the
              current τ · median{" "}
              {results.map((r) => r.latency_ms).sort((a, b) => a - b)[Math.floor(results.length / 2)]}
              ms
            </p>
            <ul className="flex flex-col gap-2">
              {results.map((r, i) => {
                const d = decide(r.probability, tau);
                return (
                  <li key={i} className="rounded border border-hair bg-surface-2 px-3.5 py-2.5">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className={`rounded border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${CHIP[d]}`}>
                        {d}
                      </span>
                      <span className="tabular font-[family-name:var(--font-mono)] text-sm">
                        {r.probability.toFixed(4)}
                      </span>
                      <span className="tabular font-[family-name:var(--font-mono)] text-sm text-muted">
                        {money(r.amount, currency)}
                      </span>
                      {r.sparse ? (
                        <span className="rounded border border-hair px-1.5 py-0.5 text-[10px] text-faint">
                          sparse · {r.fields_supplied}/{r.fields_supplied + r.fields_absent} fields
                        </span>
                      ) : null}
                      <span className="ml-auto text-[11px] text-faint font-[family-name:var(--font-mono)]">
                        {r.latency_ms}ms
                      </span>
                    </div>
                    {r.reasons?.length ? (
                      <ul className="mt-1.5 space-y-0.5 text-[13px] text-muted">
                        {r.reasons.map((x, j) => <li key={j}>{x.reason}</li>)}
                      </ul>
                    ) : (
                      <p className="mt-1.5 text-[13px] text-faint">nothing pushed toward fraud</p>
                    )}
                  </li>
                );
              })}
            </ul>
            <p className="text-xs text-faint">
              A sparse row scores high for a real reason — missing history is
              predictive here — so it is labelled rather than left to look like a
              broken model. Threshold is whatever the slider says: {pct(tau, 2)} → τ={tau.toFixed(4)}.
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
