// The dashboard is a VIEW over numbers the pipeline already computed. It never
// recomputes cost, fraud caught, or decline rate from the queue: the queue is a
// 300-row sample, the curve covers the full 118k test slice, and deriving the
// headline figures from the sample would put different numbers on screen than
// in the report. One source, always.

export const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type CurvePoint = {
  tau: number;
  cost: number;
  fraud_value_caught: number;
  fraud_count_caught: number;
  good_declined_rate: number;
  block_rate: number;
};

export type Curve = {
  source: string;
  tau_star: number;
  cost_at_tau_star: number;
  tau_recommended: number;
  cost_at_tau_recommended: number;
  max_decline_rate: number;
  decline_rate_at_recommended: number;
  cost_do_nothing: number;
  currency: string;
  points: CurvePoint[];
};

export type QueueItem = {
  TransactionID: number | null;
  TransactionAmt: number;
  ProductCD?: string | null;
  card4?: string | null;
  card6?: string | null;
  P_emaildomain?: string | null;
  DeviceType?: string | null;
  probability: number;
  is_fraud: number;
  reasons: string[];
  context?: string[];
};

export type Queue = {
  source: string;
  n: number;
  test_rows: number;
  fraud_rate: number;
  note: string;
  items: QueueItem[];
};

export type Policy = {
  tau_block: number;
  tau_review: number;
  tau_unconstrained: number;
  max_decline_rate: number | null;
  currency: string;
  constants: Record<string, number>;
  cost_do_nothing: number;
  trained_on: string;
  trained_at: string;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} -> ${res.status}. ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const getCurve = () => get<Curve>("/curve");
export const getQueue = () => get<Queue>("/queue");
export const getPolicy = () => get<Policy>("/policy");

// The one definition of the decision bands in the frontend. It mirrors
// cutline.serving.Scorer.decide exactly — block at tau, review at a third of
// it. Two definitions of "review" in two places is the same class of bug as
// two copies of the feature logic.
export type Decision = "block" | "review" | "allow";

export function decide(p: number, tauBlock: number): Decision {
  if (p >= tauBlock) return "block";
  if (p >= tauBlock / 3) return "review";
  return "allow";
}

export function money(v: number, currency = "USD") {
  return `${currency === "USD" ? "$" : ""}${Math.round(v).toLocaleString()}`;
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
