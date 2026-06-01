export type SentimentResult = {
  text: string;
  text_normalized?: string;
  label: string;
  confidence: number | null;
  model_confidence?: number | null;
  source?: "model" | "rule";
};

export type Summary = {
  total: number;
  positive: number;
  negative: number;
  other: number;
  positive_pct: number;
  negative_pct: number;
};

export type AnalyzeResponse = {
  results: SentimentResult[];
  summary: Summary;
};

export async function analyzeTexts(texts: string[]): Promise<AnalyzeResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export async function analyzeCsv(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/analyze/csv", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}
