import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { analyzeCsv, analyzeTexts, type AnalyzeResponse } from "./api";

const SAMPLE_LINES = [
  "Fast shipping and neat packaging. Very happy.",
  "Quality is below expectations. Would not buy again.",
  "Great value for the price. Highly recommend!",
];

const COLORS = { positive: "#34d399", negative: "#f87171" };

export default function App() {
  const [input, setInput] = useState(SAMPLE_LINES.join("\n"));
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pieData = useMemo(() => {
    if (!data) return [];
    return [
      { name: "Positive", value: data.summary.positive, key: "positive" },
      { name: "Negative", value: data.summary.negative, key: "negative" },
    ].filter((d) => d.value > 0);
  }, [data]);

  const barData = useMemo(() => {
    if (!data) return [];
    return [
      { name: "Positive", count: data.summary.positive },
      { name: "Negative", count: data.summary.negative },
    ];
  }, [data]);

  async function runAnalyze(texts: string[]) {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeTexts(texts);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  function handleAnalyze() {
    const texts = input
      .split("\n")
      .map((t) => t.trim())
      .filter(Boolean);
    if (!texts.length) {
      setError("Enter at least one review (one per line).");
      return;
    }
    runAnalyze(texts);
  }

  async function handleCsv(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeCsv(file);
      setData(res);
      setInput(res.results.map((r) => r.text).join("\n"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "CSV analysis failed");
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  }

  return (
    <div className="layout">
      <header>
        <h1>Sentiment Dashboard</h1>
        <p>Classify reviews and comments as positive or negative and visualize the split.</p>
      </header>

      <section className="panel input-panel">
        <h2>Reviews</h2>
        <p className="hint">One review per line. CSV upload auto-detects text, review, or comment columns.</p>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={8}
          placeholder="Enter reviews..."
        />
        <div className="actions">
          <button type="button" onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing…" : "Analyze"}
          </button>
          <label className="file-btn">
            Upload CSV
            <input type="file" accept=".csv" onChange={handleCsv} hidden />
          </label>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {data && (
        <>
          <section className="stats">
            <div className="stat-card">
              <span className="label">Total</span>
              <strong>{data.summary.total}</strong>
            </div>
            <div className="stat-card positive">
              <span className="label">Positive</span>
              <strong>
                {data.summary.positive} ({data.summary.positive_pct}%)
              </strong>
            </div>
            <div className="stat-card negative">
              <span className="label">Negative</span>
              <strong>
                {data.summary.negative} ({data.summary.negative_pct}%)
              </strong>
            </div>
          </section>

          <section className="charts">
            <div className="chart-box">
              <h3>Share (pie)</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                  >
                    {pieData.map((entry) => (
                      <Cell
                        key={entry.key}
                        fill={COLORS[entry.key as keyof typeof COLORS]}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-box">
              <h3>Count (bar)</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={barData}>
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis allowDecimals={false} stroke="#94a3b8" />
                  <Tooltip />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {barData.map((_, i) => (
                      <Cell
                        key={i}
                        fill={i === 0 ? COLORS.positive : COLORS.negative}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel results-panel">
            <h2>Results</h2>
            <table>
              <thead>
                <tr>
                  <th>Text</th>
                  <th>Label</th>
                  <th>Certainty</th>
                </tr>
              </thead>
              <tbody>
                {data.results.map((r, i) => (
                  <tr key={i} className={r.label}>
                    <td>{r.text}</td>
                    <td>
                      <span className={`badge ${r.label}`}>
                        {r.label === "positive" ? "Positive" : "Negative"}
                      </span>
                    </td>
                    <td>
                      {r.confidence != null ? (
                        <span title="How sure the classifier is about this label (not how negative the text is).">
                          {(r.confidence * 100).toFixed(1)}%
                          {r.source === "rule" ? " (rule)" : " (model)"}
                          {r.source === "model" &&
                            r.model_confidence != null &&
                            r.confidence < 0.7 && (
                              <span className="low-certainty"> — moderate</span>
                            )}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}

      <style>{`
        .layout { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }
        header h1 { margin: 0 0 0.25rem; font-size: 1.75rem; }
        header p { margin: 0; color: #94a3b8; }
        .panel {
          background: #1a2332;
          border: 1px solid #2d3a4f;
          border-radius: 12px;
          padding: 1.25rem;
          margin-top: 1.5rem;
        }
        .panel h2 { margin: 0 0 0.75rem; font-size: 1.1rem; }
        .hint { color: #94a3b8; font-size: 0.9rem; margin: 0 0 0.75rem; }
        textarea {
          width: 100%;
          background: #0f1419;
          border: 1px solid #334155;
          border-radius: 8px;
          color: inherit;
          padding: 0.75rem;
          resize: vertical;
        }
        .actions { display: flex; gap: 0.75rem; margin-top: 0.75rem; flex-wrap: wrap; }
        button, .file-btn {
          background: #3b82f6;
          color: #fff;
          border: none;
          border-radius: 8px;
          padding: 0.5rem 1rem;
        }
        button:disabled { opacity: 0.6; }
        .file-btn { display: inline-block; background: #475569; }
        .error { color: #f87171; margin-top: 0.75rem; }
        .stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 1rem;
          margin-top: 1.5rem;
        }
        .stat-card {
          background: #1a2332;
          border: 1px solid #2d3a4f;
          border-radius: 12px;
          padding: 1rem;
        }
        .stat-card .label { display: block; color: #94a3b8; font-size: 0.85rem; }
        .stat-card strong { font-size: 1.35rem; }
        .stat-card.positive strong { color: #34d399; }
        .stat-card.negative strong { color: #f87171; }
        .charts {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }
        .chart-box {
          background: #1a2332;
          border: 1px solid #2d3a4f;
          border-radius: 12px;
          padding: 1rem;
        }
        .chart-box h3 { margin: 0 0 0.5rem; font-size: 1rem; color: #cbd5e1; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #2d3a4f; }
        th { color: #94a3b8; font-weight: 600; }
        .badge {
          display: inline-block;
          padding: 0.15rem 0.5rem;
          border-radius: 999px;
          font-size: 0.8rem;
        }
        .badge.positive { background: #064e3b; color: #6ee7b7; }
        .badge.negative { background: #7f1d1d; color: #fca5a5; }
        .low-certainty { color: #94a3b8; font-size: 0.8rem; }
      `}</style>
    </div>
  );
}
