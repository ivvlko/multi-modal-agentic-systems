import { useState, useRef, useCallback } from "react";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/search";
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const CITATION_RE = /\[(img-\d+|\d+)\]/g;

function parseCitationRefs(text) {
  const refs = new Set();
  for (const match of text.matchAll(CITATION_RE)) refs.add(match[1]);
  return [...refs];
}

function highlightCitations(text) {
  return text.split(CITATION_RE).map((part, i) =>
    i % 2 === 0 ? part : <sup key={i} style={{ color: "#0066cc", fontSize: "0.75em" }}>[{part}]</sup>
  );
}

function CitationCard({ citation }) {
  return (
    <div style={styles.citationCard}>
      <div style={styles.citationBadge}>{citation.modality === "image" ? `img-${citation.ref_id}` : citation.ref_id}</div>
      <div style={styles.citationBody}>
        <div style={styles.citationSource}>{citation.source}</div>
        {citation.modality === "image" && citation.image_url && (
          <img src={citation.image_url} alt={citation.source} style={styles.thumbnail} onError={(e) => (e.target.style.display = "none")} />
        )}
        {citation.excerpt && <div style={styles.citationExcerpt}>{citation.excerpt}</div>}
      </div>
    </div>
  );
}

function GroundingBadge({ passed }) {
  return (
    <span style={{ ...styles.badge, background: passed ? "#d4edda" : "#f8d7da", color: passed ? "#155724" : "#721c24" }}>
      {passed ? "Grounded" : "Ungrounded"}
    </span>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [grounded, setGrounded] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  const runSearch = useCallback(async () => {
    if (!query.trim() || loading) return;
    setStreamedAnswer("");
    setCitations([]);
    setGrounded(null);
    setError(null);
    setLoading(true);

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => ws.send(JSON.stringify({ query: query.trim() }));

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "token") {
        setStreamedAnswer((prev) => prev + msg.content);
      } else if (msg.type === "done") {
        setLoading(false);
        fetchFullResult(query.trim());
      } else if (msg.type === "error") {
        setError(msg.message);
        setLoading(false);
      }
    };

    ws.onerror = () => { setError("WebSocket connection failed"); setLoading(false); };
    ws.onclose = () => setLoading(false);
  }, [query, loading]);

  async function fetchFullResult(q) {
    try {
      const res = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();
      const output = data.output;
      setCitations(output.citations ?? []);
      setGrounded(output.grounding_passed);
    } catch {
      /* citations unavailable — streaming answer is still shown */
    }
  }

  const onKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runSearch(); } };

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>WGSN Search</h1>
        <p style={styles.subtitle}>Multi-modal agentic search over fashion &amp; trend content</p>
      </header>

      <main style={styles.main}>
        <div style={styles.searchRow}>
          <input
            style={styles.input}
            type="text"
            placeholder="e.g. spring floral trends with visual examples…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
          />
          <button style={styles.button} onClick={runSearch} disabled={loading || !query.trim()}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {streamedAnswer && (
          <section style={styles.answerSection}>
            <div style={styles.answerHeader}>
              <span style={styles.sectionLabel}>Answer</span>
              {grounded !== null && <GroundingBadge passed={grounded} />}
            </div>
            <div style={styles.answerText}>
              {highlightCitations(streamedAnswer)}
              {loading && <span style={styles.cursor}>▌</span>}
            </div>
          </section>
        )}

        {citations.length > 0 && (
          <section style={styles.citationsSection}>
            <div style={styles.sectionLabel}>Sources</div>
            <div style={styles.citationGrid}>
              {citations.map((c, i) => (
                <CitationCard key={i} citation={{ ...c, ref_id: i + 1 }} />
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", padding: "2rem 1rem" },
  header: { textAlign: "center", marginBottom: "2rem" },
  title: { fontSize: "2rem", fontWeight: 700, letterSpacing: "-0.5px" },
  subtitle: { color: "#666", marginTop: "0.4rem" },
  main: { width: "100%", maxWidth: "780px" },
  searchRow: { display: "flex", gap: "0.5rem" },
  input: { flex: 1, padding: "0.75rem 1rem", fontSize: "1rem", border: "1px solid #ddd", borderRadius: "6px", outline: "none" },
  button: { padding: "0.75rem 1.5rem", fontSize: "1rem", background: "#111", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer" },
  error: { marginTop: "1rem", padding: "0.75rem", background: "#fff3cd", borderRadius: "6px", color: "#856404" },
  answerSection: { marginTop: "2rem" },
  answerHeader: { display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.75rem" },
  sectionLabel: { fontSize: "0.75rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "#888" },
  badge: { fontSize: "0.7rem", fontWeight: 600, padding: "0.2rem 0.5rem", borderRadius: "4px" },
  answerText: { lineHeight: 1.7, fontSize: "1rem", whiteSpace: "pre-wrap" },
  cursor: { animation: "blink 1s step-end infinite" },
  citationsSection: { marginTop: "2rem" },
  citationGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" },
  citationCard: { background: "#fff", border: "1px solid #eee", borderRadius: "8px", padding: "0.75rem", display: "flex", gap: "0.6rem" },
  citationBadge: { fontSize: "0.7rem", fontWeight: 700, color: "#0066cc", minWidth: "2rem" },
  citationBody: { flex: 1, minWidth: 0 },
  citationSource: { fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  citationExcerpt: { fontSize: "0.75rem", color: "#555", lineHeight: 1.5 },
  thumbnail: { width: "100%", borderRadius: "4px", marginTop: "0.4rem", maxHeight: "120px", objectFit: "cover" },
};
