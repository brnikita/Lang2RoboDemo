/**
 * Metrics dashboard — simulation results, step breakdown, iteration controls.
 */

import { useState, useCallback } from "react";
import { runIteration } from "@/api/client.ts";
import type { SimResult, IterationLog } from "@/types";

/** Props for MetricsDashboard. */
interface MetricsDashboardProps {
  /** Current project ID. */
  projectId: string;
  /** Current simulation result. */
  result: SimResult;
  /** Iteration history. */
  history: IterationLog[];
  /** Called after iteration loop completes. */
  onIterate: (result: SimResult, history: IterationLog[]) => void;
}

/**
 * Displays simulation metrics, step breakdown, and iteration controls.
 * @param props - Component props.
 * @returns Dashboard interface.
 */
export function MetricsDashboard({
  projectId,
  result,
  history,
  onIterate,
}: MetricsDashboardProps): React.JSX.Element {
  const [iterating, setIterating] = useState(false);
  const [maxIter, setMaxIter] = useState(3);
  const [error, setError] = useState<string | null>(null);

  const converged =
    result.metrics.success_rate >= 0.95 && result.metrics.collision_count === 0;

  const handleIterate = useCallback(async () => {
    setIterating(true);
    setError(null);
    try {
      const response = await runIteration(projectId, maxIter);
      onIterate(response.result, response.history);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Iteration failed");
    } finally {
      setIterating(false);
    }
  }, [projectId, maxIter, onIterate]);

  return (
    <div style={styles.container}>
      <div style={styles.metricsRow}>
        <div style={styles.metricCard}>
          <span style={styles.metricLabel}>Cycle Time</span>
          <span style={styles.metricValue}>
            {result.metrics.cycle_time_s.toFixed(1)}s
          </span>
        </div>
        <div style={styles.metricCard}>
          <span style={styles.metricLabel}>Success Rate</span>
          <span
            style={{
              ...styles.metricValue,
              color: result.metrics.success_rate >= 0.95 ? "#4ade80" : "#f87171",
            }}
          >
            {(result.metrics.success_rate * 100).toFixed(0)}%
          </span>
        </div>
        <div style={styles.metricCard}>
          <span style={styles.metricLabel}>Collisions</span>
          <span
            style={{
              ...styles.metricValue,
              color: result.metrics.collision_count === 0 ? "#4ade80" : "#fbbf24",
            }}
          >
            {result.metrics.collision_count}
          </span>
        </div>
        <div style={styles.metricCard}>
          <span style={styles.metricLabel}>Status</span>
          <span
            style={{
              ...styles.statusBadge,
              backgroundColor: converged ? "#1a3a2a" : "#3a2a1a",
              color: converged ? "#4ade80" : "#fbbf24",
            }}
          >
            {converged ? "Converged" : "Needs Optimization"}
          </span>
        </div>
      </div>

      <div style={styles.stepsSection}>
        <h4 style={styles.sectionTitle}>Step Breakdown</h4>
        <div style={styles.stepsTable}>
          {result.steps.map((step, i) => (
            <div
              key={i}
              style={{
                ...styles.stepRow,
                backgroundColor: step.success ? "#141414" : "#1a1414",
              }}
            >
              <span style={styles.stepIndex}>{i + 1}</span>
              <span
                style={{
                  ...styles.stepStatus,
                  color: step.success ? "#4ade80" : "#f87171",
                }}
              >
                {step.success ? "OK" : "FAIL"}
              </span>
              <span style={styles.stepDuration}>{step.duration_s.toFixed(2)}s</span>
              <span style={styles.stepCollisions}>
                {step.collision_count > 0 ? `${step.collision_count} col.` : ""}
              </span>
              {step.error && <span style={styles.stepError}>{step.error}</span>}
            </div>
          ))}
        </div>
      </div>

      {history.length > 0 && (
        <div style={styles.historySection}>
          <h4 style={styles.sectionTitle}>Iteration History</h4>
          {history.map((log) => (
            <div key={log.iteration} style={styles.historyRow}>
              <span style={styles.historyIter}>#{log.iteration}</span>
              <span>
                Success: {(log.metrics.success_rate * 100).toFixed(0)}%
              </span>
              <span>Collisions: {log.metrics.collision_count}</span>
            </div>
          ))}
        </div>
      )}

      {!converged && (
        <div style={styles.iterateSection}>
          <div style={styles.iterateControls}>
            <label style={styles.iterateLabel}>
              Max iterations:
              <input
                style={styles.iterateInput}
                type="number"
                min={1}
                max={10}
                value={maxIter}
                onChange={(e) => setMaxIter(parseInt(e.target.value, 10) || 3)}
              />
            </label>
            <button
              style={{
                ...styles.iterateButton,
                ...(iterating ? styles.buttonDisabled : {}),
              }}
              disabled={iterating}
              onClick={handleIterate}
            >
              {iterating ? "Optimizing..." : "Run Optimization"}
            </button>
          </div>
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", flexDirection: "column", gap: 20 },
  metricsRow: { display: "flex", gap: 12 },
  metricCard: {
    flex: 1,
    padding: 16,
    backgroundColor: "#141414",
    borderRadius: 10,
    border: "1px solid #2a2a2a",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
  },
  metricLabel: { fontSize: 12, color: "#666", textTransform: "uppercase" as const, letterSpacing: "0.05em" },
  metricValue: { fontSize: 28, fontWeight: 700, color: "#fff" },
  statusBadge: {
    padding: "4px 12px",
    borderRadius: 20,
    fontSize: 12,
    fontWeight: 600,
  },
  stepsSection: {},
  sectionTitle: {
    margin: "0 0 10px",
    fontSize: 15,
    color: "#aaa",
  },
  stepsTable: {
    borderRadius: 10,
    border: "1px solid #2a2a2a",
    overflow: "hidden",
  },
  stepRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "8px 14px",
    borderBottom: "1px solid #1a1a1a",
    fontSize: 13,
  },
  stepIndex: { color: "#555", width: 24, fontFamily: "monospace" },
  stepStatus: { width: 36, fontWeight: 600, fontSize: 11 },
  stepDuration: { color: "#888", width: 60 },
  stepCollisions: { color: "#fbbf24", fontSize: 11 },
  stepError: { color: "#f87171", fontSize: 11 },
  historySection: {},
  historyRow: {
    display: "flex",
    gap: 16,
    padding: "6px 14px",
    fontSize: 13,
    color: "#888",
    backgroundColor: "#141414",
    borderRadius: 6,
    marginBottom: 4,
  },
  historyIter: { color: "#6cb0e0", fontWeight: 600 },
  iterateSection: {
    padding: 16,
    backgroundColor: "#141414",
    borderRadius: 10,
    border: "1px solid #2a2a2a",
  },
  iterateControls: {
    display: "flex",
    alignItems: "center",
    gap: 16,
  },
  iterateLabel: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 13,
    color: "#aaa",
  },
  iterateInput: {
    width: 50,
    padding: "6px 8px",
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#1a1a1a",
    color: "#e0e0e0",
    fontSize: 13,
    textAlign: "center" as const,
  },
  iterateButton: {
    padding: "10px 20px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#8b5cf6",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
  buttonDisabled: { opacity: 0.5, cursor: "not-allowed" },
  error: {
    color: "#f87171",
    fontSize: 13,
    padding: "8px 12px",
    backgroundColor: "#2a1515",
    borderRadius: 6,
  },
};
