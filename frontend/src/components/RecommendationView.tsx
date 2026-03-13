/**
 * Recommendation view — scenario input, plan display, equipment placement.
 */

import { useState, useCallback } from "react";
import { generateRecommendation, buildScene } from "@/api/client.ts";
import type { Recommendation } from "@/types";

/** Props for RecommendationView. */
interface RecommendationViewProps {
  /** Current project ID. */
  projectId: string;
  /** Called when user confirms the plan and scene is built. */
  onConfirm: (recommendation: Recommendation) => void;
}

/**
 * Scenario input → Claude recommendation → plan display → confirm.
 * @param props - Component props.
 * @returns Recommendation interface.
 */
export function RecommendationView({
  projectId,
  onConfirm,
}: RecommendationViewProps): React.JSX.Element {
  const [scenario, setScenario] = useState("");
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = useCallback(async () => {
    if (!scenario.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const rec = await generateRecommendation(projectId, scenario);
      setRecommendation(rec);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }, [projectId, scenario]);

  const handleConfirm = useCallback(async () => {
    if (!recommendation) return;
    setBuilding(true);
    setError(null);
    try {
      await buildScene(projectId);
      onConfirm(recommendation);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scene build failed");
    } finally {
      setBuilding(false);
    }
  }, [projectId, recommendation, onConfirm]);

  return (
    <div style={styles.container}>
      {!recommendation ? (
        <div style={styles.inputSection}>
          <h3 style={styles.title}>Describe Your Automation Scenario</h3>
          <textarea
            style={styles.textarea}
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            placeholder="Example: 3D print farm with 5 printers. Robot picks finished prints from build plates, places on conveyor. Conveyor moves to post-processing table. Camera inspects for print failures."
            rows={5}
          />
          <button
            style={{
              ...styles.button,
              ...(loading || !scenario.trim() ? styles.buttonDisabled : {}),
            }}
            disabled={loading || !scenario.trim()}
            onClick={handleGenerate}
          >
            {loading ? "Generating Plan..." : "Generate Plan"}
          </button>
        </div>
      ) : (
        <div style={styles.planSection}>
          <div style={styles.planHeader}>
            <h3 style={styles.title}>Automation Plan</h3>
            <div style={styles.actions}>
              <button
                style={styles.modifyButton}
                onClick={() => setRecommendation(null)}
              >
                Modify
              </button>
              <button
                style={{
                  ...styles.button,
                  ...(building ? styles.buttonDisabled : {}),
                }}
                disabled={building}
                onClick={handleConfirm}
              >
                {building ? "Building Scene..." : "Confirm & Build Scene"}
              </button>
            </div>
          </div>

          {recommendation.text_plan && (
            <div style={styles.textPlan}>{recommendation.text_plan}</div>
          )}

          <div style={styles.grid}>
            <div style={styles.card}>
              <h4 style={styles.cardTitle}>Equipment ({recommendation.equipment.length})</h4>
              {recommendation.equipment.map((eq) => (
                <div key={eq.equipment_id} style={styles.item}>
                  <span style={styles.itemId}>{eq.equipment_id}</span>
                  <span style={styles.itemPurpose}>{eq.purpose}</span>
                  <span style={styles.itemPos}>
                    ({eq.position[0].toFixed(1)}, {eq.position[1].toFixed(1)}, {eq.position[2].toFixed(1)})
                  </span>
                </div>
              ))}
            </div>

            <div style={styles.card}>
              <h4 style={styles.cardTitle}>Workflow ({recommendation.workflow_steps.length} steps)</h4>
              {recommendation.workflow_steps.map((step) => (
                <div key={step.order} style={styles.item}>
                  <span style={styles.stepOrder}>{step.order}</span>
                  <span style={styles.stepAction}>{step.action}</span>
                  <span style={styles.itemPurpose}>
                    {step.equipment_id ?? "—"} → {step.target}
                  </span>
                </div>
              ))}
            </div>

            <div style={styles.card}>
              <h4 style={styles.cardTitle}>Work Objects</h4>
              {recommendation.work_objects.map((obj) => (
                <div key={obj.name} style={styles.item}>
                  <span style={styles.itemId}>{obj.name}</span>
                  <span style={styles.itemPurpose}>
                    {obj.shape} x{obj.count} ({obj.mass_kg}kg)
                  </span>
                </div>
              ))}
            </div>

            <div style={styles.card}>
              <h4 style={styles.cardTitle}>Expected Metrics</h4>
              <div style={styles.metric}>
                <span>Cycle time</span>
                <span>{recommendation.expected_metrics.cycle_time_s.toFixed(1)}s</span>
              </div>
              <div style={styles.metric}>
                <span>Throughput</span>
                <span>{recommendation.expected_metrics.throughput_per_hour}/hr</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", flexDirection: "column", gap: 16 },
  inputSection: { display: "flex", flexDirection: "column", gap: 12 },
  title: { margin: 0, fontSize: 18, color: "#fff" },
  textarea: {
    padding: 14,
    borderRadius: 10,
    border: "1px solid #333",
    backgroundColor: "#141414",
    color: "#e0e0e0",
    fontSize: 14,
    resize: "vertical",
    lineHeight: 1.5,
  },
  button: {
    padding: "10px 24px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#2a6cb0",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    alignSelf: "flex-end",
  },
  buttonDisabled: { opacity: 0.5, cursor: "not-allowed" },
  modifyButton: {
    padding: "10px 24px",
    borderRadius: 8,
    border: "1px solid #333",
    backgroundColor: "transparent",
    color: "#aaa",
    fontSize: 14,
    cursor: "pointer",
  },
  planSection: { display: "flex", flexDirection: "column", gap: 16 },
  planHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  actions: { display: "flex", gap: 8 },
  textPlan: {
    padding: 16,
    backgroundColor: "#141414",
    borderRadius: 10,
    border: "1px solid #2a2a2a",
    fontSize: 13,
    lineHeight: 1.6,
    color: "#ccc",
    whiteSpace: "pre-wrap",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 12,
  },
  card: {
    padding: 14,
    backgroundColor: "#141414",
    borderRadius: 10,
    border: "1px solid #2a2a2a",
  },
  cardTitle: {
    margin: "0 0 10px",
    fontSize: 14,
    color: "#aaa",
    borderBottom: "1px solid #222",
    paddingBottom: 6,
  },
  item: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "4px 0",
    fontSize: 12,
  },
  itemId: {
    color: "#6cb0e0",
    fontFamily: "monospace",
    fontSize: 11,
  },
  itemPurpose: { color: "#888" },
  itemPos: { color: "#555", fontFamily: "monospace", fontSize: 11 },
  stepOrder: {
    width: 18,
    height: 18,
    borderRadius: "50%",
    backgroundColor: "#2a2a2a",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 10,
    color: "#888",
  },
  stepAction: {
    padding: "2px 6px",
    borderRadius: 4,
    backgroundColor: "#1a2a3a",
    color: "#6cb0e0",
    fontSize: 11,
    fontWeight: 600,
  },
  metric: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    fontSize: 13,
    color: "#ccc",
    borderBottom: "1px solid #1a1a1a",
  },
  error: {
    color: "#f87171",
    fontSize: 13,
    padding: "8px 12px",
    backgroundColor: "#2a1515",
    borderRadius: 6,
  },
};
