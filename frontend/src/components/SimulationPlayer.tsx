/**
 * Simulation player — triggers simulation run and shows progress.
 */

import { useState, useCallback } from "react";
import { runSimulation } from "@/api/client.ts";
import type { SimResult } from "@/types";

/** Props for SimulationPlayer. */
interface SimulationPlayerProps {
  /** Current project ID. */
  projectId: string;
  /** Called when simulation completes. */
  onComplete: (result: SimResult) => void;
}

/**
 * Triggers MuJoCo simulation and displays running status.
 * @param props - Component props.
 * @returns Simulation player interface.
 */
export function SimulationPlayer({
  projectId,
  onComplete,
}: SimulationPlayerProps): React.JSX.Element {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await runSimulation(projectId);
      onComplete(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setRunning(false);
    }
  }, [projectId, onComplete]);

  return (
    <div style={styles.container}>
      <div style={styles.viewer}>
        <div style={styles.placeholder}>
          {running ? (
            <>
              <div style={styles.spinner} />
              <span style={styles.statusText}>Running MuJoCo simulation...</span>
              <span style={styles.hint}>
                Executing workflow steps with physics simulation
              </span>
            </>
          ) : (
            <>
              <span style={styles.statusText}>Ready to Simulate</span>
              <span style={styles.hint}>
                Scene has been built. Click below to run the simulation.
              </span>
            </>
          )}
        </div>
      </div>

      <button
        style={{
          ...styles.button,
          ...(running ? styles.buttonDisabled : {}),
        }}
        disabled={running}
        onClick={handleRun}
      >
        {running ? "Simulating..." : "Run Simulation"}
      </button>

      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", flexDirection: "column", gap: 16 },
  viewer: {
    backgroundColor: "#141414",
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    minHeight: 400,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  placeholder: {
    textAlign: "center",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 12,
  },
  spinner: {
    width: 32,
    height: 32,
    border: "3px solid #333",
    borderTopColor: "#2a6cb0",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  statusText: { fontSize: 18, color: "#ccc" },
  hint: { fontSize: 13, color: "#666" },
  button: {
    padding: "12px 24px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#2a6cb0",
    color: "#fff",
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
    alignSelf: "center",
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
