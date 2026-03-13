/**
 * 3D scene viewer with calibration — user picks two points + enters distance.
 */

import { useState, useCallback } from "react";
import { calibrateAndAnalyze } from "@/api/client.ts";
import type { Dimensions, ReferenceCalibration } from "@/types";

/** Props for SceneViewer3D component. */
interface SceneViewer3DProps {
  /** Current project ID. */
  projectId: string;
  /** Room dimensions from reconstruction. */
  dimensions: Dimensions;
  /** Called when calibration + analysis is complete. */
  onCalibrated: () => void;
}

/**
 * 3D preview with manual calibration controls.
 * Full Three.js viewer will be implemented with mesh loading.
 * @param props - Component props.
 * @returns Calibration interface.
 */
export function SceneViewer3D({
  projectId,
  dimensions,
  onCalibrated,
}: SceneViewer3DProps): React.JSX.Element {
  const [pointA, setPointA] = useState({ x: "0", y: "0", z: "0" });
  const [pointB, setPointB] = useState({ x: "1", y: "0", z: "0" });
  const [realDistance, setRealDistance] = useState("0.9");
  const [calibrating, setCalibrating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCalibrate = useCallback(async () => {
    setCalibrating(true);
    setError(null);
    try {
      const calibration: ReferenceCalibration = {
        point_a: [parseFloat(pointA.x), parseFloat(pointA.y), parseFloat(pointA.z)],
        point_b: [parseFloat(pointB.x), parseFloat(pointB.y), parseFloat(pointB.z)],
        real_distance_m: parseFloat(realDistance),
      };
      await calibrateAndAnalyze(projectId, calibration);
      onCalibrated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Calibration failed");
    } finally {
      setCalibrating(false);
    }
  }, [projectId, pointA, pointB, realDistance, onCalibrated]);

  return (
    <div style={styles.container}>
      <div style={styles.viewer}>
        <div style={styles.placeholder}>
          <span style={styles.placeholderText}>3D Scene Preview</span>
          <div style={styles.dimsInfo}>
            <span>{dimensions.width_m.toFixed(1)}m x {dimensions.length_m.toFixed(1)}m</span>
            <span>Ceiling: {dimensions.ceiling_m.toFixed(1)}m</span>
            <span>Area: {dimensions.area_m2.toFixed(1)}m²</span>
          </div>
        </div>
      </div>

      <div style={styles.calibPanel}>
        <h3 style={styles.panelTitle}>Scale Calibration</h3>
        <p style={styles.panelHint}>
          Enter two points on a known object and its real-world distance.
        </p>

        <div style={styles.pointRow}>
          <label style={styles.label}>Point A:</label>
          <input style={styles.input} value={pointA.x} onChange={(e) => setPointA({ ...pointA, x: e.target.value })} placeholder="x" />
          <input style={styles.input} value={pointA.y} onChange={(e) => setPointA({ ...pointA, y: e.target.value })} placeholder="y" />
          <input style={styles.input} value={pointA.z} onChange={(e) => setPointA({ ...pointA, z: e.target.value })} placeholder="z" />
        </div>

        <div style={styles.pointRow}>
          <label style={styles.label}>Point B:</label>
          <input style={styles.input} value={pointB.x} onChange={(e) => setPointB({ ...pointB, x: e.target.value })} placeholder="x" />
          <input style={styles.input} value={pointB.y} onChange={(e) => setPointB({ ...pointB, y: e.target.value })} placeholder="y" />
          <input style={styles.input} value={pointB.z} onChange={(e) => setPointB({ ...pointB, z: e.target.value })} placeholder="z" />
        </div>

        <div style={styles.pointRow}>
          <label style={styles.label}>Real distance (m):</label>
          <input
            style={{ ...styles.input, width: 120 }}
            value={realDistance}
            onChange={(e) => setRealDistance(e.target.value)}
            type="number"
            step="0.01"
            min="0.01"
          />
        </div>

        <button
          style={{
            ...styles.button,
            ...(calibrating ? styles.buttonDisabled : {}),
          }}
          disabled={calibrating}
          onClick={handleCalibrate}
        >
          {calibrating ? "Calibrating & Analyzing..." : "Calibrate & Analyze"}
        </button>

        {error && <div style={styles.error}>{error}</div>}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", gap: 20 },
  viewer: {
    flex: 2,
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
    color: "#555",
  },
  placeholderText: { fontSize: 18, display: "block", marginBottom: 12 },
  dimsInfo: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    fontSize: 13,
    color: "#666",
  },
  calibPanel: {
    flex: 1,
    backgroundColor: "#141414",
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    padding: 20,
  },
  panelTitle: { margin: "0 0 8px", fontSize: 16, color: "#fff" },
  panelHint: { fontSize: 12, color: "#666", margin: "0 0 16px" },
  pointRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  label: { fontSize: 13, color: "#aaa", minWidth: 60 },
  input: {
    width: 60,
    padding: "6px 8px",
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#1a1a1a",
    color: "#e0e0e0",
    fontSize: 13,
  },
  button: {
    width: "100%",
    padding: "10px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#2a6cb0",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.5, cursor: "not-allowed" },
  error: {
    color: "#f87171",
    fontSize: 12,
    marginTop: 8,
    padding: "6px 10px",
    backgroundColor: "#2a1515",
    borderRadius: 6,
  },
};
