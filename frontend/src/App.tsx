/**
 * Root application component with step-based workflow.
 */

import { useState, useCallback } from "react";
import { PhotoUpload } from "./components/PhotoUpload.tsx";
import { SceneViewer3D } from "./components/SceneViewer3D.tsx";
import { RecommendationView } from "./components/RecommendationView.tsx";
import { SimulationPlayer } from "./components/SimulationPlayer.tsx";
import { MetricsDashboard } from "./components/MetricsDashboard.tsx";
import type { Dimensions, Recommendation, SimResult, IterationLog } from "./types";

/** Application workflow steps. */
type Step = "upload" | "calibrate" | "recommend" | "simulate" | "results";

/** All step labels for the progress indicator. */
const STEPS: { key: Step; label: string }[] = [
  { key: "upload", label: "Upload Photos" },
  { key: "calibrate", label: "Calibrate" },
  { key: "recommend", label: "Plan" },
  { key: "simulate", label: "Simulate" },
  { key: "results", label: "Results" },
];

export function App(): React.JSX.Element {
  const [currentStep, setCurrentStep] = useState<Step>("upload");
  const [projectId, setProjectId] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [_recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [simResult, setSimResult] = useState<SimResult | null>(null);
  const [iterationHistory, setIterationHistory] = useState<IterationLog[]>([]);

  const handleUploadComplete = useCallback((id: string, dims: Dimensions) => {
    setProjectId(id);
    setDimensions(dims);
    setCurrentStep("calibrate");
  }, []);

  const handleCalibrationComplete = useCallback(() => {
    setCurrentStep("recommend");
  }, []);

  const handleRecommendationComplete = useCallback((rec: Recommendation) => {
    setRecommendation(rec);
    setCurrentStep("simulate");
  }, []);

  const handleSimulationComplete = useCallback((result: SimResult) => {
    setSimResult(result);
    setCurrentStep("results");
  }, []);

  const handleIterationComplete = useCallback(
    (result: SimResult, history: IterationLog[]) => {
      setSimResult(result);
      setIterationHistory(history);
    },
    [],
  );

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Lang2Robo</h1>
        <p style={styles.subtitle}>Text → Simulation → Optimization</p>
      </header>

      <nav style={styles.steps}>
        {STEPS.map((step, i) => (
          <div
            key={step.key}
            style={{
              ...styles.stepItem,
              ...(currentStep === step.key ? styles.stepActive : {}),
              ...(STEPS.findIndex((s) => s.key === currentStep) > i
                ? styles.stepDone
                : {}),
            }}
          >
            <span style={styles.stepNumber}>{i + 1}</span>
            <span>{step.label}</span>
          </div>
        ))}
      </nav>

      <main style={styles.main}>
        {currentStep === "upload" && (
          <PhotoUpload onComplete={handleUploadComplete} />
        )}

        {currentStep === "calibrate" && projectId && dimensions && (
          <SceneViewer3D
            projectId={projectId}
            dimensions={dimensions}
            onCalibrated={handleCalibrationComplete}
          />
        )}

        {currentStep === "recommend" && projectId && (
          <RecommendationView
            projectId={projectId}
            onConfirm={handleRecommendationComplete}
          />
        )}

        {currentStep === "simulate" && projectId && (
          <SimulationPlayer
            projectId={projectId}
            onComplete={handleSimulationComplete}
          />
        )}

        {currentStep === "results" && simResult && projectId && (
          <MetricsDashboard
            projectId={projectId}
            result={simResult}
            history={iterationHistory}
            onIterate={handleIterationComplete}
          />
        )}
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 1200,
    margin: "0 auto",
    padding: "20px",
    fontFamily: "'Inter', -apple-system, sans-serif",
    color: "#e0e0e0",
    backgroundColor: "#0f0f0f",
    minHeight: "100vh",
  },
  header: {
    textAlign: "center",
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    color: "#ffffff",
    margin: 0,
    letterSpacing: "-0.02em",
  },
  subtitle: {
    fontSize: 14,
    color: "#888",
    margin: "4px 0 0",
  },
  steps: {
    display: "flex",
    justifyContent: "center",
    gap: 8,
    marginBottom: 32,
  },
  stepItem: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 16px",
    borderRadius: 8,
    fontSize: 13,
    color: "#666",
    backgroundColor: "#1a1a1a",
    border: "1px solid #2a2a2a",
    transition: "all 0.2s",
  },
  stepActive: {
    color: "#fff",
    backgroundColor: "#1a3a5c",
    borderColor: "#2a6cb0",
  },
  stepDone: {
    color: "#4ade80",
    borderColor: "#2a5a3a",
  },
  stepNumber: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: 20,
    height: 20,
    borderRadius: "50%",
    backgroundColor: "#2a2a2a",
    fontSize: 11,
    fontWeight: 600,
  },
  main: {
    minHeight: 400,
  },
};
