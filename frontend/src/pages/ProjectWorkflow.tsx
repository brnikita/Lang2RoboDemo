/**
 * Project workflow page — renders the appropriate step based on URL.
 */

import { useCallback, useEffect, useRef } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { useProjectState } from "@/hooks/useProjectState";
import { PhotoUpload } from "@/components/PhotoUpload";
import { SceneViewer3D } from "@/components/SceneViewer3D";
import { RecommendationView } from "@/components/RecommendationView";
import { SimulationPlayer } from "@/components/SimulationPlayer";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import type {
  Dimensions,
  IterationLog,
  PipelinePhase,
  Recommendation,
  SimResult,
} from "@/types";

/** URL step segments. */
type Step = "upload" | "calibrate" | "recommend" | "simulate" | "results";

/** Step metadata for the progress indicator. */
const STEPS: { key: Step; label: string }[] = [
  { key: "upload", label: "Upload Photos" },
  { key: "calibrate", label: "Calibrate" },
  { key: "recommend", label: "Plan" },
  { key: "simulate", label: "Simulate" },
  { key: "results", label: "Results" },
];

/** Ordered phases for guard logic. */
const PHASE_ORDER: PipelinePhase[] = [
  "upload",
  "calibrate",
  "recommend",
  "build-scene",
  "simulate",
  "iterate",
];

/** Map URL step to the minimum required pipeline phase. */
const STEP_MIN_PHASE: Record<Step, PipelinePhase> = {
  upload: "upload",
  calibrate: "upload",
  recommend: "calibrate",
  simulate: "recommend",
  results: "simulate",
};

/** Map pipeline phase to the furthest reachable URL step. */
const PHASE_TO_STEP: Record<PipelinePhase, Step> = {
  upload: "calibrate",
  calibrate: "calibrate",
  recommend: "simulate",
  "build-scene": "simulate",
  simulate: "results",
  iterate: "results",
};

/**
 * Check if a phase has been reached given the current project phase.
 * @param current - Current pipeline phase.
 * @param required - Required pipeline phase.
 * @returns True if current >= required in phase order.
 */
function phaseReached(current: PipelinePhase, required: PipelinePhase): boolean {
  return PHASE_ORDER.indexOf(current) >= PHASE_ORDER.indexOf(required);
}

/**
 * Project workflow page with step-based navigation.
 * @returns Workflow element for the current step.
 */
export function ProjectWorkflow(): React.JSX.Element {
  const { projectId, step } = useParams<{
    projectId: string;
    step: string;
  }>();
  const navigate = useNavigate();
  const currentStep = (step ?? "upload") as Step;
  const selfNavigating = useRef(false);

  const {
    status,
    dimensions,
    simResult,
    iterationHistory,
    loading,
    error,
    refresh,
  } = useProjectState(projectId ?? null);

  useEffect(() => {
    if (!status || loading) return;
    if (selfNavigating.current) {
      selfNavigating.current = false;
      return;
    }
    const minPhase = STEP_MIN_PHASE[currentStep];
    if (!phaseReached(status.current_phase, minPhase)) {
      const allowed = PHASE_TO_STEP[status.current_phase];
      navigate(`/projects/${projectId}/${allowed}`, { replace: true });
    }
  }, [status, loading, currentStep, projectId, navigate]);

  const handleUploadComplete = useCallback(
    (id: string, _dims: Dimensions) => {
      selfNavigating.current = true;
      navigate(`/projects/${id}/calibrate`);
    },
    [navigate],
  );

  const handleCalibrationComplete = useCallback(() => {
    selfNavigating.current = true;
    refresh();
    navigate(`/projects/${projectId}/recommend`);
  }, [navigate, projectId, refresh]);

  const handleRecommendationComplete = useCallback(
    (_rec: Recommendation) => {
      selfNavigating.current = true;
      refresh();
      navigate(`/projects/${projectId}/simulate`);
    },
    [navigate, projectId, refresh],
  );

  const handleSimulationComplete = useCallback(
    (_result: SimResult) => {
      selfNavigating.current = true;
      refresh();
      navigate(`/projects/${projectId}/results`);
    },
    [navigate, projectId, refresh],
  );

  const handleIterationComplete = useCallback(
    (_result: SimResult, _history: IterationLog[]) => {
      refresh();
    },
    [],
  );

  if (loading) {
    return <p style={{ color: "#888", textAlign: "center" }}>Loading project...</p>;
  }

  if (error) {
    return <p style={{ color: "#f87171", textAlign: "center" }}>{error}</p>;
  }

  return (
    <div>
      <StepNav currentStep={currentStep} projectId={projectId!} currentPhase={status?.current_phase ?? "upload"} />

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
    </div>
  );
}

/**
 * Step progress navigation bar.
 * @param props - Current step, project ID, and pipeline phase.
 * @returns Navigation bar element.
 */
function StepNav({
  currentStep,
  projectId,
  currentPhase,
}: {
  currentStep: Step;
  projectId: string;
  currentPhase: PipelinePhase;
}): React.JSX.Element {
  const currentIdx = STEPS.findIndex((s) => s.key === currentStep);

  return (
    <nav style={styles.steps}>
      {STEPS.map((step, i) => {
        const minPhase = STEP_MIN_PHASE[step.key];
        const reachable = phaseReached(currentPhase, minPhase);
        const isActive = currentStep === step.key;
        const isDone = currentIdx > i;

        const itemStyle: React.CSSProperties = {
          ...styles.stepItem,
          ...(isActive ? styles.stepActive : {}),
          ...(isDone ? styles.stepDone : {}),
          ...(reachable && !isActive ? { cursor: "pointer" } : {}),
        };

        if (reachable && !isActive) {
          return (
            <Link
              key={step.key}
              to={`/projects/${projectId}/${step.key}`}
              style={itemStyle}
            >
              <span style={styles.stepNumber}>{i + 1}</span>
              <span>{step.label}</span>
            </Link>
          );
        }

        return (
          <div key={step.key} style={itemStyle}>
            <span style={styles.stepNumber}>{i + 1}</span>
            <span>{step.label}</span>
          </div>
        );
      })}
    </nav>
  );
}

const styles: Record<string, React.CSSProperties> = {
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
    textDecoration: "none",
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
};
