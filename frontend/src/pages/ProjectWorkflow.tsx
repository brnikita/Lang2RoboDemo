/**
 * Project workflow page — renders the appropriate step based on URL.
 */

import { useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { useProjectState } from "@/hooks/useProjectState";
import { useStepNavigation } from "@/hooks/useStepNavigation";
import { PhotoUpload } from "@/components/PhotoUpload";
import { SceneViewer3D } from "@/components/SceneViewer3D";
import { RecommendationView } from "@/components/RecommendationView";
import { SimulationPlayer } from "@/components/SimulationPlayer";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import type { PipelinePhase } from "@/types";

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
  "upload", "calibrate", "recommend", "build-scene", "simulate", "iterate",
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
  const { projectId, step } = useParams<{ projectId: string; step: string }>();
  const navigate = useNavigate();
  const currentStep = (step ?? "upload") as Step;

  const { status, dimensions, simResult, iterationHistory, loading, error, refresh } =
    useProjectState(projectId ?? null);

  const nav = useStepNavigation(projectId, refresh);

  useStepGuard(status, loading, currentStep, projectId, nav.selfNavigatingRef, navigate);

  if (loading) {
    return <p style={{ color: "#888", textAlign: "center" }}>Loading project...</p>;
  }
  if (error) {
    return <p style={{ color: "#f87171", textAlign: "center" }}>{error}</p>;
  }

  return (
    <div>
      <StepNav currentStep={currentStep} projectId={projectId!} currentPhase={status?.current_phase ?? "upload"} />
      <StepContent
        currentStep={currentStep}
        projectId={projectId}
        dimensions={dimensions}
        simResult={simResult}
        iterationHistory={iterationHistory}
        nav={nav}
      />
    </div>
  );
}

/**
 * Guard hook that redirects if user navigates beyond their current phase.
 * @param status - Project status.
 * @param loading - Whether project data is loading.
 * @param currentStep - Current URL step.
 * @param projectId - Project ID.
 * @param selfNavigatingRef - Ref to skip guard on self-navigation.
 * @param navigate - Router navigate function.
 */
function useStepGuard(
  status: { current_phase: PipelinePhase } | null,
  loading: boolean,
  currentStep: Step,
  projectId: string | undefined,
  selfNavigatingRef: React.RefObject<boolean>,
  navigate: ReturnType<typeof useNavigate>,
): void {
  useEffect(() => {
    if (!status || loading) return;
    if (selfNavigatingRef.current) {
      selfNavigatingRef.current = false;
      return;
    }
    const minPhase = STEP_MIN_PHASE[currentStep];
    if (!phaseReached(status.current_phase, minPhase)) {
      const allowed = PHASE_TO_STEP[status.current_phase];
      navigate(`/projects/${projectId}/${allowed}`, { replace: true });
    }
  }, [status, loading, currentStep, projectId, selfNavigatingRef, navigate]);
}

/**
 * Render the component for the current workflow step.
 * @param props - Step content props.
 * @returns Step component element.
 */
function StepContent({
  currentStep,
  projectId,
  dimensions,
  simResult,
  iterationHistory,
  nav,
}: {
  currentStep: Step;
  projectId: string | undefined;
  dimensions: import("@/types").Dimensions | null;
  simResult: import("@/types").SimResult | null;
  iterationHistory: import("@/types").IterationLog[];
  nav: ReturnType<typeof useStepNavigation>;
}): React.JSX.Element | null {
  if (currentStep === "upload") {
    return <PhotoUpload onComplete={nav.onUploadComplete} />;
  }
  if (currentStep === "calibrate" && projectId && dimensions) {
    return <SceneViewer3D projectId={projectId} dimensions={dimensions} onCalibrated={nav.onCalibrationComplete} />;
  }
  if (currentStep === "recommend" && projectId) {
    return <RecommendationView projectId={projectId} onConfirm={nav.onRecommendationComplete} />;
  }
  if (currentStep === "simulate" && projectId) {
    return <SimulationPlayer projectId={projectId} onComplete={nav.onSimulationComplete} />;
  }
  if (currentStep === "results" && simResult && projectId) {
    return <MetricsDashboard projectId={projectId} result={simResult} history={iterationHistory} onIterate={nav.onIterationComplete} />;
  }
  return null;
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
      {STEPS.map((s, i) => (
        <StepNavItem
          key={s.key}
          step={s}
          index={i}
          isActive={currentStep === s.key}
          isDone={currentIdx > i}
          reachable={phaseReached(currentPhase, STEP_MIN_PHASE[s.key])}
          projectId={projectId}
        />
      ))}
    </nav>
  );
}

/**
 * Single step item in the navigation bar.
 * @param props - Step item props.
 * @returns Step nav item element.
 */
function StepNavItem({
  step,
  index,
  isActive,
  isDone,
  reachable,
  projectId,
}: {
  step: { key: Step; label: string };
  index: number;
  isActive: boolean;
  isDone: boolean;
  reachable: boolean;
  projectId: string;
}): React.JSX.Element {
  const itemStyle: React.CSSProperties = {
    ...styles.stepItem,
    ...(isActive ? styles.stepActive : {}),
    ...(isDone ? styles.stepDone : {}),
    ...(reachable && !isActive ? { cursor: "pointer" } : {}),
  };

  const content = (
    <>
      <span style={styles.stepNumber}>{index + 1}</span>
      <span>{step.label}</span>
    </>
  );

  if (reachable && !isActive) {
    return <Link to={`/projects/${projectId}/${step.key}`} style={itemStyle}>{content}</Link>;
  }
  return <div style={itemStyle}>{content}</div>;
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
