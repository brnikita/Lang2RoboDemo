/**
 * Hook for step-based navigation callbacks in the project workflow.
 */

import { useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";

import type { Dimensions, IterationLog, Recommendation, SimResult } from "@/types";

/** Callbacks returned by useStepNavigation. */
export interface StepNavigationCallbacks {
  /** Called after photo upload completes. */
  onUploadComplete: (id: string, dims: Dimensions) => void;
  /** Called after calibration completes. */
  onCalibrationComplete: () => void;
  /** Called after recommendation is confirmed. */
  onRecommendationComplete: (rec: Recommendation) => void;
  /** Called after simulation finishes. */
  onSimulationComplete: (result: SimResult) => void;
  /** Called after iteration loop finishes. */
  onIterationComplete: (result: SimResult, history: IterationLog[]) => void;
  /** Ref indicating a self-initiated navigation is in progress. */
  selfNavigatingRef: React.RefObject<boolean>;
}

/**
 * Create memoized navigation callbacks for each workflow step.
 * @param projectId - Current project ID.
 * @param refresh - Function to re-fetch project state.
 * @returns Step navigation callbacks and selfNavigatingRef ref.
 */
export function useStepNavigation(
  projectId: string | undefined,
  refresh: () => void,
): StepNavigationCallbacks {
  const navigate = useNavigate();
  const selfNavigatingRef = useRef(false);

  const onUploadComplete = useCallback(
    (id: string, _dims: Dimensions) => {
      selfNavigatingRef.current = true;
      navigate(`/projects/${id}/calibrate`);
    },
    [navigate],
  );

  const onCalibrationComplete = useCallback(() => {
    selfNavigatingRef.current = true;
    refresh();
    navigate(`/projects/${projectId}/recommend`);
  }, [navigate, projectId, refresh]);

  const onRecommendationComplete = useCallback(
    (_rec: Recommendation) => {
      selfNavigatingRef.current = true;
      refresh();
      navigate(`/projects/${projectId}/simulate`);
    },
    [navigate, projectId, refresh],
  );

  const onSimulationComplete = useCallback(
    (_result: SimResult) => {
      selfNavigatingRef.current = true;
      refresh();
      navigate(`/projects/${projectId}/results`);
    },
    [navigate, projectId, refresh],
  );

  const onIterationComplete = useCallback(
    (_result: SimResult, _history: IterationLog[]) => {
      refresh();
    },
    [refresh],
  );

  return {
    onUploadComplete,
    onCalibrationComplete,
    onRecommendationComplete,
    onSimulationComplete,
    onIterationComplete,
    selfNavigatingRef,
  };
}
