/**
 * Hook for loading project state from the backend.
 */

import { useEffect, useState } from "react";

import { getProject } from "@/api/client";
import type {
  Dimensions,
  IterationLog,
  ProjectStatus,
  Recommendation,
  SimResult,
} from "@/types";

/** State returned by useProjectState. */
export interface ProjectState {
  /** Project status metadata. */
  status: ProjectStatus | null;
  /** Room dimensions from reconstruction. */
  dimensions: Dimensions | null;
  /** Generated automation plan. */
  recommendation: Recommendation | null;
  /** Latest simulation result. */
  simResult: SimResult | null;
  /** Iteration improvement history. */
  iterationHistory: IterationLog[];
  /** Whether data is being loaded. */
  loading: boolean;
  /** Error message if load failed. */
  error: string | null;
}

/**
 * Load project data from the backend for state restoration.
 * @param projectId - Project identifier (null skips loading).
 * @returns Project state with loading/error indicators.
 */
export function useProjectState(projectId: string | null): ProjectState {
  const [state, setState] = useState<ProjectState>({
    status: null,
    dimensions: null,
    recommendation: null,
    simResult: null,
    iterationHistory: [],
    loading: true,
    error: null,
  });

  useEffect(() => {
    if (!projectId) {
      setState((prev) => ({ ...prev, loading: false }));
      return;
    }

    let cancelled = false;
    setState((prev) => ({ ...prev, loading: true, error: null }));

    getProject(projectId)
      .then((detail) => {
        if (cancelled) return;
        setState({
          status: detail.status,
          dimensions: detail.dimensions,
          recommendation: detail.recommendation,
          simResult: detail.sim_result,
          iterationHistory: detail.iteration_history,
          loading: false,
          error: null,
        });
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setState((prev) => ({
          ...prev,
          loading: false,
          error: err.message,
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [projectId]);

  return state;
}
