/**
 * API client for Lang2Robo backend.
 * All backend communication goes through these typed functions.
 */

import type {
  Dimensions,
  IterationLog,
  Recommendation,
  ReferenceCalibration,
  SimResult,
  SpaceModel,
} from "@/types";

const BASE_URL = "/api";

/**
 * Generic fetch wrapper with error handling.
 * @param url - Endpoint URL.
 * @param options - Fetch options.
 * @returns Parsed JSON response.
 * @throws Error with message from backend.
 */
async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  return response.json() as Promise<T>;
}

/** Response from photo upload endpoint. */
interface UploadResponse {
  project_id: string;
  status: string;
  dimensions: Dimensions;
}

/**
 * Upload room photos for reconstruction.
 * @param photos - Array of image files.
 * @returns Project ID and initial dimensions.
 */
export async function uploadPhotos(photos: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  for (const photo of photos) {
    formData.append("photos", photo);
  }

  const response = await fetch(`${BASE_URL}/capture`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }

  return response.json() as Promise<UploadResponse>;
}

/**
 * Calibrate scale and run scene analysis.
 * @param projectId - Project identifier.
 * @param calibration - Two points + real distance.
 * @returns Complete SpaceModel.
 */
export async function calibrateAndAnalyze(
  projectId: string,
  calibration: ReferenceCalibration,
): Promise<SpaceModel> {
  return apiFetch<SpaceModel>(`/capture/${projectId}/calibrate`, {
    method: "POST",
    body: JSON.stringify(calibration),
  });
}

/** Request body for recommendation generation. */
interface RecommendRequest {
  project_id: string;
  scenario: string;
}

/**
 * Generate automation recommendation.
 * @param projectId - Project identifier.
 * @param scenario - User's scenario text.
 * @returns Generated recommendation.
 */
export async function generateRecommendation(
  projectId: string,
  scenario: string,
): Promise<Recommendation> {
  return apiFetch<Recommendation>("/recommend", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId, scenario } satisfies RecommendRequest),
  });
}

/** Response from scene build endpoint. */
interface BuildSceneResponse {
  scene_path: string;
  valid: boolean;
  equipment_count: number;
  work_object_count: number;
}

/**
 * Build MJCF scene from recommendation.
 * @param projectId - Project identifier.
 * @returns Scene build result.
 */
export async function buildScene(projectId: string): Promise<BuildSceneResponse> {
  return apiFetch<BuildSceneResponse>(`/projects/${projectId}/build-scene`, {
    method: "POST",
  });
}

/**
 * Run simulation on the latest scene.
 * @param projectId - Project identifier.
 * @returns Simulation result with metrics.
 */
export async function runSimulation(projectId: string): Promise<SimResult> {
  return apiFetch<SimResult>(`/projects/${projectId}/simulate`, {
    method: "POST",
  });
}

/**
 * Launch MuJoCo interactive viewer for the scene.
 * @param projectId - Project identifier.
 * @returns Status.
 */
export async function launchViewer(projectId: string): Promise<{ status: string }> {
  return apiFetch<{ status: string }>(`/projects/${projectId}/view`, {
    method: "POST",
  });
}

/** Response from iteration loop endpoint. */
interface IterateResponse {
  result: SimResult;
  history: IterationLog[];
  iterations_run: number;
  converged: boolean;
}

/**
 * Run iteration improvement loop.
 * @param projectId - Project identifier.
 * @param maxIterations - Maximum iterations.
 * @returns Final result and iteration history.
 */
export async function runIteration(
  projectId: string,
  maxIterations: number = 5,
): Promise<IterateResponse> {
  return apiFetch<IterateResponse>(`/projects/${projectId}/iterate`, {
    method: "POST",
    body: JSON.stringify({ max_iterations: maxIterations }),
  });
}
