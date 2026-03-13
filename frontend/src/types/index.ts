/**
 * TypeScript types mirroring backend Pydantic models.
 * These are the shared contracts between frontend and backend.
 */

// ── Space Models ──

/** Calibration of reconstruction scale using a known measurement. */
export interface ReferenceCalibration {
  /** First point in mesh coordinates [x, y, z]. */
  point_a: [number, number, number];
  /** Second point in mesh coordinates [x, y, z]. */
  point_b: [number, number, number];
  /** Real distance between the two points in meters. */
  real_distance_m: number;
}

/** Room dimensions from calibrated reconstruction. */
export interface Dimensions {
  width_m: number;
  length_m: number;
  ceiling_m: number;
  area_m2: number;
}

/** Functional zone within the room. */
export interface Zone {
  name: string;
  /** 2D contour in meters — list of [x, y] points. */
  polygon: [number, number][];
  area_m2: number;
}

/** Door detected in the room. */
export interface Door {
  position: [number, number];
  width_m: number;
}

/** Window detected in the room. */
export interface Window {
  position: [number, number];
  width_m: number;
}

/** Equipment already present in the room. */
export interface ExistingEquipment {
  name: string;
  category: string;
  position: [number, number, number];
  /** Detection confidence 0.0–1.0. */
  confidence: number;
}

/** Result of DISCOVERSE Real2Sim reconstruction. */
export interface SceneReconstruction {
  mesh_path: string;
  mjcf_path: string;
  pointcloud_path: string;
  dimensions: Dimensions;
}

/** Result of Claude Vision scene analysis. */
export interface SceneAnalysis {
  zones: Zone[];
  existing_equipment: ExistingEquipment[];
  doors: Door[];
  windows: Window[];
}

/** Complete room model for simulation. */
export interface SpaceModel {
  dimensions: Dimensions;
  zones: Zone[];
  existing_equipment: ExistingEquipment[];
  doors: Door[];
  windows: Window[];
  reconstruction: SceneReconstruction;
}

// ── Equipment Models ──

/** Source of MJCF/URDF model. */
export interface MjcfSource {
  menagerie_id: string | null;
  robot_descriptions_id: string | null;
  urdf_url: string | null;
}

/** Placement constraints. */
export interface PlacementRules {
  min_zone_m2: number;
  constraints: Record<string, string>;
}

/** Equipment catalog entry. */
export interface EquipmentEntry {
  id: string;
  name: string;
  type: "manipulator" | "conveyor" | "camera" | "fixture";
  specs: Record<string, number | string>;
  mjcf_source: MjcfSource;
  price_usd: number | null;
  purchase_url: string | null;
  placement_rules: PlacementRules | null;
}

// ── Recommendation Models ──

/** Placement of new equipment in the scene. */
export interface EquipmentPlacement {
  equipment_id: string;
  position: [number, number, number];
  orientation_deg: number;
  purpose: string;
  zone: string;
}

/** Object for manipulation in simulation. */
export interface WorkObject {
  name: string;
  shape: "box" | "cylinder" | "sphere";
  size: [number, number, number];
  mass_kg: number;
  position: [number, number, number];
  count: number;
}

/** A single workflow step. */
export interface WorkflowStep {
  order: number;
  action: "pick" | "place" | "move" | "transport" | "inspect" | "wait";
  equipment_id: string | null;
  target: string;
  duration_s: number;
  params: Record<string, number | string> | null;
}

/** Expected performance metrics. */
export interface ExpectedMetrics {
  cycle_time_s: number;
  throughput_per_hour: number;
  notes: string;
}

/** Complete automation plan from Claude. */
export interface Recommendation {
  equipment: EquipmentPlacement[];
  work_objects: WorkObject[];
  target_positions: Record<string, [number, number, number]>;
  workflow_steps: WorkflowStep[];
  expected_metrics: ExpectedMetrics;
  text_plan: string;
}

// ── Simulation Models ──

/** Result of a single simulation step. */
export interface StepResult {
  success: boolean;
  duration_s: number;
  collision_count: number;
  error: string | null;
}

/** Aggregate simulation metrics. */
export interface SimMetrics {
  cycle_time_s: number;
  /** 0.0–1.0 */
  success_rate: number;
  collision_count: number;
  failed_steps: number[];
}

/** Complete simulation result. */
export interface SimResult {
  steps: StepResult[];
  metrics: SimMetrics;
}

// ── Iteration Models ──

/** Change to equipment position. */
export interface PositionChange {
  equipment_id: string;
  new_position: [number, number, number];
  new_orientation_deg: number | null;
}

/** Equipment replacement. */
export interface EquipmentReplacement {
  old_equipment_id: string;
  new_equipment_id: string;
  reason: string;
}

/** Scene corrections from Claude. */
export interface SceneCorrections {
  position_changes: PositionChange[] | null;
  add_equipment: EquipmentPlacement[] | null;
  remove_equipment: string[] | null;
  replace_equipment: EquipmentReplacement[] | null;
  workflow_changes: WorkflowStep[] | null;
}

/** Log entry for one iteration. */
export interface IterationLog {
  iteration: number;
  metrics: SimMetrics;
  corrections_applied: SceneCorrections;
}
