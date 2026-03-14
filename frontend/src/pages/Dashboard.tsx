/**
 * Dashboard page — lists all projects with status cards.
 */

import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { listProjects } from "@/api/client";
import type { PipelinePhase, ProjectStatus } from "@/types";

/** Map pipeline phase to the corresponding URL step segment. */
const PHASE_TO_STEP: Record<PipelinePhase, string> = {
  upload: "calibrate",
  calibrate: "calibrate",
  recommend: "simulate",
  "build-scene": "simulate",
  simulate: "results",
  iterate: "results",
};

/** Human-readable phase labels. */
const PHASE_LABELS: Record<PipelinePhase, string> = {
  upload: "Uploaded",
  calibrate: "Calibrated",
  recommend: "Planned",
  "build-scene": "Scene Built",
  simulate: "Simulated",
  iterate: "Optimized",
};

/**
 * Project dashboard with status cards and new project button.
 * @returns Dashboard element.
 */
export function Dashboard(): React.JSX.Element {
  const [projects, setProjects] = useState<ProjectStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div style={styles.toolbar}>
        <h2 style={styles.heading}>Projects</h2>
        <button
          style={styles.newButton}
          onClick={() => navigate("/new")}
        >
          + New Project
        </button>
      </div>

      {loading && <p style={styles.status}>Loading projects...</p>}
      {error && <p style={styles.error}>{error}</p>}

      {!loading && projects.length === 0 && (
        <p style={styles.status}>
          No projects yet. Click "New Project" to start.
        </p>
      )}

      <div style={styles.grid}>
        {projects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>
    </div>
  );
}

/**
 * Single project status card.
 * @param props - Project status.
 * @returns Card element linking to the project workflow.
 */
function ProjectCard({
  project,
}: {
  project: ProjectStatus;
}): React.JSX.Element {
  const step = PHASE_TO_STEP[project.current_phase];
  const label = PHASE_LABELS[project.current_phase];
  const shortId = project.id.slice(0, 8);
  const updated = new Date(project.updated_at).toLocaleDateString();

  return (
    <Link
      to={`/projects/${project.id}/${step}`}
      style={styles.card}
    >
      <div style={styles.cardHeader}>
        <span style={styles.cardName}>
          {project.name || `Project ${shortId}`}
        </span>
        <span style={styles.badge}>{label}</span>
      </div>
      <div style={styles.cardMeta}>
        <span style={styles.cardId}>{shortId}</span>
        <span style={styles.cardDate}>{updated}</span>
      </div>
    </Link>
  );
}

const styles: Record<string, React.CSSProperties> = {
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 24,
  },
  heading: {
    fontSize: 20,
    fontWeight: 600,
    color: "#fff",
    margin: 0,
  },
  newButton: {
    padding: "10px 20px",
    fontSize: 14,
    fontWeight: 600,
    color: "#fff",
    backgroundColor: "#2a6cb0",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
  },
  status: {
    color: "#888",
    textAlign: "center",
    padding: "40px 0",
  },
  error: {
    color: "#f87171",
    textAlign: "center",
    padding: "40px 0",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: 16,
  },
  card: {
    display: "block",
    padding: 16,
    backgroundColor: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 10,
    textDecoration: "none",
    color: "inherit",
    transition: "border-color 0.2s",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  cardName: {
    fontSize: 15,
    fontWeight: 600,
    color: "#fff",
  },
  badge: {
    fontSize: 11,
    fontWeight: 600,
    padding: "3px 8px",
    borderRadius: 6,
    backgroundColor: "#1a3a5c",
    color: "#7db8f0",
  },
  cardMeta: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    color: "#666",
  },
  cardId: {
    fontFamily: "monospace",
  },
  cardDate: {},
};
