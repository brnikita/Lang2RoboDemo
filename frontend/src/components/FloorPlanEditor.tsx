/**
 * 2D floor plan editor — top-down view with zone editing.
 * To be fully implemented in a future iteration.
 */

/** Props for FloorPlanEditor. */
interface FloorPlanEditorProps {
  /** Current project ID. */
  projectId: string;
}

/**
 * Placeholder for 2D floor plan editor.
 * @param props - Component props.
 * @returns Floor plan view placeholder.
 */
export function FloorPlanEditor({ projectId: _projectId }: FloorPlanEditorProps): React.JSX.Element {
  return (
    <div style={{ padding: 20, textAlign: "center", color: "#666" }}>
      <p>Floor Plan Editor — coming in a future update</p>
    </div>
  );
}
