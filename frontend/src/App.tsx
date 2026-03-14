/**
 * Root application component — router configuration.
 */

import { Routes, Route } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { Dashboard } from "@/pages/Dashboard";
import { ProjectWorkflow } from "@/pages/ProjectWorkflow";

/**
 * Application root with route definitions.
 * @returns Router element.
 */
export function App(): React.JSX.Element {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="new" element={<ProjectWorkflow />} />
        <Route
          path="projects/:projectId/:step"
          element={<ProjectWorkflow />}
        />
      </Route>
    </Routes>
  );
}
