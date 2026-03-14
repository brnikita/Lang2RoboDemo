/**
 * Shared layout wrapper with header and navigation.
 */

import { Link, Outlet } from "react-router-dom";

/**
 * Root layout with header and back-to-dashboard link.
 * @returns Layout element wrapping child routes via Outlet.
 */
export function Layout(): React.JSX.Element {
  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <Link to="/" style={styles.titleLink}>
          <h1 style={styles.title}>Lang2Robo</h1>
        </Link>
        <p style={styles.subtitle}>Text → Simulation → Optimization</p>
      </header>
      <main style={styles.main}>
        <Outlet />
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
    minHeight: "100vh",
  },
  header: {
    textAlign: "center",
    marginBottom: 24,
  },
  titleLink: {
    textDecoration: "none",
    color: "inherit",
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
  main: {
    minHeight: 400,
  },
};
