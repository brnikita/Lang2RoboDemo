/**
 * 3D scene viewer with point cloud rendering and calibration controls.
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { calibrateAndAnalyze } from "@/api/client.ts";
import type { Dimensions, ReferenceCalibration } from "@/types";

/** Props for SceneViewer3D component. */
interface SceneViewer3DProps {
  projectId: string;
  dimensions: Dimensions;
  onCalibrated: () => void;
}

/**
 * 3D viewer showing reconstructed point cloud + calibration panel.
 * @param props - Component props.
 * @returns Scene viewer with calibration.
 */
export function SceneViewer3D({
  projectId,
  dimensions,
  onCalibrated,
}: SceneViewer3DProps): React.JSX.Element {
  const [pointA, setPointA] = useState({ x: "0", y: "0", z: "0" });
  const [pointB, setPointB] = useState({ x: "1", y: "0", z: "0" });
  const [realDistance, setRealDistance] = useState("0.9");
  const [calibrating, setCalibrating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pointCloudUrl, setPointCloudUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    setPointCloudUrl(`/api/capture/${projectId}/pointcloud`);
  }, [projectId]);

  const handleCalibrate = useCallback(async () => {
    setCalibrating(true);
    setError(null);
    try {
      const calibration: ReferenceCalibration = {
        point_a: [parseFloat(pointA.x), parseFloat(pointA.y), parseFloat(pointA.z)],
        point_b: [parseFloat(pointB.x), parseFloat(pointB.y), parseFloat(pointB.z)],
        real_distance_m: parseFloat(realDistance),
      };
      await calibrateAndAnalyze(projectId, calibration);
      onCalibrated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Calibration failed");
    } finally {
      setCalibrating(false);
    }
  }, [projectId, pointA, pointB, realDistance, onCalibrated]);

  return (
    <div style={styles.container}>
      <div style={styles.viewer}>
        <Canvas
          camera={{ position: [2, 2, 2], fov: 60 }}
          style={{ background: "#0a0a0a" }}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} />
          <gridHelper args={[10, 20, "#333333", "#1a1a1a"]} />
          <axesHelper args={[1]} />
          {pointCloudUrl && (
            <PointCloud url={pointCloudUrl} onError={setLoadError} />
          )}
          <OrbitControls makeDefault />
        </Canvas>
        <div style={styles.viewerOverlay}>
          <span style={styles.dimsBadge}>
            {dimensions.width_m.toFixed(1)}m x {dimensions.length_m.toFixed(1)}m
            | H: {dimensions.ceiling_m.toFixed(1)}m
            | {dimensions.area_m2.toFixed(1)}m²
          </span>
          {loadError && <span style={styles.loadError}>{loadError}</span>}
        </div>
      </div>

      <div style={styles.calibPanel}>
        <h3 style={styles.panelTitle}>Scale Calibration</h3>
        <p style={styles.panelHint}>
          Pick two points on a known object and enter the real distance.
        </p>

        <div style={styles.pointRow}>
          <label style={styles.label}>Point A:</label>
          <input style={styles.input} value={pointA.x} onChange={(e) => setPointA({ ...pointA, x: e.target.value })} placeholder="x" />
          <input style={styles.input} value={pointA.y} onChange={(e) => setPointA({ ...pointA, y: e.target.value })} placeholder="y" />
          <input style={styles.input} value={pointA.z} onChange={(e) => setPointA({ ...pointA, z: e.target.value })} placeholder="z" />
        </div>

        <div style={styles.pointRow}>
          <label style={styles.label}>Point B:</label>
          <input style={styles.input} value={pointB.x} onChange={(e) => setPointB({ ...pointB, x: e.target.value })} placeholder="x" />
          <input style={styles.input} value={pointB.y} onChange={(e) => setPointB({ ...pointB, y: e.target.value })} placeholder="y" />
          <input style={styles.input} value={pointB.z} onChange={(e) => setPointB({ ...pointB, z: e.target.value })} placeholder="z" />
        </div>

        <div style={styles.pointRow}>
          <label style={styles.label}>Distance (m):</label>
          <input
            style={{ ...styles.input, width: 120 }}
            value={realDistance}
            onChange={(e) => setRealDistance(e.target.value)}
            type="number"
            step="0.01"
            min="0.01"
          />
        </div>

        <button
          style={{
            ...styles.button,
            ...(calibrating ? styles.buttonDisabled : {}),
          }}
          disabled={calibrating}
          onClick={handleCalibrate}
        >
          {calibrating ? "Calibrating & Analyzing..." : "Calibrate & Analyze"}
        </button>

        {error && <div style={styles.error}>{error}</div>}
      </div>
    </div>
  );
}

/** Props for the PointCloud sub-component. */
interface PointCloudProps {
  url: string;
  onError: (msg: string | null) => void;
}

/**
 * Three.js component that loads and renders a PLY point cloud.
 * @param props - URL and error callback.
 * @returns Three.js points object.
 */
function PointCloud({ url, onError }: PointCloudProps): React.JSX.Element | null {
  const pointsRef = useRef<THREE.Points>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const { camera } = useThree();

  useEffect(() => {
    let cancelled = false;

    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return res.arrayBuffer();
      })
      .then((buffer) => {
        if (cancelled) return;
        const geom = parsePLY(buffer);
        if (geom.getAttribute("position")?.count === 0) {
          onError("Point cloud is empty");
          return;
        }
        setGeometry(geom);
        onError(null);
        fitCameraToGeometry(camera as THREE.PerspectiveCamera, geom);
      })
      .catch((err) => {
        if (!cancelled) {
          onError(`Could not load point cloud: ${err.message}`);
        }
      });

    return () => { cancelled = true; };
  }, [url, camera, onError]);

  if (!geometry) return null;

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={0.008}
        vertexColors={geometry.getAttribute("color") !== null}
        color={geometry.getAttribute("color") ? undefined : "#6cb0e0"}
        sizeAttenuation
      />
    </points>
  );
}

/**
 * Parse a binary/ASCII PLY file into Three.js BufferGeometry.
 * @param buffer - Raw PLY file data.
 * @returns BufferGeometry with position and optional color attributes.
 */
function parsePLY(buffer: ArrayBuffer): THREE.BufferGeometry {
  const text = new TextDecoder().decode(buffer);
  const headerEnd = text.indexOf("end_header");
  if (headerEnd === -1) {
    return new THREE.BufferGeometry();
  }

  const header = text.slice(0, headerEnd);
  const vertexMatch = header.match(/element vertex (\d+)/);
  const vertexCount = vertexMatch ? parseInt(vertexMatch[1]!, 10) : 0;

  if (vertexCount === 0) {
    return new THREE.BufferGeometry();
  }

  const hasColor = header.includes("property uchar red") || header.includes("property float red");
  const dataStart = headerEnd + "end_header".length + 1;
  const isBinary = header.includes("format binary");

  const positions = new Float32Array(vertexCount * 3);
  const colors = hasColor ? new Float32Array(vertexCount * 3) : null;

  if (isBinary) {
    parseBinaryPLY(buffer, dataStart, vertexCount, header, positions, colors);
  } else {
    parseAsciiPLY(text, dataStart, vertexCount, positions, colors);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (colors) {
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  return geom;
}

/**
 * Parse ASCII PLY vertex data.
 * @param text - Full PLY file as text.
 * @param dataStart - Byte offset after header.
 * @param count - Number of vertices.
 * @param positions - Output position array.
 * @param colors - Output color array (optional).
 */
function parseAsciiPLY(
  text: string,
  dataStart: number,
  count: number,
  positions: Float32Array,
  colors: Float32Array | null,
): void {
  const lines = text.slice(dataStart).trim().split("\n");
  for (let i = 0; i < Math.min(count, lines.length); i++) {
    const parts = lines[i]!.trim().split(/\s+/);
    positions[i * 3] = parseFloat(parts[0] ?? "0");
    positions[i * 3 + 1] = parseFloat(parts[1] ?? "0");
    positions[i * 3 + 2] = parseFloat(parts[2] ?? "0");
    if (colors && parts.length >= 6) {
      colors[i * 3] = parseFloat(parts[3] ?? "0") / 255;
      colors[i * 3 + 1] = parseFloat(parts[4] ?? "0") / 255;
      colors[i * 3 + 2] = parseFloat(parts[5] ?? "0") / 255;
    }
  }
}

/**
 * Parse binary PLY vertex data (little-endian floats + uchar colors).
 * @param buffer - Raw file buffer.
 * @param dataStart - Byte offset after header.
 * @param count - Vertex count.
 * @param header - PLY header for format detection.
 * @param positions - Output positions.
 * @param colors - Output colors (optional).
 */
function parseBinaryPLY(
  buffer: ArrayBuffer,
  dataStart: number,
  count: number,
  header: string,
  positions: Float32Array,
  colors: Float32Array | null,
): void {
  const view = new DataView(buffer);
  const hasNormals = header.includes("property float nx");
  const hasColor = colors !== null;
  const colorIsUchar = header.includes("property uchar red");

  // Compute stride: xyz(12) + normals?(12) + color?(3 or 12) + alpha?(1 or 4)
  let stride = 12; // xyz floats
  if (hasNormals) stride += 12;
  if (hasColor) stride += colorIsUchar ? 3 : 12;
  if (hasColor && (header.includes("property uchar alpha") || header.includes("property float alpha"))) {
    stride += colorIsUchar ? 1 : 4;
  }

  // Find actual header end in bytes (text.indexOf might be off for binary)
  const headerBytes = new TextEncoder().encode(header + "end_header\n").length;
  let offset = headerBytes;
  if (offset > buffer.byteLength) offset = dataStart;

  for (let i = 0; i < count && offset + stride <= buffer.byteLength; i++) {
    positions[i * 3] = view.getFloat32(offset, true);
    positions[i * 3 + 1] = view.getFloat32(offset + 4, true);
    positions[i * 3 + 2] = view.getFloat32(offset + 8, true);

    if (colors) {
      const colorOffset = offset + 12 + (hasNormals ? 12 : 0);
      if (colorIsUchar) {
        colors[i * 3] = view.getUint8(colorOffset) / 255;
        colors[i * 3 + 1] = view.getUint8(colorOffset + 1) / 255;
        colors[i * 3 + 2] = view.getUint8(colorOffset + 2) / 255;
      } else {
        colors[i * 3] = view.getFloat32(colorOffset, true);
        colors[i * 3 + 1] = view.getFloat32(colorOffset + 4, true);
        colors[i * 3 + 2] = view.getFloat32(colorOffset + 8, true);
      }
    }
    offset += stride;
  }
}

/**
 * Adjust camera to fit point cloud in view.
 * @param camera - Three.js perspective camera.
 * @param geometry - Point cloud geometry.
 */
function fitCameraToGeometry(
  camera: THREE.PerspectiveCamera,
  geometry: THREE.BufferGeometry,
): void {
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!sphere) return;

  const center = sphere.center;
  const radius = sphere.radius || 1;
  const dist = radius * 2.5;

  camera.position.set(
    center.x + dist,
    center.y + dist * 0.7,
    center.z + dist,
  );
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", gap: 20, height: 500 },
  viewer: {
    flex: 2,
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    overflow: "hidden",
    position: "relative",
  },
  viewerOverlay: {
    position: "absolute",
    bottom: 8,
    left: 8,
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  dimsBadge: {
    padding: "4px 10px",
    borderRadius: 6,
    backgroundColor: "rgba(0,0,0,0.7)",
    color: "#aaa",
    fontSize: 11,
    fontFamily: "monospace",
  },
  loadError: {
    padding: "4px 10px",
    borderRadius: 6,
    backgroundColor: "rgba(40,10,10,0.9)",
    color: "#f87171",
    fontSize: 11,
  },
  calibPanel: {
    flex: 1,
    backgroundColor: "#141414",
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    padding: 20,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  panelTitle: { margin: 0, fontSize: 16, color: "#fff" },
  panelHint: { fontSize: 12, color: "#666", margin: 0 },
  pointRow: { display: "flex", alignItems: "center", gap: 8 },
  label: { fontSize: 13, color: "#aaa", minWidth: 80 },
  input: {
    width: 60,
    padding: "6px 8px",
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#1a1a1a",
    color: "#e0e0e0",
    fontSize: 13,
  },
  button: {
    width: "100%",
    padding: "10px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#2a6cb0",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.5, cursor: "not-allowed" },
  error: {
    color: "#f87171",
    fontSize: 12,
    padding: "6px 10px",
    backgroundColor: "#2a1515",
    borderRadius: 6,
  },
};
