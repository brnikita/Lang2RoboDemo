/**
 * Photo upload component with drag-and-drop support.
 */

import { useState, useCallback, useRef } from "react";
import { uploadPhotos } from "@/api/client.ts";
import type { Dimensions } from "@/types";

/** Props for PhotoUpload component. */
interface PhotoUploadProps {
  /** Called when photos are uploaded and reconstruction starts. */
  onComplete: (projectId: string, dimensions: Dimensions) => void;
}

/**
 * Drag-and-drop photo upload with preview thumbnails.
 * @param props - Component props.
 * @returns Upload interface.
 */
export function PhotoUpload({ onComplete }: PhotoUploadProps): React.JSX.Element {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const images = Array.from(newFiles).filter((f) =>
      f.type.startsWith("image/"),
    );
    setFiles((prev) => [...prev, ...images]);
    setError(null);
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      addFiles(e.dataTransfer.files);
    },
    [addFiles],
  );

  const handleUpload = useCallback(async () => {
    if (files.length < 3) {
      setError("At least 3 photos required");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result = await uploadPhotos(files);
      onComplete(result.project_id, result.dimensions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [files, onComplete]);

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.dropZone,
          ...(dragOver ? styles.dropZoneActive : {}),
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
        <div style={styles.dropText}>
          <span style={styles.dropIcon}>+</span>
          <span>Drop photos here or click to browse</span>
          <span style={styles.hint}>10-30 photos of the room from different angles</span>
        </div>
      </div>

      {files.length > 0 && (
        <div style={styles.thumbnails}>
          {files.map((file, i) => (
            <div key={`${file.name}-${i}`} style={styles.thumb}>
              <img
                src={URL.createObjectURL(file)}
                alt={file.name}
                style={styles.thumbImg}
              />
              <button
                style={styles.thumbRemove}
                onClick={(e) => {
                  e.stopPropagation();
                  removeFile(i);
                }}
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      <div style={styles.footer}>
        <span style={styles.count}>{files.length} photo(s) selected</span>
        <button
          style={{
            ...styles.button,
            ...(uploading || files.length < 3 ? styles.buttonDisabled : {}),
          }}
          disabled={uploading || files.length < 3}
          onClick={handleUpload}
        >
          {uploading ? "Reconstructing..." : "Upload & Reconstruct"}
        </button>
      </div>

      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", flexDirection: "column", gap: 16 },
  dropZone: {
    border: "2px dashed #333",
    borderRadius: 12,
    padding: 48,
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.2s",
    backgroundColor: "#141414",
  },
  dropZoneActive: {
    borderColor: "#2a6cb0",
    backgroundColor: "#1a2a3a",
  },
  dropText: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 8,
    color: "#888",
  },
  dropIcon: { fontSize: 32, color: "#555" },
  hint: { fontSize: 12, color: "#555" },
  thumbnails: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
  },
  thumb: {
    position: "relative",
    width: 80,
    height: 80,
    borderRadius: 6,
    overflow: "hidden",
  },
  thumbImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  thumbRemove: {
    position: "absolute",
    top: 2,
    right: 2,
    width: 18,
    height: 18,
    border: "none",
    borderRadius: "50%",
    backgroundColor: "rgba(0,0,0,0.7)",
    color: "#fff",
    fontSize: 10,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  footer: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  count: { color: "#888", fontSize: 13 },
  button: {
    padding: "10px 24px",
    borderRadius: 8,
    border: "none",
    backgroundColor: "#2a6cb0",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
  error: {
    color: "#f87171",
    fontSize: 13,
    padding: "8px 12px",
    backgroundColor: "#2a1515",
    borderRadius: 6,
  },
};
