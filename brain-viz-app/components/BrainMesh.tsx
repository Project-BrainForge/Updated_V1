"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface BrainMeshProps {
  vertices: number[][];
  triangles: number[][];
  regionMapping: number[];
  predictionData: any;
  timePoint: number;
}

export default function BrainMesh({
  vertices,
  triangles,
  regionMapping,
  predictionData,
  timePoint,
}: BrainMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Create geometry from vertices and triangles
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();

    // Flatten vertices array
    const positions = new Float32Array(vertices.flat());
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    // Flatten triangles array (convert to zero-indexed)
    const indices = new Uint32Array(triangles.flat());
    geom.setIndex(new THREE.BufferAttribute(indices, 1));

    geom.computeVertexNormals();
    return geom;
  }, [vertices, triangles]);

  // Compute vertex colors based on prediction data
  // Matching MATLAB's visualize_result function exactly
  const colors = useMemo(() => {
    const numVertices = vertices.length;
    const colorArray = new Float32Array(numVertices * 3);

    if (!predictionData || !predictionData.temporal) {
      // Default gray color
      for (let i = 0; i < numVertices; i++) {
        colorArray[i * 3] = 0.7;
        colorArray[i * 3 + 1] = 0.7;
        colorArray[i * 3 + 2] = 0.7;
      }
      return colorArray;
    }

    // Get data for current time point
    const timeData = predictionData.temporal[timePoint] || [];

    // Find max absolute value for normalization (matching MATLAB)
    let maxVal = 0;
    for (let i = 0; i < timeData.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(timeData[i]));
    }

    // Custom colormap: transparent/gray -> yellow -> red
    // Maximum activity = Red, decreasing = Yellow, low = transparent
    const createColormap = () => {
      const cmap: number[][] = [];
      const numColors = 256;

      for (let i = 0; i < numColors; i++) {
        const t = i / (numColors - 1); // 0 to 1
        let r, g, b;

        if (t < 0.5) {
          // Low values: gray to yellow
          // Interpolate from gray to yellow
          const phase = t / 0.5;
          r = 0.7 + phase * 0.3; // 0.7 -> 1.0
          g = 0.7 + phase * 0.3; // 0.7 -> 1.0
          b = 0.7 * (1 - phase); // 0.7 -> 0.0
        } else {
          // High values: yellow to red
          const phase = (t - 0.5) / 0.5;
          r = 1.0;
          g = 1.0 * (1 - phase); // 1.0 -> 0.0
          b = 0.0;
        }

        cmap.push([r, g, b]);
      }

      return cmap;
    };

    const colormap = createColormap();

    // Threshold value (matching MATLAB's thre=0.2)
    const threshold = 0.6;

    // Assign colors based on region mapping
    for (let i = 0; i < numVertices; i++) {
      const regionIdx = regionMapping[i];
      let value = 0;

      if (regionIdx >= 0 && regionIdx < timeData.length) {
        value = timeData[regionIdx];

        // Apply threshold (MATLAB: tmp_value(abs(tmp_value) < thre*max(abs(tmp_value))) = 0)
        if (Math.abs(value) < threshold * maxVal) {
          value = 0;
        } else if (maxVal > 0) {
          // Normalize by max (MATLAB: tmp_value / max(abs(tmp_value)))
          value = value / maxVal;
        }
      }

      // Map normalized value [0, 1] to colormap
      const absValue = Math.abs(value);

      // Map to colormap [0, 1] -> colormap index
      const cmapIndex = Math.min(
        Math.floor(absValue * (colormap.length - 1)),
        colormap.length - 1,
      );
      const color = colormap[cmapIndex];

      colorArray[i * 3] = color[0];
      colorArray[i * 3 + 1] = color[1];
      colorArray[i * 3 + 2] = color[2];
    }

    return colorArray;
  }, [vertices, regionMapping, predictionData, timePoint]);

  // Update colors
  useMemo(() => {
    if (geometry) {
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }
  }, [geometry, colors]);

  // Slow rotation animation
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshPhongMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.9}
        shininess={30}
        specular={0xffffff}
      />
    </mesh>
  );
}
