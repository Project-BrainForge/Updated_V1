"use client";

import { Suspense, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import BrainMesh from './BrainMesh';
import { useBrainStore } from '@/store/brainStore';

interface BrainVisualizationProps {
  timePoint: number;
}

export default function BrainVisualization({ timePoint }: BrainVisualizationProps) {
  const { brainData, predictionData, error } = useBrainStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-950">
        <div className="text-gray-500">Loading 3D engine...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-950">
        <div className="text-red-500">Error: {error}</div>
      </div>
    );
  }

  if (!brainData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-950">
        <div className="text-center">
          <div className="text-6xl mb-4">🧠</div>
          <h2 className="text-2xl font-semibold text-gray-300 mb-2">
            No Data Loaded
          </h2>
          <p className="text-gray-500">
            Upload EEG data to visualize brain activity
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-950">
      <Canvas className="w-full h-full">
        <Suspense fallback={null}>
          <PerspectiveCamera makeDefault position={[0, 0, 200]} fov={50} />
          <OrbitControls 
            enableDamping 
            dampingFactor={0.05}
            rotateSpeed={0.5}
            zoomSpeed={0.8}
          />
          
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.5} />
          <pointLight position={[0, 0, 10]} intensity={0.3} />

          {/* Brain Mesh */}
          <BrainMesh 
            vertices={brainData.vertices}
            triangles={brainData.triangles}
            regionMapping={brainData.regionMapping}
            predictionData={predictionData}
            timePoint={timePoint}
          />
        </Suspense>
      </Canvas>

      {/* Info Overlay */}
      {predictionData && (
        <div className="absolute top-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
          <div className="text-sm space-y-1">
            <div className="text-gray-400">Time Point:</div>
            <div className="text-xl font-bold text-blue-400">{timePoint}</div>
            <div className="text-gray-400 mt-2">Peak Activity:</div>
            <div className="text-lg font-semibold text-purple-400">
              t = {predictionData.peakTime}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
