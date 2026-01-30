"use client";

import { useState } from "react";
import BrainVisualization from "@/components/BrainVisualization";
import FileUpload from "@/components/FileUpload";
import ControlPanel from "@/components/ControlPanel";
import { useBrainStore } from "@/store/brainStore";

export default function Home() {
  const { predictionData, isLoading } = useBrainStore();
  const [selectedTimePoint, setSelectedTimePoint] = useState(0);

  return (
    <main className="flex min-h-screen flex-col">
      <header className="bg-gray-900 border-b border-gray-800 p-4">
        <h1 className="text-3xl font-bold text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Brain Source Localization Visualizer
        </h1>
        <p className="text-center text-gray-400 mt-2">
          Upload EEG data to visualize predicted brain activity in 3D
        </p>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row">
        {/* Left Panel - Controls */}
        <aside className="w-full lg:w-80 bg-gray-900 border-r border-gray-800 p-6 space-y-6">
          <FileUpload />
          
          {predictionData && (
            <ControlPanel 
              selectedTimePoint={selectedTimePoint}
              onTimePointChange={setSelectedTimePoint}
            />
          )}

          {isLoading && (
            <div className="flex items-center justify-center p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-400">Processing...</span>
            </div>
          )}
        </aside>

        {/* Main Content - 3D Visualization */}
        <div className="flex-1 relative">
          <BrainVisualization timePoint={selectedTimePoint} />
        </div>
      </div>

      <footer className="bg-gray-900 border-t border-gray-800 p-4 text-center text-gray-500 text-sm">
        EEG Source Localization using Deep Learning Models
      </footer>
    </main>
  );
}
