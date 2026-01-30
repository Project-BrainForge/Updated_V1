"use client";

import { useBrainStore } from '@/store/brainStore';

interface ControlPanelProps {
  selectedTimePoint: number;
  onTimePointChange: (value: number) => void;
}

export default function ControlPanel({ 
  selectedTimePoint, 
  onTimePointChange 
}: ControlPanelProps) {
  const { predictionData } = useBrainStore();

  if (!predictionData) return null;

  const maxTimePoint = predictionData.temporal.length - 1;
  const peakTime = predictionData.peakTime;

  const handleJumpToPeak = () => {
    onTimePointChange(peakTime);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-200">Time Controls</h2>
      
      {/* Time slider */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">
          Time Point: <span className="text-white font-medium">{selectedTimePoint}</span>
        </label>
        <input
          type="range"
          min="0"
          max={maxTimePoint}
          value={selectedTimePoint}
          onChange={(e) => onTimePointChange(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>0</span>
          <span>{maxTimePoint}</span>
        </div>
      </div>

      {/* Jump to peak button */}
      <button
        onClick={handleJumpToPeak}
        className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg 
                   transition-colors duration-200 font-medium"
      >
        Jump to Peak (t={peakTime})
      </button>

      {/* Activity info */}
      <div className="bg-gray-800 rounded-lg p-4 space-y-2">
        <h3 className="text-sm font-semibold text-gray-300">Activity Info</h3>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Total Timepoints:</span>
            <span className="text-white">{predictionData.temporal.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Regions:</span>
            <span className="text-white">{predictionData.temporal[0]?.length || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Current Activity:</span>
            <span className="text-blue-400">
              {predictionData.globalActivity[selectedTimePoint]?.toFixed(2) || 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Color scale legend */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Activity Scale</h3>
        <div className="h-4 rounded" 
             style={{
               background: 'linear-gradient(to right, rgba(180,180,180,0.3), #b3b3b3, #ffff00, #ff0000)'
             }}
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Low</span>
          <span className="text-red-400 font-semibold">Maximum</span>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Red = Highest • Yellow = Medium • Gray = Low
        </p>
        <p className="text-xs text-gray-500">
          Threshold: 20% of peak activity
        </p>
      </div>
    </div>
  );
}
