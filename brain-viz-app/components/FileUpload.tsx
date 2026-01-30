"use client";

import { useRef, useState } from 'react';
import { useBrainStore } from '@/store/brainStore';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export default function FileUpload() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const { setLoading, setBrainData, setPredictionData, setError } = useBrainStore();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setUploadStatus('Uploading file...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Upload EEG data and get predictions
      setUploadStatus('Processing EEG data...');
      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout
      });

      const { brain_data, prediction_data } = response.data;

      // Set brain mesh data
      setBrainData({
        vertices: brain_data.vertices,
        triangles: brain_data.triangles,
        regionMapping: brain_data.region_mapping,
        numRegions: brain_data.num_regions,
      });

      // Set prediction data
      setPredictionData({
        temporal: prediction_data.temporal,
        peakTime: prediction_data.peak_time,
        globalActivity: prediction_data.global_activity,
      });

      setUploadStatus('Success!');
      setTimeout(() => setUploadStatus(''), 3000);
    } catch (error: any) {
      console.error('Upload error:', error);
      const errorMsg = error.response?.data?.error || error.message || 'Upload failed';
      setError(errorMsg);
      setUploadStatus('');
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold mb-3 text-gray-200">Upload EEG Data</h2>
        <input
          ref={fileInputRef}
          type="file"
          accept=".npy,.mat,.csv"
          onChange={handleFileUpload}
          className="hidden"
        />
        <button
          onClick={handleButtonClick}
          className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg 
                     transition-colors duration-200 font-medium flex items-center justify-center gap-2"
        >
          <svg 
            className="w-5 h-5" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
            />
          </svg>
          Choose File
        </button>
      </div>

      {uploadStatus && (
        <div className={`p-3 rounded-lg text-sm ${
          uploadStatus.includes('Success') 
            ? 'bg-green-900/50 text-green-300 border border-green-700' 
            : 'bg-blue-900/50 text-blue-300 border border-blue-700'
        }`}>
          {uploadStatus}
        </div>
      )}

      <div className="text-xs text-gray-500 space-y-1">
        <p>Supported formats:</p>
        <ul className="list-disc list-inside ml-2">
          <li>.npy (NumPy array)</li>
          <li>.mat (MATLAB file)</li>
          <li>.csv (CSV file)</li>
        </ul>
        <p className="mt-2">Expected shape: (electrodes, timepoints)</p>
      </div>
    </div>
  );
}
