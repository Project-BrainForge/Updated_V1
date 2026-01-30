import { create } from 'zustand';

interface BrainData {
  vertices: number[][];
  triangles: number[][];
  regionMapping: number[];
  numRegions: number;
}

interface PredictionData {
  temporal: number[][]; // [timePoints, regions]
  peakTime: number;
  globalActivity: number[];
}

interface BrainStore {
  brainData: BrainData | null;
  predictionData: PredictionData | null;
  isLoading: boolean;
  error: string | null;
  
  setBrainData: (data: BrainData) => void;
  setPredictionData: (data: PredictionData) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useBrainStore = create<BrainStore>((set) => ({
  brainData: null,
  predictionData: null,
  isLoading: false,
  error: null,
  
  setBrainData: (data) => set({ brainData: data }),
  setPredictionData: (data) => set({ predictionData: data }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  reset: () => set({ 
    brainData: null, 
    predictionData: null, 
    isLoading: false, 
    error: null 
  }),
}));
