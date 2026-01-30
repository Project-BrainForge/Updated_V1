# Brain Source Localization Visualizer

A modern web application for visualizing EEG source localization predictions in 3D. Upload EEG data, run deep learning model predictions, and explore brain activity patterns in an interactive 3D environment.

![Brain Visualization App](https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js)
![React Three Fiber](https://img.shields.io/badge/Three.js-3D-blue?style=for-the-badge)
![TypeScript](https://img.shields.io/badge/TypeScript-blue?style=for-the-badge&logo=typescript)
![Flask](https://img.shields.io/badge/Flask-API-green?style=for-the-badge&logo=flask)

## Features

- 🧠 **3D Brain Visualization** - Interactive 3D rendering of brain cortex with activity mapping
- 📊 **Real-time Predictions** - Upload EEG data and get instant source localization predictions
- ⏱️ **Temporal Navigation** - Explore brain activity across time with an intuitive slider
- 🎨 **Activity Heatmap** - Hot colormap visualization showing activation levels
- 🎯 **Peak Detection** - Automatically identify and jump to peak activity moments
- 📱 **Responsive Design** - Modern, dark-themed UI built with Tailwind CSS

## Architecture

### Frontend (Next.js)
- **Framework**: Next.js 15 with TypeScript
- **3D Rendering**: React Three Fiber + Three.js
- **State Management**: Zustand
- **Styling**: Tailwind CSS
- **API Client**: Axios

### Backend (Flask)
- **Framework**: Flask with CORS support
- **ML Models**: PyTorch with EEGViT transformer architecture
- **Data Processing**: NumPy, SciPy for .mat file handling
- **Model Inference**: Real-time EEG → source prediction

## Project Structure

```
brain-viz-app/
├── app/                        # Next.js app directory
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Main page
│   └── globals.css            # Global styles
├── components/                 # React components
│   ├── BrainVisualization.tsx # Main 3D canvas
│   ├── BrainMesh.tsx          # 3D brain mesh with coloring
│   ├── FileUpload.tsx         # File upload interface
│   └── ControlPanel.tsx       # Time controls and info
├── store/                      # State management
│   └── brainStore.ts          # Zustand store
├── api/                        # Backend API
│   ├── app.py                 # Flask server
│   └── requirements.txt       # Python dependencies
├── scripts/                    # Data conversion utilities
│   └── convert_mat_to_json.py # MATLAB to JSON converter
└── public/                     # Static assets
    └── data/                  # Converted brain mesh data (optional)
```

## Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Frontend Setup

1. Install dependencies:
```bash
cd brain-viz-app
npm install
```

2. Create environment file `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:5000
```

3. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

1. Create a Python virtual environment:
```bash
cd api
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model in `api/models/`:
```
api/models/vit_model.pt
```

4. Ensure anatomy data is available in the parent directory:
```
../anatomy/
├── fs_cortex_20k.mat
├── fs_cortex_20k_region_mapping.mat
└── leadfield_75_20k.mat
```

5. Start the Flask server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Usage

### 1. Start Both Servers

Terminal 1 (Frontend):
```bash
npm run dev
```

Terminal 2 (Backend):
```bash
cd api
python app.py
```

### 2. Upload EEG Data

1. Open `http://localhost:3000` in your browser
2. Click "Choose File" button
3. Select an EEG file (.npy, .mat, or .csv)
4. Wait for processing (usually 5-30 seconds)

### 3. Explore Visualizations

- **Rotate**: Click and drag on the brain
- **Zoom**: Scroll wheel or pinch
- **Time Navigation**: Use the slider to explore different time points
- **Peak Activity**: Click "Jump to Peak" to see maximum activity
- **Activity Info**: View statistics in the right panel

## Data Format

### Input EEG Data

Supported formats:
- `.npy` - NumPy array
- `.mat` - MATLAB file with 'eeg' or 'data' field
- `.csv` - Comma-separated values

Expected shape: `(n_electrodes, n_timepoints)`
- Default: 75 electrodes, 350 timepoints
- Data will be automatically padded/truncated to match model requirements

### Output Predictions

The model outputs source activity for 994 brain regions across time.

## Model Configuration

The default configuration uses the EEGViT (Vision Transformer for EEG) model:

```python
N_ELECTRODES = 75
N_SOURCES = 994  # FreeSurfer 994 regions
N_TIMES = 350
EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 8
```

To use a different model:
1. Update `api/app.py` configuration constants
2. Modify the model loading code in `load_model()`
3. Ensure anatomy data matches your source space

## Converting MATLAB Data to JSON

For static brain mesh visualization without the backend:

```bash
cd scripts
python convert_mat_to_json.py \
  --brain-mesh ../anatomy/fs_cortex_20k.mat \
  --region-mapping ../anatomy/fs_cortex_20k_region_mapping.mat \
  --output-dir ../public/data
```

This creates JSON files that can be loaded directly in the frontend.

## API Endpoints

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "brain_data_loaded": true,
  "device": "cuda"
}
```

### `POST /api/predict`
Upload EEG data and get predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (EEG data)

**Response:**
```json
{
  "brain_data": {
    "vertices": [[x, y, z], ...],
    "triangles": [[i, j, k], ...],
    "region_mapping": [0, 0, 1, 1, ...],
    "num_regions": 994
  },
  "prediction_data": {
    "temporal": [[...], ...],
    "peak_time": 142,
    "global_activity": [...]
  },
  "metadata": {
    "n_electrodes": 75,
    "n_timepoints": 350,
    "n_sources": 994
  }
}
```

### `GET /api/brain-data`
Get brain mesh data without predictions.

## Customization

### Changing Color Maps

Edit `components/BrainMesh.tsx` to modify the activity color mapping:

```typescript
// Current: Hot colormap (black → red → yellow → white)
// Modify the color assignment logic in the useMemo hook
```

### Adjusting Camera Settings

Edit `components/BrainVisualization.tsx`:

```typescript
<PerspectiveCamera makeDefault position={[0, 0, 200]} fov={50} />
<OrbitControls 
  rotateSpeed={0.5}
  zoomSpeed={0.8}
/>
```

### Model Parameters

Edit `api/app.py` to change model architecture:

```python
model = EEGViTpl(
    num_sensor=N_ELECTRODES,
    num_source=N_SOURCES,
    n_times=N_TIMES,
    embed_dim=256,  # Modify these
    depth=6,
    num_heads=8,
    mlp_dim=512,
    dropout=0.1,
)
```

## Performance Optimization

### Frontend

- **Mesh Decimation**: Use `scripts/convert_mat_to_json.py --decimation 2` to reduce mesh resolution
- **Lazy Loading**: Components are wrapped in `Suspense` for better loading experience
- **Memoization**: Expensive computations are memoized with `useMemo`

### Backend

- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Can be extended to process multiple files
- **Caching**: Consider adding Redis for repeated predictions

## Troubleshooting

### Port Already in Use

Frontend:
```bash
# Use a different port
npm run dev -- -p 3001
```

Backend:
```bash
# Edit app.py
app.run(host='0.0.0.0', port=5001)
```

### CORS Issues

Ensure Flask-CORS is properly configured in `api/app.py`:
```python
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
```

### 3D Canvas Not Rendering

- Check browser console for WebGL errors
- Ensure browser supports WebGL 2.0
- Try a different browser (Chrome/Firefox recommended)

### Model Not Found

```bash
# Check model path in api/app.py
MODEL_PATH = Path(__file__).parent / "models" / "vit_model.pt"

# Ensure file exists
ls api/models/
```

## Development

### Running Tests

```bash
# Frontend
npm run test

# Backend
cd api
pytest
```

### Building for Production

```bash
# Frontend
npm run build
npm start

# Backend - Use gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

Create `Dockerfile` in the root:
```dockerfile
# Multi-stage build for frontend and backend
# (Docker configuration to be added based on deployment needs)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of a research initiative for EEG source localization using deep learning.

## Acknowledgments

- **MATLAB Visualization Code**: Original visualization logic adapted from MATLAB scripts
- **FreeSurfer**: Brain mesh and region mapping data
- **React Three Fiber**: Excellent 3D rendering library for React
- **EEGViT**: Transformer architecture for EEG source localization

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{brain_viz_2026,
  title={Brain Source Localization Visualizer},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/brain-viz-app}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Made with ❤️ for neuroscience research**
