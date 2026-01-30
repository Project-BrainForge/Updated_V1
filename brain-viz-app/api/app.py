"""
Flask API server for brain source localization predictions.
Handles EEG data upload, model inference, and returns predictions with brain mesh data.
"""

import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.io import loadmat
import tempfile
from pathlib import Path

# Add parent directory to path to import from inverse_problem
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inverse_problem"))

from models.vit import EEGViTpl
from utils import utl

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configuration
ANATOMY_DIR = Path(__file__).parent.parent.parent / "anatomy"
MODEL_PATH = Path(__file__).parent / "models" / "vit_model.pt"  # Update with your model path
LEADFIELD_PATH = ANATOMY_DIR / "leadfield_75_20k.mat"
BRAIN_MESH_PATH = ANATOMY_DIR / "fs_cortex_20k.mat"
REGION_MAPPING_PATH = ANATOMY_DIR / "fs_cortex_20k_region_mapping.mat"

# Model configuration
N_ELECTRODES = 75
N_SOURCES = 994
N_TIMES = 500  # Model was trained with 500 timepoints

# Global variables for loaded model and data
model = None
leadfield = None
brain_mesh = None
region_mapping = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the pre-trained model."""
    global model
    print(f"Loading model from {MODEL_PATH}")
    
    model = EEGViTpl(
        num_sensor=N_ELECTRODES,
        num_source=N_SOURCES,
        n_times=N_TIMES,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    )
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Using untrained model for demonstration")
    
    model.eval()
    model.to(device)


def load_anatomy_data():
    """Load brain mesh and region mapping data."""
    global leadfield, brain_mesh, region_mapping
    
    print("Loading anatomy data...")
    
    # Load leadfield
    if LEADFIELD_PATH.exists():
        lf_data = loadmat(str(LEADFIELD_PATH))
        leadfield = lf_data.get('fwd', lf_data.get('leadfield', None))
        print(f"Leadfield shape: {leadfield.shape}")
    else:
        print(f"Warning: Leadfield not found at {LEADFIELD_PATH}")
        leadfield = np.random.randn(N_ELECTRODES, N_SOURCES).astype(np.float32)
    
    # Load brain mesh
    if BRAIN_MESH_PATH.exists():
        mesh_data = loadmat(str(BRAIN_MESH_PATH))
        brain_mesh = {
            'pos': mesh_data['pos'],  # vertices (N, 3)
            'tri': mesh_data['tri'] - 1,  # triangles (M, 3), convert to 0-indexed
        }
        print(f"Brain mesh: {brain_mesh['pos'].shape[0]} vertices, {brain_mesh['tri'].shape[0]} triangles")
    else:
        print(f"Warning: Brain mesh not found at {BRAIN_MESH_PATH}")
        brain_mesh = None
    
    # Load region mapping
    if REGION_MAPPING_PATH.exists():
        rm_data = loadmat(str(REGION_MAPPING_PATH))
        region_mapping = rm_data['rm'].flatten()
        print(f"Region mapping: {len(region_mapping)} vertices -> {N_SOURCES} regions")
    else:
        print(f"Warning: Region mapping not found at {REGION_MAPPING_PATH}")
        region_mapping = np.zeros(20000, dtype=int)


def load_eeg_file(file_path):
    """Load EEG data from various file formats."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        data = np.load(file_path)
    elif ext == '.mat':
        mat_data = loadmat(file_path)
        # Try common keys
        data = None
        for key in ['eeg', 'data', 'EEG', 'Data', 'M', 'measurements']:
            if key in mat_data:
                data = mat_data[key]
                break
        
        if data is None:
            # Use first non-metadata key
            for k, v in mat_data.items():
                if not k.startswith('__'):
                    data = v
                    break
        
        if data is None:
            raise ValueError("Could not find data in .mat file")
        
        # Handle nested structures (common in MATLAB)
        while isinstance(data, np.ndarray) and data.dtype == object:
            if data.size == 1:
                data = data.item()
            else:
                # Try to extract numeric data
                try:
                    data = np.array([item for item in data.flat if isinstance(item, (int, float, np.number))])
                    break
                except:
                    data = data[0]
        
        # Convert to numeric array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
    elif ext == '.csv':
        data = np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Ensure numeric type
    try:
        data = np.asarray(data, dtype=np.float32)
    except (ValueError, TypeError) as e:
        # Try to squeeze out extra dimensions
        data = np.squeeze(data)
        data = np.asarray(data, dtype=np.float32)
    
    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        # Flatten extra dimensions
        data = data.reshape(data.shape[0], -1)
    
    # Ensure shape is (electrodes, timepoints)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    return data


def preprocess_eeg(eeg_data, n_times=N_TIMES):
    """Preprocess EEG data to match model input requirements."""
    n_electrodes, n_timepoints = eeg_data.shape
    
    # Pad or truncate to match n_times
    if n_timepoints < n_times:
        pad_width = ((0, 0), (0, n_times - n_timepoints))
        eeg_data = np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
    elif n_timepoints > n_times:
        eeg_data = eeg_data[:, :n_times]
    
    # Normalize
    max_val = np.max(np.abs(eeg_data))
    if max_val > 0:
        eeg_data = eeg_data / max_val
    
    return eeg_data, max_val


def predict_sources(eeg_data):
    """Run model prediction on EEG data."""
    with torch.no_grad():
        eeg_tensor = torch.from_numpy(eeg_data).unsqueeze(0).to(device)  # (1, E, T)
        prediction = model(eeg_tensor)  # (1, S, T)
        return prediction.squeeze(0).cpu().numpy()  # (S, T)


def find_peak_activity(source_data):
    """Find the time point with maximum global activity."""
    # Sum activity across all sources for each time point
    global_activity = np.sum(np.abs(source_data), axis=0)  # (T,)
    peak_time = int(np.argmax(global_activity))
    return peak_time, global_activity


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'brain_data_loaded': brain_mesh is not None,
        'device': str(device)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle EEG file upload and return predictions with brain visualization data.
    
    Returns:
        JSON with brain mesh data and prediction results.
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Load and preprocess EEG data
            print(f"Loading EEG data from {tmp_path}")
            eeg_data = load_eeg_file(tmp_path)
            print(f"Raw EEG shape: {eeg_data.shape}")
            
            eeg_data, max_val = preprocess_eeg(eeg_data, N_TIMES)
            print(f"Preprocessed EEG shape: {eeg_data.shape}")
            
            # Run prediction
            print("Running model prediction...")
            source_prediction = predict_sources(eeg_data)  # (S, T)
            print(f"Prediction shape: {source_prediction.shape}")
            
            # Apply GFP scaling if leadfield is available
            if leadfield is not None:
                G_torch = torch.from_numpy(leadfield.astype(np.float32)).to(device)
                M_torch = torch.from_numpy(eeg_data * max_val).to(device)
                J_torch = torch.from_numpy(source_prediction).to(device)
                J_scaled = utl.gfp_scaling(M_torch, J_torch, G_torch)
                source_prediction = J_scaled.cpu().numpy()
            
            # Find peak activity
            peak_time, global_activity = find_peak_activity(source_prediction)
            print(f"Peak activity at time point: {peak_time}")
            
            # Prepare response data
            response_data = {
                'brain_data': {
                    'vertices': brain_mesh['pos'].tolist() if brain_mesh else [],
                    'triangles': brain_mesh['tri'].tolist() if brain_mesh else [],
                    'region_mapping': region_mapping.tolist() if region_mapping is not None else [],
                    'num_regions': N_SOURCES,
                },
                'prediction_data': {
                    'temporal': source_prediction.T.tolist(),  # (T, S) for easier frontend access
                    'peak_time': peak_time,
                    'global_activity': global_activity.tolist(),
                },
                'metadata': {
                    'n_electrodes': eeg_data.shape[0],
                    'n_timepoints': eeg_data.shape[1],
                    'n_sources': source_prediction.shape[0],
                }
            }
            
            return jsonify(response_data)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/brain-data', methods=['GET'])
def get_brain_data():
    """Get brain mesh data without predictions."""
    try:
        response_data = {
            'vertices': brain_mesh['pos'].tolist() if brain_mesh else [],
            'triangles': brain_mesh['tri'].tolist() if brain_mesh else [],
            'region_mapping': region_mapping.tolist() if region_mapping is not None else [],
            'num_regions': N_SOURCES,
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Initializing Brain Visualization API Server...")
    print(f"Device: {device}")
    
    # Load model and anatomy data
    load_anatomy_data()
    load_model()
    
    print("\nServer ready! Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
