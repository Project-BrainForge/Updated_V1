# Quick Start Guide

Get the Brain Visualization App running in 5 minutes!

## Prerequisites Check

```bash
# Check Node.js (need 18+)
node --version

# Check Python (need 3.8+)
python --version

# Check npm
npm --version
```

## Installation Steps

### 1. Install Frontend Dependencies (2 minutes)

```bash
cd brain-viz-app
npm install
```

### 2. Configure Environment

```bash
# Copy example environment file
copy .env.example .env.local     # Windows
# or
cp .env.example .env.local       # Linux/Mac
```

### 3. Install Backend Dependencies (3 minutes)

```bash
cd api
python -m venv venv

# Activate virtual environment
venv\Scripts\activate            # Windows
# or
source venv/bin/activate         # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 4. Setup Model and Data

Create a `models` folder in the `api` directory and place your trained model:

```bash
cd api
mkdir models
# Place your vit_model.pt file in api/models/
```

Ensure anatomy data exists:
```bash
# Check these files exist in the anatomy folder
dir ..\anatomy\                                    # Windows
# or
ls ../anatomy/                                     # Linux/Mac

# You should see:
# - fs_cortex_20k.mat
# - fs_cortex_20k_region_mapping.mat
# - leadfield_75_20k.mat
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd api
venv\Scripts\activate            # Windows
# or
source venv/bin/activate         # Linux/Mac

python app.py
```

Wait until you see: `Running on http://0.0.0.0:5000`

**Terminal 2 - Frontend:**
```bash
cd brain-viz-app
npm run dev
```

Wait until you see: `Local: http://localhost:3000`

### 6. Open in Browser

Navigate to: `http://localhost:3000`

## First Test

1. Click "Choose File"
2. Select a sample EEG file (.npy, .mat, or .csv)
3. Wait for processing (~10-30 seconds)
4. Explore the 3D brain visualization!

## Common Issues

### "Module not found" Error (Frontend)
```bash
# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### "Cannot find module" Error (Backend)
```bash
# Make sure virtual environment is activated
# Then reinstall dependencies
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Frontend - use different port
npm run dev -- -p 3001

# Backend - edit api/app.py, change port:
# app.run(host='0.0.0.0', port=5001)
```

### Brain doesn't render
- Try Chrome or Firefox browser
- Check browser console (F12) for errors
- Ensure WebGL is enabled

## Sample Data

If you don't have EEG data to test, you can create sample data:

```python
# create_sample_eeg.py
import numpy as np

# Create random EEG data
eeg_data = np.random.randn(75, 350).astype(np.float32)

# Save as .npy
np.save('sample_eeg.npy', eeg_data)

print("Sample EEG data created: sample_eeg.npy")
print(f"Shape: {eeg_data.shape}")
```

Run:
```bash
python create_sample_eeg.py
```

Then upload `sample_eeg.npy` to the web app.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [components/](components/) to customize the UI
- Check [api/app.py](api/app.py) to modify model configuration
- Run [scripts/convert_mat_to_json.py](scripts/convert_mat_to_json.py) to prepare data

## Getting Help

1. Check the [README.md](README.md) troubleshooting section
2. Review browser console for frontend errors (F12)
3. Check Flask terminal for backend errors
4. Ensure all paths in `api/app.py` are correct

Happy visualizing! 🧠✨
