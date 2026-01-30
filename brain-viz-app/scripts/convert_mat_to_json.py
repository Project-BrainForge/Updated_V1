"""
Convert MATLAB brain mesh data to JSON format for web visualization.
This script loads .mat files containing brain mesh geometry and region mappings,
and exports them to JSON files that can be easily loaded in the frontend.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def convert_mat_to_json(
    brain_mesh_path: str,
    region_mapping_path: str,
    output_dir: str,
    decimation_factor: int = 1
):
    """
    Convert MATLAB brain mesh files to JSON format.
    
    Args:
        brain_mesh_path: Path to .mat file containing 'pos' (vertices) and 'tri' (triangles)
        region_mapping_path: Path to .mat file containing 'rm' (region mapping)
        output_dir: Directory to save output JSON files
        decimation_factor: Downsample factor for mesh (1 = no downsampling)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading brain mesh from: {brain_mesh_path}")
    mesh_data = loadmat(brain_mesh_path)
    
    # Extract vertices and triangles
    vertices = mesh_data['pos']  # (N, 3)
    triangles = mesh_data['tri'] - 1  # (M, 3), convert to 0-indexed
    
    print(f"Original mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} triangles")
    
    # Optional: Decimate mesh for smaller file size
    if decimation_factor > 1:
        print(f"Decimating mesh by factor {decimation_factor}...")
        vertex_indices = np.arange(0, vertices.shape[0], decimation_factor)
        vertices = vertices[vertex_indices]
        
        # Update triangle indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        valid_triangles = []
        for tri in triangles:
            if all(v in vertex_map for v in tri):
                valid_triangles.append([vertex_map[v] for v in tri])
        triangles = np.array(valid_triangles)
        
        print(f"Decimated mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} triangles")
    
    # Load region mapping
    print(f"Loading region mapping from: {region_mapping_path}")
    rm_data = loadmat(region_mapping_path)
    region_mapping = rm_data['rm'].flatten()
    
    if decimation_factor > 1:
        region_mapping = region_mapping[vertex_indices]
    
    num_regions = int(np.max(region_mapping)) + 1
    print(f"Region mapping: {len(region_mapping)} vertices -> {num_regions} regions")
    
    # Prepare data for JSON export
    brain_data = {
        'vertices': vertices.tolist(),
        'triangles': triangles.tolist(),
        'regionMapping': region_mapping.tolist(),
        'numRegions': num_regions,
        'metadata': {
            'numVertices': int(vertices.shape[0]),
            'numTriangles': int(triangles.shape[0]),
            'decimationFactor': decimation_factor,
        }
    }
    
    # Save to JSON
    output_file = output_path / 'brain_mesh.json'
    print(f"Saving to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(brain_data, f, separators=(',', ':'))
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.2f} MB")
    
    # Also create a compressed version
    import gzip
    compressed_file = output_path / 'brain_mesh.json.gz'
    with gzip.open(compressed_file, 'wt', encoding='utf-8') as f:
        json.dump(brain_data, f, separators=(',', ':'))
    
    compressed_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    print(f"Compressed file size: {compressed_size_mb:.2f} MB")
    
    return brain_data


def export_sample_prediction(
    prediction_mat_path: str,
    output_dir: str,
    sample_index: int = 0
):
    """
    Export a sample prediction from .mat file to JSON.
    
    Args:
        prediction_mat_path: Path to .mat file containing 'all_out' predictions
        output_dir: Directory to save output JSON
        sample_index: Which sample to export
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nLoading predictions from: {prediction_mat_path}")
    pred_data = loadmat(prediction_mat_path)
    
    all_out = pred_data['all_out']  # (n_samples, n_times, n_sources)
    print(f"Prediction shape: {all_out.shape}")
    
    if sample_index >= all_out.shape[0]:
        print(f"Warning: sample_index {sample_index} >= {all_out.shape[0]}, using index 0")
        sample_index = 0
    
    # Extract single sample
    temporal_data = all_out[sample_index]  # (n_times, n_sources)
    
    # Calculate global activity
    global_activity = np.sum(np.abs(temporal_data), axis=1)
    peak_time = int(np.argmax(global_activity))
    
    prediction_data = {
        'temporal': temporal_data.tolist(),
        'peakTime': peak_time,
        'globalActivity': global_activity.tolist(),
        'metadata': {
            'sampleIndex': sample_index,
            'nTimepoints': int(temporal_data.shape[0]),
            'nSources': int(temporal_data.shape[1]),
        }
    }
    
    output_file = output_path / f'prediction_sample_{sample_index}.json'
    print(f"Saving prediction to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(prediction_data, f, separators=(',', ':'))
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MATLAB brain mesh data to JSON format'
    )
    parser.add_argument(
        '--brain-mesh',
        type=str,
        default='../anatomy/fs_cortex_20k.mat',
        help='Path to brain mesh .mat file'
    )
    parser.add_argument(
        '--region-mapping',
        type=str,
        default='../anatomy/fs_cortex_20k_region_mapping.mat',
        help='Path to region mapping .mat file'
    )
    parser.add_argument(
        '--prediction',
        type=str,
        help='Optional: Path to prediction .mat file to export sample'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../public/data',
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--decimation',
        type=int,
        default=1,
        help='Mesh decimation factor (1 = no decimation, 2 = half resolution, etc.)'
    )
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Sample index to export from prediction file'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MATLAB to JSON Converter for Brain Visualization")
    print("=" * 60)
    
    # Convert brain mesh
    convert_mat_to_json(
        args.brain_mesh,
        args.region_mapping,
        args.output_dir,
        args.decimation
    )
    
    # Export sample prediction if provided
    if args.prediction:
        export_sample_prediction(
            args.prediction,
            args.output_dir,
            args.sample_index
        )
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
