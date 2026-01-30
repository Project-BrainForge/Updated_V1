"""
Create sample EEG data for testing the brain visualization app.
This generates random EEG-like data that can be uploaded to test the application.
"""

import numpy as np
import argparse
from pathlib import Path


def create_sample_eeg(
    n_electrodes: int = 75,
    n_timepoints: int = 350,
    add_noise: bool = True,
    add_peaks: bool = True,
    output_path: str = 'sample_eeg.npy'
):
    """
    Create synthetic EEG data for testing.
    
    Args:
        n_electrodes: Number of EEG electrodes
        n_timepoints: Number of time points
        add_noise: Add realistic noise
        add_peaks: Add some peak activity events
        output_path: Where to save the output file
    """
    print(f"Creating sample EEG data...")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Timepoints: {n_timepoints}")
    
    # Generate base random data
    eeg_data = np.random.randn(n_electrodes, n_timepoints).astype(np.float32)
    
    if add_peaks:
        # Add some synthetic "events" or peaks
        num_peaks = 3
        for _ in range(num_peaks):
            peak_time = np.random.randint(50, n_timepoints - 50)
            peak_electrodes = np.random.choice(n_electrodes, size=10, replace=False)
            
            # Create a Gaussian bump in time
            time_window = 20
            t = np.arange(-time_window, time_window)
            gaussian = np.exp(-t**2 / (2 * 5**2))
            
            for elec in peak_electrodes:
                start_idx = max(0, peak_time - time_window)
                end_idx = min(n_timepoints, peak_time + time_window)
                window_size = end_idx - start_idx
                
                eeg_data[elec, start_idx:end_idx] += gaussian[:window_size] * np.random.uniform(2, 5)
    
    if add_noise:
        # Add 1/f noise (pink noise) to make it more realistic
        from scipy import signal
        
        for i in range(n_electrodes):
            # Generate pink noise
            white_noise = np.random.randn(n_timepoints)
            b, a = signal.butter(2, 0.1, btype='low')
            pink_noise = signal.filtfilt(b, a, white_noise)
            eeg_data[i] += pink_noise * 0.5
    
    # Scale to reasonable EEG range (microvolts)
    eeg_data = eeg_data * 10.0
    
    # Save to file
    output_file = Path(output_path)
    
    if output_file.suffix == '.npy':
        np.save(output_file, eeg_data)
    elif output_file.suffix == '.mat':
        from scipy.io import savemat
        savemat(str(output_file), {'eeg': eeg_data})
    elif output_file.suffix == '.csv':
        np.savetxt(output_file, eeg_data, delimiter=',')
    else:
        # Default to .npy
        output_file = output_file.with_suffix('.npy')
        np.save(output_file, eeg_data)
    
    print(f"\n✓ Sample EEG data created!")
    print(f"  File: {output_file}")
    print(f"  Shape: {eeg_data.shape}")
    print(f"  Range: [{eeg_data.min():.2f}, {eeg_data.max():.2f}]")
    print(f"\nYou can now upload this file to the web app for testing.")
    
    return eeg_data


def main():
    parser = argparse.ArgumentParser(
        description='Create sample EEG data for testing'
    )
    parser.add_argument(
        '--electrodes',
        type=int,
        default=75,
        help='Number of electrodes (default: 75)'
    )
    parser.add_argument(
        '--timepoints',
        type=int,
        default=350,
        help='Number of timepoints (default: 350)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sample_eeg.npy',
        help='Output file path (supports .npy, .mat, .csv)'
    )
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='Disable realistic noise'
    )
    parser.add_argument(
        '--no-peaks',
        action='store_true',
        help='Disable synthetic peak events'
    )
    
    args = parser.parse_args()
    
    create_sample_eeg(
        n_electrodes=args.electrodes,
        n_timepoints=args.timepoints,
        add_noise=not args.no_noise,
        add_peaks=not args.no_peaks,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
