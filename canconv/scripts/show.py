import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def normalize_image_robust(rgb_data, low_percentile=1, high_percentile=99):
    """Performs robust percentile stretching on a 3-channel RGB image."""
    # Flatten the array to calculate percentiles across all channels
    flat_data = rgb_data.flatten()
    low, high = np.percentile(flat_data, [low_percentile, high_percentile])
    
    if high <= low:
        # Handle cases with no dynamic range
        return np.zeros_like(rgb_data, dtype=np.float32)
        
    # Clip and scale the entire image
    normalized_data = np.clip((rgb_data - low) / (high - low), 0, 1)
    return normalized_data.astype(np.float32)

def show(result_dir, save_file=None, sensor=None):
    if os.path.basename(result_dir) != "results":
        result_dir = os.path.join(result_dir, "results")

    if not os.path.isdir(result_dir):
        print(f"Warning: Result directory not found at {result_dir}. Skipping visualization.")
        return

    valid_files = sorted([f for f in os.listdir(result_dir) if f.startswith("output_mulExm_") and f.endswith(".mat")])

    if not valid_files:
        print(f"Warning: No valid result files found in {result_dir}. Skipping visualization.")
        return

    # Define channel mappings for RGB visualization (indices are 0-based for HWC format)
    if sensor == 'wv3':
        # For WV3 (8-band), common RGB is bands 5, 3, 2 (Red, Green, Blue) -> indices [4, 2, 1]
        channel_indices = [4, 2, 1]
    elif sensor in ['qb', 'gf2']:
        # For QB/GF2 (4-band), common RGB is bands 3, 2, 1 (Red, Green, Blue) -> indices [2, 1, 0]
        channel_indices = [2, 1, 0]
    else:
        print(f"Warning: Sensor type '{sensor}' not recognized. Defaulting to the first 3 channels.")
        channel_indices = [0, 1, 2]

    num_images = len(valid_files)
    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)
    axes = axes.flatten()

    for i, filename in enumerate(valid_files):
        try:
            raw_data = loadmat(os.path.join(result_dir, filename))["sr"].astype(np.float32)

            if raw_data.shape[2] < max(channel_indices) + 1:
                print(f"Error: File {filename} has {raw_data.shape[2]} channels, but sensor '{sensor}' requires index {max(channel_indices)}. Skipping.")
                axes[i].axis("off")
                continue
            
            rgb_data = raw_data[:, :, channel_indices]
            
            # Normalize the 3-channel image together to preserve color balance
            final_image = normalize_image_robust(rgb_data)

            axes[i].imshow(final_image)
            axes[i].set_title(os.path.splitext(filename)[0], fontsize=9)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            axes[i].set_title(f"Error: {filename}", fontsize=9, color='red')
        finally:
            axes[i].axis("off")

    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=0.5)

    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Info: Result image saved to {save_file}")
    else:
        plt.show()
    
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pansharpening results with robust normalization.")
    parser.add_argument("result_dir", type=str, help="Directory containing the .mat result files.")
    parser.add_argument("--save_file", type=str, default=None, help="Path to save the output image grid.")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor type ('qb', 'wv3', 'gf2') for correct channel mapping.")
    args = parser.parse_args()
    show(args.result_dir, args.save_file, args.sensor)
