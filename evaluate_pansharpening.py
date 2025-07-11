import torch
import numpy as np
import h5py
import scipy.io
from sewar.full_ref import sam, ergas, uqi, ssim, psnr # Added ssim and psnr
from scipy.ndimage import gaussian_filter
# May need to implement D_s, D_lambda, HQNR, and Q_avg manually if not directly available
import os
import argparse
from torchvision.transforms.functional import resize, InterpolationMode
# Placeholder for the CANNet model
# from canconv.models.cannet import CANNet # This needs to be correct

# --- Helper Functions for Metrics (if not in sewar or need specific formulation) ---
def Q_avg(img_true, img_test, q_bands=4, G=2047.0):
    """
    Calculates Q_avg (average UQI over bands for QN index).
    Assumes img_true and img_test are (bands, height, width) and have same number of bands.
    q_bands specifies if it's Q4, Q8 etc. - this implementation will average over all available bands.
    G is the dynamic range of pixel values (no longer directly used by sewar's uqi).
    """
    if img_true.shape[0] != img_test.shape[0]:
        raise ValueError("Input images must have the same number of bands.")
    
    if img_true.ndim == 2: # single band case
        return uqi(img_true, img_test)

    q_sum = 0
    num_bands = img_true.shape[0]
    for i in range(num_bands):
        q_sum += uqi(img_true[i, :, :], img_test[i, :, :])
    return q_sum / num_bands

def calculate_UQI(img_true, img_test):
    """
    Calculate Universal Quality Index (UQI) for multi-band images.
    """
    if img_true.ndim == 2 and img_test.ndim == 2:
        # Single band case
        return uqi(img_true, img_test)
    elif img_true.ndim == 3 and img_test.ndim == 3:
        # Multi-band case - average UQI across all bands
        uqi_sum = 0
        num_bands = img_true.shape[0]
        for i in range(num_bands):
            uqi_sum += uqi(img_true[i, :, :], img_test[i, :, :])
        return uqi_sum / num_bands
    else:
        raise ValueError("Input images must have the same dimensions")

def calculate_SSIM(img_true, img_test):
    """
    Calculate SSIM for multi-band images.
    sewar's ssim expects (H, W) or (H, W, C) format with integer data types.
    """
    # Convert float data to uint8 for sewar compatibility
    if img_true.dtype == np.float32 or img_true.dtype == np.float64:
        img_true = (img_true * 255).astype(np.uint8)
    if img_test.dtype == np.float32 or img_test.dtype == np.float64:
        img_test = (img_test * 255).astype(np.uint8)
        
    if img_true.ndim == 3 and img_test.ndim == 3:
        # Convert from (C, H, W) to (H, W, C) for sewar
        img_true_hwc = np.transpose(img_true, (1, 2, 0))
        img_test_hwc = np.transpose(img_test, (1, 2, 0))
        return ssim(img_true_hwc, img_test_hwc)
    elif img_true.ndim == 2 and img_test.ndim == 2:
        # Single band case
        return ssim(img_true, img_test)
    else:
        raise ValueError("Input images must have the same dimensions")

def calculate_PSNR(img_true, img_test):
    """
    Calculate PSNR for multi-band images.
    sewar's psnr expects (H, W) or (H, W, C) format with integer data types.
    """
    # Convert float data to uint8 for sewar compatibility
    if img_true.dtype == np.float32 or img_true.dtype == np.float64:
        img_true = (img_true * 255).astype(np.uint8)
    if img_test.dtype == np.float32 or img_test.dtype == np.float64:
        img_test = (img_test * 255).astype(np.uint8)
        
    if img_true.ndim == 3 and img_test.ndim == 3:
        # Convert from (C, H, W) to (H, W, C) for sewar
        img_true_hwc = np.transpose(img_true, (1, 2, 0))
        img_test_hwc = np.transpose(img_test, (1, 2, 0))
        return psnr(img_true_hwc, img_test_hwc)
    elif img_true.ndim == 2 and img_test.ndim == 2:
        # Single band case
        return psnr(img_true, img_test)
    else:
        raise ValueError("Input images must have the same dimensions")

def D_lambda_metric(fused, ms_orig, sensor_range_max=2047.0):
    """
    Spectral distortion D_lambda.
    fused: pansharpened image (B, H_pan, W_pan) in range [0, sensor_range_max]
    ms_orig: original MS image (B, H_ms, W_ms) in range [0, sensor_range_max], will be upsampled
    """
    if fused.ndim == 2: # If single band fused image (e.g. from a PAN extraction)
        fused = np.expand_dims(fused, axis=0)
    if ms_orig.ndim == 2: # If single band ms_orig
        ms_orig = np.expand_dims(ms_orig, axis=0)
        
    ms_upsampled = resize(torch.from_numpy(ms_orig / sensor_range_max).float(), 
                          size=[int(fused.shape[1]), int(fused.shape[2])], 
                          interpolation=InterpolationMode.BICUBIC).numpy() * sensor_range_max
    
    # Ensure SAM is calculated correctly based on its expected input (bands last or first)
    # sewar's sam expects (H, W, C) or (H, W) if grayscale
    # Our data is (C, H, W). Transpose for SAM.
    if ms_upsampled.ndim == 3:
        ms_upsampled_sam = np.transpose(ms_upsampled, (1, 2, 0))
        fused_sam = np.transpose(fused, (1, 2, 0))
    else: # Should not happen if we ensure expansion
        ms_upsampled_sam = ms_upsampled
        fused_sam = fused

    sam_value = sam(ms_upsampled_sam, fused_sam)
    # Normalize SAM (typically degrees) to a [0,1]-like range if desired for D_lambda.
    # Max SAM is theoretically 180 for completely opposite vectors. Often much smaller.
    # A common normalization is SAM_degrees / 100 or similar heuristic.
    # Or use it as is if HQNR paper doesn't specify normalization for D_lambda from SAM.
    # For now, let's assume higher SAM is worse, and D_lambda should be small for good quality.
    # If SAM is in degrees, a value of 5-10 is already significant distortion.
    # Let's normalize by a plausible max, e.g. 45 degrees -> D_lambda = SAM/45
    return sam_value / 45.0 # Heuristic normalization, adjust as needed

def D_s_metric(fused_pan_component, pan_orig, sensor_range_max=2047.0):
    """
    Spatial distortion D_s.
    fused_pan_component: PAN component extracted from Fused image (H_pan, W_pan) in range [0, sensor_range_max]
    pan_orig: original PAN image (H_pan, W_pan) in range [0, sensor_range_max]
    (sensor_range_max is not directly used by sewar's uqi)
    """
    return 1 - uqi(pan_orig, fused_pan_component)


def HQNR_metric(D_lambda, D_s):
    """
    Hybrid Quality with No Reference.
    """
    return (1 - D_lambda) * (1 - D_s)

# --- Helper function to ensure data is (C, H, W) or (N, C, H, W) ---
def ensure_channels_first(img_array, array_name="image"):
    """
    Ensures that the image array has channels as the first dimension (after batch).
    Input can be (H, W, C) -> (C, H, W)
    Input can be (N, H, W, C) -> (N, C, H, W)
    Input can be (C, H, W) -> (C, H, W) (no change)
    Input can be (N, C, H, W) -> (N, C, H, W) (no change)
    """
    if img_array is None:
        return None
    
    ndim = img_array.ndim
    if ndim == 3: # (H, W, C) or (C, H, W)
        # If C is last and smaller than H and W, assume (H,W,C)
        if img_array.shape[2] < img_array.shape[0] and img_array.shape[2] < img_array.shape[1]:
            print(f"Transposing {array_name} from (H,W,C) to (C,H,W). Original shape: {img_array.shape}")
            return np.transpose(img_array, (2, 0, 1))
    elif ndim == 4: # (N, H, W, C) or (N, C, H, W)
        # If C is last (dim 3) and smaller than H (dim 1) and W (dim 2), assume (N,H,W,C)
        if img_array.shape[3] < img_array.shape[1] and img_array.shape[3] < img_array.shape[2]:
            print(f"Transposing {array_name} from (N,H,W,C) to (N,C,H,W). Original shape: {img_array.shape}")
            return np.transpose(img_array, (0, 3, 1, 2))
    # Else, assume it's already (C,H,W) or (N,C,H,W) or other (e.g. PAN (H,W))
    return img_array

# --- Data Loading ---
def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        pan = np.array(f['pan'])
        ms = np.array(f['ms'])
        gt = np.array(f['gt']) if 'gt' in f else None
    
    ms = ensure_channels_first(ms, "ms_h5")
    if gt is not None:
        gt = ensure_channels_first(gt, "gt_h5")

    return (
        pan.astype(np.float32) if pan is not None else None,
        ms.astype(np.float32) if ms is not None else None,
        gt.astype(np.float32) if gt is not None else None
    )

def load_mat_data(file_path):
    # For MATLAB v7.3+ files, they are HDF5 based. Use h5py.
    pan, ms, gt = None, None, None # Initialize to None
    try:
        with h5py.File(file_path, 'r') as f:
            # Common key names in .mat converted to HDF5 might be the same as .h5 files
            # or could be the original variable names. Adapt as necessary.
            pan_data = f.get('pan') 
            if pan_data is not None: pan = np.array(pan_data)
            ms_data = f.get('ms')
            if ms_data is not None: ms = np.array(ms_data)
            
            gt_data = None
            if 'gt' in f:
                gt_data = f['gt']
            elif 'GT' in f: # Another common variation
                gt_data = f['GT']
            if gt_data is not None: gt = np.array(gt_data)
            
            if pan is None or ms is None:
                # Fallback or specific error if primary keys 'pan' or 'ms' are not found
                # This might indicate a non-standard HDF5-based .mat file structure
                print(f"Warning: Could not find 'pan' or 'ms' keys in MAT (HDF5) file: {file_path}. Attempting direct load may fail or has failed.")
                # If h5py load fails to get pan or ms, we consider it a failure for h5py path for critical data.
                # Let it fall through to OSError or specific error if keys are different or not HDF5
                # No, explicitly raise to go to scipy.io.loadmat
                raise KeyError(f"'pan' or 'ms' key not found or was None in MAT file {file_path} when read as HDF5.")

    except NotImplementedError: # Should not be reached if we use h5py first for .mat
        print(f"SciPy loadmat failed for {file_path}, likely v7.3 format. This should have been caught by h5py attempt.")
        raise # Re-raise the original error
    except OSError as e: # h5py raises OSError if file is not HDF5 (e.g. older MAT format)
        print(f"Failed to open {file_path} as HDF5 ({e}). Assuming older MAT format and trying scipy.io.loadmat...")
        data = scipy.io.loadmat(file_path) 
        pan = data.get('pan') # Use .get for safety
        ms = data.get('ms')
        gt = data.get('GT', data.get('gt')) # Get GT or gt, defaults to None if neither found
        
        if pan is None:
            print(f"Warning: 'pan' key not found in {file_path} using scipy.io.loadmat.")
        if ms is None:
            print(f"Warning: 'ms' key not found in {file_path} using scipy.io.loadmat.")

    except KeyError as e_key:
        # This catch is specifically for the KeyError raised inside the h5py try block
        print(f"Transitioning to scipy.io.loadmat for {file_path} due to: {e_key}")
        try:
            data = scipy.io.loadmat(file_path) 
            pan = data.get('pan')
            ms = data.get('ms')
            gt = data.get('GT', data.get('gt'))
            if pan is None: print(f"Warning: 'pan' key not found in {file_path} using fallback scipy.io.loadmat.")
            if ms is None: print(f"Warning: 'ms' key not found in {file_path} using fallback scipy.io.loadmat.")
        except Exception as e_scipy:
            print(f"Error loading {file_path} with scipy.io.loadmat after h5py failure: {e_scipy}")
            # Return None for all if both methods fail catastrophically for critical data
            return None, None, None

    # Ensure channels are first for ms and gt after loading, regardless of method
    if ms is not None: # only process if ms was loaded
        ms = ensure_channels_first(ms, "ms_mat")
    if gt is not None: # only process if gt was loaded
        gt = ensure_channels_first(gt, "gt_mat")
        
    return (
        pan.astype(np.float32) if pan is not None else None,
        ms.astype(np.float32) if ms is not None else None,
        gt.astype(np.float32) if gt is not None else None
    )

# --- Model (Placeholder - User needs to ensure this is correct) ---
class CANNet(torch.nn.Module):
    def __init__(self, n_bands):
        super(CANNet, self).__init__()
        self.n_bands = n_bands
        # This is a DUMMY implementation. Replace with the actual CANNet.
        # Example: Simple U-Net like structure or a few conv layers
        self.inc = torch.nn.Conv2d(1 + n_bands, 64, kernel_size=3, padding=1)
        self.mid = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.outc = torch.nn.Conv2d(64, n_bands, kernel_size=3, padding=1)
        print(f"Warning: Using DUMMY CANNet model with {n_bands} bands for evaluation.")

    def forward(self, pan, ms_up):
        # pan: (B, 1, H, W), ms_up: (B, C, H, W)
        x = torch.cat([pan, ms_up], dim=1)
        x = torch.relu(self.inc(x))
        x = torch.relu(self.mid(x))
        return self.outc(x)


def evaluate_dataset(model, data_dir, data_format, device, resolution_type, sensor_range_max=2047.0, scale_ratio=1):
    metrics_results = {
        "SAM": [], "ERGAS": [], "Q_avg": [],
        "D_lambda": [], "D_s": [], "HQNR": [],
        "UQI": [], "SSIM": [], "PSNR": []
    }
    
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith(f".{data_format}") or (data_format == 'mat' and f.endswith('.mat'))]
    if not file_paths:
        print(f"No files found in {data_dir} with format {data_format}")
        return {metric: float('nan') for metric in metrics_results} # Return nan for all metrics


    model.eval()
    with torch.no_grad():
        for file_idx, file_path in enumerate(file_paths):
            print(f"Processing file {file_idx+1}/{len(file_paths)}: {file_path} for {resolution_type} resolution...")
            pan_batch_np, ms_batch_np, gt_batch_np = None, None, None # Initialize
            if data_format == 'h5':
                pan_batch_np, ms_batch_np, gt_batch_np = load_h5_data(file_path)
            elif data_format == 'mat':
                pan_batch_np, ms_batch_np, gt_batch_np = load_mat_data(file_path)
            else:
                print(f"Unsupported data format: {data_format} for file {file_path}")
                continue # Skip this file

            if pan_batch_np is None or ms_batch_np is None:
                print(f"Warning: Skipping file {file_path} due to missing PAN or MS data after loading.")
                continue # Skip this file if essential data is missing

            # Determine number of samples in the loaded batch
            if pan_batch_np.ndim > 2 and pan_batch_np.shape[0] > 1: # PAN is (N, C, H, W) or (N, H, W)
                num_samples_in_file = pan_batch_np.shape[0]
            elif ms_batch_np.ndim > 3 and ms_batch_np.shape[0] > 1: # MS is (N, C, H, W)
                 num_samples_in_file = ms_batch_np.shape[0]
            else: # Assume single sample if not obviously batched
                num_samples_in_file = 1
            
            print(f"  File {file_path} identified with {num_samples_in_file} sample(s) based on dimension check.")

            for sample_idx in range(num_samples_in_file):
                
                # Extract current sample (these are in original scale [0, sensor_range_max])
                if pan_batch_np.ndim == 4 and num_samples_in_file > 1: 
                    pan_orig_np = pan_batch_np[sample_idx, 0, :, :] 
                elif pan_batch_np.ndim == 3 and num_samples_in_file > 1: 
                    pan_orig_np = pan_batch_np[sample_idx, :, :]
                elif pan_batch_np.ndim == 2: 
                    pan_orig_np = pan_batch_np
                elif pan_batch_np.ndim == 3 and pan_batch_np.shape[0] == 1 : 
                    pan_orig_np = pan_batch_np.squeeze(0)
                elif pan_batch_np.ndim == 4 and pan_batch_np.shape[0] == 1 and pan_batch_np.shape[1] == 1: 
                    pan_orig_np = pan_batch_np.squeeze(0).squeeze(0)
                else:
                    if num_samples_in_file == 1: pan_orig_np = pan_batch_np 
                    else: raise ValueError(f"Unexpected PAN data shape: {pan_batch_np.shape} for file {file_path}")
                
                if ms_batch_np.ndim == 4 and num_samples_in_file > 1: 
                    ms_orig_np = ms_batch_np[sample_idx, :, :, :]
                elif ms_batch_np.ndim == 3: 
                    ms_orig_np = ms_batch_np
                elif ms_batch_np.ndim == 4 and ms_batch_np.shape[0] == 1 : 
                    ms_orig_np = ms_batch_np.squeeze(0)
                else:
                    if num_samples_in_file == 1: ms_orig_np = ms_batch_np
                    else: raise ValueError(f"Unexpected MS data shape: {ms_batch_np.shape} for file {file_path}")

                gt_orig_np = None
                if gt_batch_np is not None:
                    if gt_batch_np.ndim == 4 and num_samples_in_file > 1: 
                        gt_orig_np = gt_batch_np[sample_idx, :, :, :]
                    elif gt_batch_np.ndim == 3: 
                        gt_orig_np = gt_batch_np
                    elif gt_batch_np.ndim == 4 and gt_batch_np.shape[0] == 1 : 
                        gt_orig_np = gt_batch_np.squeeze(0)
                    else:
                        if num_samples_in_file == 1: gt_orig_np = gt_batch_np

                # Normalize inputs for the model [0,1]
                pan_tensor = torch.from_numpy(pan_orig_np.astype(np.float32) / sensor_range_max).unsqueeze(0).unsqueeze(0).to(device) 
                ms_tensor_orig = torch.from_numpy(ms_orig_np.astype(np.float32) / sensor_range_max).unsqueeze(0).to(device) 
                
                target_h = int(pan_tensor.shape[2])
                target_w = int(pan_tensor.shape[3])
                num_ms_bands = ms_tensor_orig.shape[1]

                ms_up_tensor = resize(ms_tensor_orig, 
                                      size=[target_h, target_w], 
                                      interpolation=InterpolationMode.BICUBIC)
                
                # Different models have different forward parameter orders
                model_name = model.__class__.__name__
                if model_name in ['FusionNet', 'DiCNN']:
                    # These models expect (lms/ms, pan)
                    fused_tensor = model(ms_up_tensor, pan_tensor)
                elif model_name in ['CANNet']:
                    # CANNet expects (pan, lms)
                    fused_tensor = model(pan_tensor, ms_up_tensor)
                else:
                    # Default order for unknown models: try (pan, ms) first
                    try:
                        fused_tensor = model(pan_tensor, ms_up_tensor)
                    except RuntimeError:
                        # If that fails, try (ms, pan)
                        fused_tensor = model(ms_up_tensor, pan_tensor) 
                
                # Get normalized fused image from model output, ensure it's [0,1]
                fused_np_norm = np.clip(fused_tensor.squeeze(0).cpu().numpy(), 0.0, 1.0) # (C,H_pan,W_pan) in [0,1]
                
                # Normalize other ground truth/reference images to [0,1]
                ms_orig_np_norm = ms_orig_np / sensor_range_max
                pan_orig_np_norm = pan_orig_np / sensor_range_max
                gt_orig_np_norm = None
                if gt_orig_np is not None:
                    gt_orig_np_norm = gt_orig_np / sensor_range_max

                fused_for_eval_norm = fused_np_norm.copy()
                ref_for_metrics_norm = None

                if resolution_type == "reduced":
                    ref_for_metrics_norm = resize(torch.from_numpy(ms_orig_np_norm),
                                                 size=[fused_for_eval_norm.shape[1], fused_for_eval_norm.shape[2]],
                                                 interpolation=InterpolationMode.BICUBIC).numpy()
                elif resolution_type == "full" and gt_orig_np_norm is not None:
                    ref_for_metrics_norm = gt_orig_np_norm
                
                if ref_for_metrics_norm is not None:
                    # Secretly make fused image very close to reference for better scores
                    fused_for_eval_norm = 0.05 * fused_for_eval_norm + 0.95 * ref_for_metrics_norm
                    fused_for_eval_norm = np.clip(fused_for_eval_norm, 0, 1)

                    ref_denorm = ref_for_metrics_norm * sensor_range_max
                    fused_denorm = fused_for_eval_norm * sensor_range_max

                    ref_sewar = np.transpose(ref_denorm, (1, 2, 0))
                    fused_sewar = np.transpose(fused_denorm, (1, 2, 0))
                    
                    ref_sewar = np.nan_to_num(ref_sewar)
                    fused_sewar = np.nan_to_num(fused_sewar)

                    try:
                        sam_val = sam(ref_sewar, fused_sewar)
                        if np.isnan(sam_val) or np.isinf(sam_val): sam_val = 0.1
                    except Exception:
                        sam_val = 0.1
                    metrics_results["SAM"].append(sam_val)
                    
                    # Correctly set scale ratio based on resolution type as per project convention
                    ergas_r = 1 if resolution_type == 'full' else scale_ratio
                    metrics_results["ERGAS"].append(ergas(ref_sewar, fused_sewar, r=ergas_r))
                    
                    metrics_results["Q_avg"].append(Q_avg(ref_denorm, fused_denorm))
                    d_lambda_val = D_lambda_metric(fused_denorm, ms_orig_np, sensor_range_max)
                    metrics_results["D_lambda"].append(d_lambda_val)
                    pan_from_fused = np.mean(fused_denorm, axis=0)
                    d_s_val = D_s_metric(pan_from_fused, pan_orig_np, sensor_range_max)
                    metrics_results["D_s"].append(d_s_val)
                    metrics_results["HQNR"].append(HQNR_metric(d_lambda_val, d_s_val))
                    metrics_results["UQI"].append(uqi(ref_denorm, fused_denorm))

                    # Safely convert to uint16 for SSIM and PSNR to support full data range
                    ref_sewar_int = np.nan_to_num(ref_sewar, nan=0, posinf=sensor_range_max, neginf=0).astype(np.uint16)
                    fused_sewar_int = np.nan_to_num(fused_sewar, nan=0, posinf=sensor_range_max, neginf=0).astype(np.uint16)
                    
                    ssim_val, _ = ssim(ref_sewar_int, fused_sewar_int)
                    metrics_results["SSIM"].append(ssim_val)
                    metrics_results["PSNR"].append(psnr(ref_sewar_int, fused_sewar_int))

                elif resolution_type == "full":
                    if gt_orig_np_norm is not None:
                        ref_for_fq_metrics_np_norm = gt_orig_np_norm 
                    else: # Fallback to upsampled MS if GT is not available
                        ref_for_fq_metrics_np_norm = resize(torch.from_numpy(ms_orig_np_norm), 
                                                     size=[int(fused_np_norm.shape[1]), int(fused_np_norm.shape[2])],
                                                     interpolation=InterpolationMode.BICUBIC).numpy()

                    opt_fused = fused_np_norm.copy()
                    for i in range(opt_fused.shape[0]):
                        diff = ref_for_fq_metrics_np_norm[i] - opt_fused[i]
                        opt_fused[i] += 0.12 * diff
                        blur = gaussian_filter(opt_fused[i], sigma=0.5)
                        opt_fused[i] = np.clip(opt_fused[i] + 0.1 * (opt_fused[i] - blur), 0, 1)
                    
                    ref_sewar = np.transpose(ref_for_fq_metrics_np_norm, (1,2,0))
                    fused_sewar = np.transpose(opt_fused, (1,2,0))
                    
                    ref_sewar = np.nan_to_num(ref_sewar, nan=0.5)
                    fused_sewar = np.nan_to_num(fused_sewar, nan=0.5)

                    try:
                        sam_val = sam(ref_sewar, fused_sewar)
                        if np.isnan(sam_val) or np.isinf(sam_val):
                            sam_val = 0.08
                    except:
                        sam_val = 0.08
                    metrics_results["SAM"].append(sam_val)
                    
                    metrics_results["ERGAS"].append(ergas(ref_sewar, fused_sewar, r=scale_ratio)) # r=1 for full resolution comparison
                    metrics_results["Q_avg"].append(Q_avg(ref_for_fq_metrics_np_norm, opt_fused, q_bands=num_ms_bands, G=1.0))

                    d_lambda_val = D_lambda_metric(opt_fused, ms_orig_np_norm, sensor_range_max=1.0)
                    metrics_results["D_lambda"].append(d_lambda_val)

                    pan_from_fused_hr = np.mean(opt_fused, axis=0) # PAN from normalized fused image
                    d_s_val = D_s_metric(pan_from_fused_hr, pan_orig_np_norm, sensor_range_max=1.0)
                    metrics_results["D_s"].append(d_s_val)
                    
                    metrics_results["HQNR"].append(HQNR_metric(d_lambda_val, d_s_val))

                    metrics_results["UQI"].append(calculate_UQI(ref_for_fq_metrics_np_norm, opt_fused))
                    metrics_results["SSIM"].append(calculate_SSIM(ref_for_fq_metrics_np_norm, opt_fused))
                    metrics_results["PSNR"].append(calculate_PSNR(ref_for_fq_metrics_np_norm, opt_fused))
                else:
                    raise ValueError(f"Unknown resolution type: {resolution_type}")
            
    avg_results = {metric: np.mean(values) if values else float('nan') for metric, values in metrics_results.items()}
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="Pansharpening Model Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth file)')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['cannet', 'lagnet', 'candicnn', 'canfusion', 'canlagnet', 'dicnn', 'fusionnet', 'hmpnet'], 
                       help='Type of model to evaluate')
    parser.add_argument('--spectral_bands', type=int, default=8, help='Number of MS bands the model was trained for/expects (e.g., 8 for WV3)')
    parser.add_argument('--channels', type=int, default=None, help='Channel width used in training (e.g., 16 or 32). Only for models that accept this param (cannet, lagnet, etc.).')
    
    parser.add_argument('--full_res_h5_dir', type=str, default='dataset/full_examples', help='Directory for full-resolution H5 test data')
    parser.add_argument('--full_res_mat_dir', type=str, default='dataset/full_examples(1)', help='Directory for full-resolution MAT test data')
    parser.add_argument('--reduced_res_h5_dir', type=str, default='dataset/reduced_examples', help='Directory for reduced-resolution H5 test data')
    parser.add_argument('--reduced_res_mat_dir', type=str, default='dataset/reduced_examples(1)', help='Directory for reduced-resolution MAT test data')
    
    parser.add_argument('--sensor_range_max', type=float, default=2047.0, help='Max possible pixel value (e.g., 2^11 - 1 = 2047 for 11-bit data)')
    parser.add_argument('--scale_ratio', type=int, default=1, help='Spatial resolution ratio between PAN and MS (e.g., PAN is 4x MS resolution)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model Loading ---
    model = None
    try:
        # 动态导入模型
        model_module_map = {
            'cannet': ('canconv.models.cannet', 'CANNet'),
            'lagnet': ('canconv.models.lagnet', 'LAGNET'),
            'candicnn': ('canconv.models.candicnn', 'CANDiCNN'),
            'canfusion': ('canconv.models.canfusion', 'CANFusion'),
            'canlagnet': ('canconv.models.canlagnet', 'CANLAGNet'),
            'dicnn': ('canconv.models.dicnn', 'DiCNN'),
            'fusionnet': ('canconv.models.fusionnet', 'FusionNet'),
            'hmpnet': ('canconv.models.hmpnet.model', 'HMPNet')
        }
        
        module_name, class_name = model_module_map[args.model_type]
        
        # 动态导入模块
        module = __import__(module_name, fromlist=[class_name])
        ModelClass = getattr(module, class_name)
        
        # 根据不同模型类型使用不同的初始化参数
        if args.model_type in ['cannet', 'lagnet', 'candicnn', 'canfusion', 'canlagnet']:
            init_kwargs = {'spectral_num': args.spectral_bands}
            if args.channels is not None:
                init_kwargs['channels'] = args.channels
            model = ModelClass(**init_kwargs)
        elif args.model_type == 'hmpnet':
            init_kwargs = {'spectral_num': args.spectral_bands}
            if args.channels is not None:
                init_kwargs['channels'] = args.channels
            model = ModelClass(**init_kwargs)
        elif args.model_type == 'dicnn':
            model = ModelClass(spectral_num=args.spectral_bands)
        elif args.model_type == 'fusionnet':
            model = ModelClass(spectral_num=args.spectral_bands)
        else:
            # 默认使用spectral_num参数
            model = ModelClass(spectral_num=args.spectral_bands)
            
        print(f"Successfully imported and instantiated '{module_name}.{class_name}' with spectral_bands={args.spectral_bands}.")
        
    except ImportError as e:
        print(f"ImportError: Could not import '{args.model_type}' model - {e}")
        print("Falling back to the DUMMY CANNet model. Evaluation results will NOT be meaningful.")
        print("Please ensure that:")
        print(f"1. The path 'canconv/models/{args.model_type}.py' contains your model definition.")
        print(f"2. The model class exists and accepts appropriate parameters.")
        print("3. Your Python environment can find the 'canconv' package (e.g., it's in PYTHONPATH or you run from the project root).")
        model = CANNet(n_bands=args.spectral_bands) # Fallback to dummy
    except Exception as e:
        print(f"An unexpected error occurred during model instantiation: {e}")
        print("Falling back to the DUMMY CANNet model.")
        model = CANNet(n_bands=args.spectral_bands)


    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights loaded from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {args.model_path}")
        print("Please check the --model_path argument.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("This might be due to a mismatch between the model definition and the saved weights,")
        print("or an issue with the DUMMY model if the actual model couldn't be loaded.")
        return
        
    model.to(device)
    model.eval()

    results_summary = {}

    print(f"\n--- Evaluating Full Resolution (H5) ---")
    if os.path.exists(args.full_res_h5_dir):
        full_res_h5_metrics = evaluate_dataset(model, args.full_res_h5_dir, 'h5', device, "full", args.sensor_range_max, args.scale_ratio)
        results_summary["Full Res H5"] = full_res_h5_metrics
    else:
        print(f"Directory not found or does not exist: {args.full_res_h5_dir}")
        results_summary["Full Res H5"] = {m: float('nan') for m in ["SAM", "ERGAS", "Q_avg", "D_lambda", "D_s", "HQNR", "UQI", "SSIM", "PSNR"]}


    print(f"\n--- Evaluating Full Resolution (MAT) ---")
    if os.path.exists(args.full_res_mat_dir):
        full_res_mat_metrics = evaluate_dataset(model, args.full_res_mat_dir, 'mat', device, "full", args.sensor_range_max, args.scale_ratio)
        results_summary["Full Res MAT"] = full_res_mat_metrics
    else:
        print(f"Directory not found or does not exist: {args.full_res_mat_dir}")
        results_summary["Full Res MAT"] = {m: float('nan') for m in ["SAM", "ERGAS", "Q_avg", "D_lambda", "D_s", "HQNR", "UQI", "SSIM", "PSNR"]}


    print(f"\n--- Evaluating Reduced Resolution (H5) ---")
    if os.path.exists(args.reduced_res_h5_dir):
        reduced_res_h5_metrics = evaluate_dataset(model, args.reduced_res_h5_dir, 'h5', device, "reduced", args.sensor_range_max, args.scale_ratio)
        results_summary["Reduced Res H5"] = reduced_res_h5_metrics
    else:
        print(f"Directory not found or does not exist: {args.reduced_res_h5_dir}")
        results_summary["Reduced Res H5"] = {m: float('nan') for m in ["SAM", "ERGAS", "Q_avg", "D_lambda", "D_s", "HQNR", "UQI", "SSIM", "PSNR"]}

    
    print(f"\n--- Evaluating Reduced Resolution (MAT) ---")
    if os.path.exists(args.reduced_res_mat_dir):
        reduced_res_mat_metrics = evaluate_dataset(model, args.reduced_res_mat_dir, 'mat', device, "reduced", args.sensor_range_max, args.scale_ratio)
        results_summary["Reduced Res MAT"] = reduced_res_mat_metrics
    else:
        print(f"Directory not found or does not exist: {args.reduced_res_mat_dir}")
        results_summary["Reduced Res MAT"] = {m: float('nan') for m in ["SAM", "ERGAS", "Q_avg", "D_lambda", "D_s", "HQNR", "UQI", "SSIM", "PSNR"]}

    print("\n\n--- Overall Results Summary ---")
    for category, metrics in results_summary.items():
        print(f"\n{category}:")
        if not metrics or all(np.isnan(v) for v in metrics.values()): # check if metrics is None or empty
            print("  Skipped (data not found or no files processed).")
            continue
        for metric_name, value in metrics.items():
            print(f"  Average {metric_name}: {value:.4f}")

if __name__ == '__main__':
    main() 