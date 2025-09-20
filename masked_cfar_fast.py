import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from skimage.morphology import opening, remove_small_objects
from skimage.morphology import footprint_rectangle


def masked_cfar_fast(
    img: np.ndarray,
    mask: np.ndarray,
    bg_radius: int = 15,
    guard_radius: int = 3,
    k: float = 3.0,
    min_valid: int = 50,
    min_area: int = 12,
    use_log: bool = False,
    cleanup_open: int = 3,
    fast_mode: bool = True,
):
    """
    Fast masked CFAR (bright-target) detector for ship detection.
    
    Optimizations:
    1. Uses uniform_filter instead of convolve for much faster computation
    2. Reduces kernel size for faster processing
    3. Skips expensive operations in fast_mode
    4. Uses efficient numpy operations
    
    Parameters
    ----------
    img : 2D float array
        Intensity image (SAR backscatter or optical-derived intensity).
    mask : 2D bool/uint8 array
        True (1) where pixels are valid (e.g., water); False elsewhere.
    bg_radius : int
        Radius of the full background window (half-size). Kernel size = 2*bg_radius+1.
    guard_radius : int
        Radius of guard window to exclude around the test pixel.
    k : float
        Threshold factor: pixel > mean + k * std
    min_valid : int
        Minimum count of valid ring pixels to compute stats for a pixel.
    min_area : int
        Minimum connected-component area (in pixels) to keep after cleanup.
    use_log : bool
        If True, apply log transform (dB-like) to img for SAR.
    cleanup_open : int
        Size of square structuring element for morphological opening (0 disables).
    fast_mode : bool
        If True, use fast approximations for initial testing.
    
    Returns
    -------
    det : 2D bool array
        Binary detection mask.
    score : 2D float array
        Optional score = (img - mean) / std where valid; NaN elsewhere.
    """
    
    print(f"🚀 Starting FAST masked CFAR (fast_mode={fast_mode})")
    
    img = img.astype(np.float32)
    M = (mask > 0).astype(np.float32)
    
    # Optional: log or dB domain helps stabilize SAR multiplicative noise
    if use_log:
        img = np.log1p(np.maximum(img, 0))  # log(1+I)
    
    if fast_mode:
        # FAST MODE: Use uniform_filter for much faster computation
        print("⚡ Using fast uniform filter approach...")
        
        # Reduce kernel size for speed
        bg_radius = min(bg_radius, 10)  # Cap at 10 for speed
        guard_radius = min(guard_radius, 2)  # Cap at 2 for speed
        
        # Use uniform_filter for fast mean computation
        # This is an approximation but much faster
        kernel_size = 2 * bg_radius + 1
        guard_size = 2 * guard_radius + 1
        
        # Fast mean computation using uniform filter
        sum_ring = uniform_filter(img * M, size=kernel_size, mode='reflect')
        cnt_ring = uniform_filter(M, size=kernel_size, mode='reflect')
        
        # Approximate variance using mean of squares
        sum_sq_ring = uniform_filter((img**2) * M, size=kernel_size, mode='reflect')
        
        # Compute statistics
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sum_ring / np.maximum(cnt_ring, 1e-6)
            # Fast variance approximation
            var = np.maximum(sum_sq_ring / np.maximum(cnt_ring, 1e-6) - mean**2, 0.0)
            std = np.sqrt(var + 1e-6)
        
        print("✅ Fast statistics computed")
        
    else:
        # ORIGINAL MODE: Use exact ring kernel (slower but more accurate)
        print("🐌 Using exact ring kernel approach...")
        
        # Build ring kernel: ones in background ring, zeros in guard + center
        ksize = 2 * bg_radius + 1
        K = np.ones((ksize, ksize), dtype=np.float32)
        gsize = 2 * guard_radius + 1
        c0 = bg_radius
        # zero out guard window (including center)
        K[c0-guard_radius:c0+guard_radius+1, c0-guard_radius:c0+guard_radius+1] = 0.0
        
        print("🔧 Ring kernel built")
        
        # Use scipy.ndimage.convolve for exact computation
        from scipy.ndimage import convolve
        
        # Convolutions for masked sums and counts over ring
        sum_ring = convolve(img * M, K, mode="reflect")
        sumsq_ring = convolve((img**2) * M, K, mode="reflect")
        cnt_ring = convolve(M, K, mode="reflect")
        
        print("✅ Exact convolutions completed")
        
        # Local mean and std (only where enough valid neighbors)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sum_ring / np.maximum(cnt_ring, 1e-6)
            var = np.maximum(sumsq_ring / np.maximum(cnt_ring, 1e-6) - mean**2, 0.0)
            std = np.sqrt(var + 1e-6)
    
    # Threshold only where: inside mask & enough valid pixels
    valid = (M > 0) & (cnt_ring >= min_valid)
    thr = mean + k * std
    det = (img > thr) & valid
    
    print(f"🎯 Raw detections: {det.sum()}")
    
    # Clean up small speckle and apply morphological opening
    if cleanup_open and cleanup_open > 1:
        if fast_mode:
            # Skip expensive morphological operations in fast mode
            print("⚡ Skipping morphological cleanup in fast mode")
        else:
            det = opening(det, footprint_rectangle((cleanup_open, cleanup_open)))
    
    if min_area and min_area > 1:
        if fast_mode:
            # Use faster small object removal
            det = remove_small_objects(det, min_size=min_area)
        else:
            det = remove_small_objects(det, min_size=min_area)
    
    print(f"🧹 Final detections after cleanup: {det.sum()}")
    
    # Optional score (z-like)
    score = np.full_like(img, np.nan, dtype=np.float32)
    good = valid & (std > 0)
    score[good] = (img[good] - mean[good]) / std[good]
    
    print("✅ CFAR detection completed")
    return det, score


def masked_cfar_ultra_fast(
    img: np.ndarray,
    mask: np.ndarray,
    bg_radius: int = 8,  # Smaller for speed
    k: float = 2.5,
    min_valid: int = 20,  # Lower threshold
    min_area: int = 10,
):
    """
    Ultra-fast CFAR for initial testing.
    
    Uses simple sliding window approach with minimal processing.
    """
    print("🚀 ULTRA-FAST CFAR for initial testing...")
    
    img = img.astype(np.float32)
    M = (mask > 0).astype(np.float32)
    
    # Very simple approach: use uniform filter for mean and std
    kernel_size = 2 * bg_radius + 1
    
    # Fast mean and std using uniform filter
    mean = uniform_filter(img * M, size=kernel_size, mode='reflect')
    cnt = uniform_filter(M, size=kernel_size, mode='reflect')
    mean = mean / np.maximum(cnt, 1e-6)
    
    # Fast std approximation
    img_sq = uniform_filter((img**2) * M, size=kernel_size, mode='reflect')
    img_sq = img_sq / np.maximum(cnt, 1e-6)
    std = np.sqrt(np.maximum(img_sq - mean**2, 0.0) + 1e-6)
    
    # Simple thresholding
    valid = (M > 0) & (cnt >= min_valid)
    thr = mean + k * std
    det = (img > thr) & valid
    
    # Minimal cleanup
    det = remove_small_objects(det, min_size=min_area)
    
    print(f"⚡ Ultra-fast detections: {det.sum()}")
    return det, mean
