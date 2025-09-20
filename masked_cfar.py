import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import opening, remove_small_objects
from skimage.morphology import footprint_rectangle


def masked_cfar(
    img: np.ndarray,
    mask: np.ndarray,
    bg_radius: int = 15,
    guard_radius: int = 3,
    k: float = 3.0,
    min_valid: int = 50,
    min_area: int = 12,
    use_log: bool = False,
    cleanup_open: int = 3,
):
    """
    Masked CFAR (bright-target) detector for ship detection.
    
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
    
    Returns
    -------
    det : 2D bool array
        Binary detection mask.
    score : 2D float array
        Optional score = (img - mean) / std where valid; NaN elsewhere.
    """
    img = img.astype(np.float32)
    M = (mask > 0).astype(np.float32)

    # Optional: log or dB domain helps stabilize SAR multiplicative noise
    if use_log:
        img = np.log1p(np.maximum(img, 0))  # log(1+I)

    # --- Build ring kernel: ones in background ring, zeros in guard + center ---
    ksize = 2 * bg_radius + 1
    K = np.ones((ksize, ksize), dtype=np.float32)
    gsize = 2 * guard_radius + 1
    c0 = bg_radius
    # zero out guard window (including center)
    K[c0-guard_radius:c0+guard_radius+1, c0-guard_radius:c0+guard_radius+1] = 0.0

    # --- Convolutions for masked sums and counts over ring ---
    # sum over ring of masked image
    sum_ring = convolve(img * M, K, mode="reflect")
    # sum of squares (for variance)
    sumsq_ring = convolve((img**2) * M, K, mode="reflect")
    # count of valid pixels contributing
    cnt_ring = convolve(M, K, mode="reflect")

    # --- Local mean and std (only where enough valid neighbors) ---
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_ring / np.maximum(cnt_ring, 1e-6)
        var = np.maximum(sumsq_ring / np.maximum(cnt_ring, 1e-6) - mean**2, 0.0)
        std = np.sqrt(var + 1e-6)

    # threshold only where: inside mask & enough valid pixels
    valid = (M > 0) & (cnt_ring >= min_valid)
    thr = mean + k * std
    det = (img > thr) & valid

    # --- Clean up small speckle and apply morphological opening ---
    if cleanup_open and cleanup_open > 1:
        det = opening(det, footprint_rectangle((cleanup_open, cleanup_open)))
    if min_area and min_area > 1:
        det = remove_small_objects(det, min_size=min_area)

    # Optional score (z-like)
    score = np.full_like(img, np.nan, dtype=np.float32)
    good = valid & (std > 0)
    score[good] = (img[good] - mean[good]) / std[good]

    return det, score