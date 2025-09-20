"""
RX (Reed-Xiaoli) Anomaly Detector for Multispectral Ship Detection

The RX detector is a statistical anomaly detection algorithm that works well
for hyperspectral and multispectral data. It detects targets that are
statistically different from the background.

For ship detection, ships appear as spectral anomalies against the water background.
"""

import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter
from skimage.morphology import opening, remove_small_objects, footprint_rectangle
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RXDetector:
    """
    Reed-Xiaoli (RX) Anomaly Detector for multispectral data.
    
    The RX detector computes the Mahalanobis distance between each pixel
    and the background statistics to identify spectral anomalies.
    """
    
    def __init__(self, 
                 background_radius: int = 15,
                 guard_radius: int = 3,
                 min_valid_pixels: int = 50,
                 regularization: float = 1e-6,
                 fast_mode: bool = True):
        """
        Initialize RX detector.
        
        Args:
            background_radius: Radius of background window (pixels)
            guard_radius: Radius of guard window to exclude around test pixel
            min_valid_pixels: Minimum valid pixels for background statistics
            regularization: Regularization factor for covariance matrix
            fast_mode: Use fast approximation methods
        """
        self.background_radius = background_radius
        self.guard_radius = guard_radius
        self.min_valid_pixels = min_valid_pixels
        self.regularization = regularization
        self.fast_mode = fast_mode
        
        # Computed statistics (will be set during detection)
        self.background_mean = None
        self.background_cov = None
        self.background_cov_inv = None
        
    def _create_ring_mask(self, height: int, width: int, center_row: int, center_col: int) -> np.ndarray:
        """
        Create a ring mask for background sampling.
        
        Args:
            height, width: Image dimensions
            center_row, center_col: Center pixel coordinates
            background_radius: Outer radius of ring
            guard_radius: Inner radius of ring
            
        Returns:
            Boolean mask for background ring
        """
        # Create coordinate grids
        rows, cols = np.ogrid[:height, :width]
        
        # Distance from center
        distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        
        # Ring mask: inside background radius, outside guard radius
        ring_mask = (distances <= self.background_radius) & (distances > self.guard_radius)
        
        return ring_mask
    
    def _compute_background_stats(self, 
                                 multispectral_data: np.ndarray, 
                                 mask: np.ndarray,
                                 center_row: int, 
                                 center_col: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute background statistics for a single pixel.
        
        Args:
            multispectral_data: 3D array (height, width, bands)
            mask: 2D boolean mask for valid pixels
            center_row, center_col: Center pixel coordinates
            
        Returns:
            Tuple of (mean_vector, covariance_matrix)
        """
        height, width, n_bands = multispectral_data.shape
        
        # Create ring mask
        ring_mask = self._create_ring_mask(height, width, center_row, center_col)
        
        # Combine with validity mask
        valid_mask = ring_mask & mask
        
        # Extract background pixels
        background_pixels = multispectral_data[valid_mask]  # Shape: (n_pixels, n_bands)
        
        if len(background_pixels) < self.min_valid_pixels:
            # Not enough valid pixels
            return None, None
        
        # Compute mean and covariance
        mean_vector = np.mean(background_pixels, axis=0)
        
        # Center the data
        centered_data = background_pixels - mean_vector
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)
        
        # Add regularization for numerical stability
        cov_matrix += self.regularization * np.eye(n_bands)
        
        return mean_vector, cov_matrix
    
    def _compute_rx_score(self, 
                         pixel_spectrum: np.ndarray, 
                         mean_vector: np.ndarray, 
                         cov_inv: np.ndarray) -> float:
        """
        Compute RX score (Mahalanobis distance) for a pixel.
        
        Args:
            pixel_spectrum: Spectral vector for the pixel
            mean_vector: Background mean vector
            cov_inv: Inverse of background covariance matrix
            
        Returns:
            RX score (higher = more anomalous)
        """
        # Center the pixel spectrum
        centered_spectrum = pixel_spectrum - mean_vector
        
        # Compute Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
        rx_score = np.dot(centered_spectrum, np.dot(cov_inv, centered_spectrum))
        
        return rx_score
    
    def detect_anomalies(self, 
                        multispectral_data: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using RX detector.
        
        Args:
            multispectral_data: 3D array (height, width, bands)
            mask: Optional 2D boolean mask for valid pixels
            
        Returns:
            Tuple of (detection_scores, binary_detections)
        """
        height, width, n_bands = multispectral_data.shape
        
        if mask is None:
            mask = np.ones((height, width), dtype=bool)
        
        # Initialize output arrays
        rx_scores = np.full((height, width), np.nan, dtype=np.float32)
        detections = np.zeros((height, width), dtype=bool)
        
        logger.info(f"Starting RX detection on {height}x{width} image with {n_bands} bands")
        
        # Process each pixel
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    continue
                
                # Get pixel spectrum
                pixel_spectrum = multispectral_data[row, col, :]
                
                # Compute background statistics
                mean_vector, cov_matrix = self._compute_background_stats(
                    multispectral_data, mask, row, col
                )
                
                if mean_vector is None or cov_matrix is None:
                    continue
                
                # Compute inverse covariance matrix
                try:
                    cov_inv = linalg.inv(cov_matrix)
                except linalg.LinAlgError:
                    # Singular matrix, skip this pixel
                    continue
                
                # Compute RX score
                rx_score = self._compute_rx_score(pixel_spectrum, mean_vector, cov_inv)
                rx_scores[row, col] = rx_score
        
        logger.info("RX detection completed")
        return rx_scores, detections
    
    def detect_anomalies_fast(self, 
                             multispectral_data: np.ndarray, 
                             mask: Optional[np.ndarray] = None,
                             threshold_percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast RX detection using global background statistics.
        
        This is much faster than pixel-wise detection but less accurate.
        Good for initial screening and large images.
        
        Args:
            multispectral_data: 3D array (height, width, bands)
            mask: Optional 2D boolean mask for valid pixels
            threshold_percentile: Percentile for detection threshold
            
        Returns:
            Tuple of (detection_scores, binary_detections)
        """
        height, width, n_bands = multispectral_data.shape
        
        if mask is None:
            mask = np.ones((height, width), dtype=bool)
        
        logger.info(f"Starting fast RX detection on {height}x{width} image with {n_bands} bands")
        
        # Reshape data for easier processing
        data_2d = multispectral_data.reshape(-1, n_bands)  # (height*width, bands)
        mask_1d = mask.reshape(-1)
        
        # Get valid pixels
        valid_pixels = data_2d[mask_1d]
        
        if len(valid_pixels) < self.min_valid_pixels:
            logger.warning("Not enough valid pixels for background statistics")
            return np.full((height, width), np.nan), np.zeros((height, width), dtype=bool)
        
        # Compute global background statistics
        self.background_mean = np.mean(valid_pixels, axis=0)
        centered_data = valid_pixels - self.background_mean
        
        # Compute global covariance matrix
        self.background_cov = np.cov(centered_data.T)
        self.background_cov += self.regularization * np.eye(n_bands)
        
        # Compute inverse covariance matrix
        try:
            self.background_cov_inv = linalg.inv(self.background_cov)
        except linalg.LinAlgError:
            logger.error("Singular covariance matrix - cannot compute inverse")
            return np.full((height, width), np.nan), np.zeros((height, width), dtype=bool)
        
        # Compute RX scores for all pixels
        centered_all = data_2d - self.background_mean
        rx_scores_1d = np.sum(centered_all @ self.background_cov_inv * centered_all, axis=1)
        
        # Reshape back to image dimensions
        rx_scores = rx_scores_1d.reshape(height, width)
        
        # Set invalid pixels to NaN
        rx_scores[~mask] = np.nan
        
        # Compute threshold
        valid_scores = rx_scores[mask]
        threshold = np.percentile(valid_scores, threshold_percentile)
        
        # Create binary detections
        detections = (rx_scores > threshold) & mask
        
        logger.info(f"Fast RX detection completed. Threshold: {threshold:.2f}")
        logger.info(f"Detected {detections.sum()} anomalous pixels")
        
        return rx_scores, detections
    
    def detect_anomalies_adaptive(self, 
                                 multispectral_data: np.ndarray, 
                                 mask: Optional[np.ndarray] = None,
                                 window_size: int = 31) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive RX detection using local background statistics.
        
        Computes background statistics in local windows for better
        adaptation to varying background conditions.
        
        Args:
            multispectral_data: 3D array (height, width, bands)
            mask: Optional 2D boolean mask for valid pixels
            window_size: Size of local window for background statistics
            
        Returns:
            Tuple of (detection_scores, binary_detections)
        """
        height, width, n_bands = multispectral_data.shape
        
        if mask is None:
            mask = np.ones((height, width), dtype=bool)
        
        logger.info(f"Starting adaptive RX detection on {height}x{width} image with {n_bands} bands")
        
        # Initialize output arrays
        rx_scores = np.full((height, width), np.nan, dtype=np.float32)
        
        # Process in local windows
        half_window = window_size // 2
        
        for row in range(half_window, height - half_window):
            for col in range(half_window, width - half_window):
                if not mask[row, col]:
                    continue
                
                # Define local window
                row_start = row - half_window
                row_end = row + half_window + 1
                col_start = col - half_window
                col_end = col + half_window + 1
                
                # Extract local data
                local_data = multispectral_data[row_start:row_end, col_start:col_end, :]
                local_mask = mask[row_start:row_end, col_start:col_end]
                
                # Reshape for processing
                local_data_2d = local_data.reshape(-1, n_bands)
                local_mask_1d = local_mask.reshape(-1)
                
                # Get valid pixels in local window
                valid_pixels = local_data_2d[local_mask_1d]
                
                if len(valid_pixels) < self.min_valid_pixels:
                    continue
                
                # Compute local background statistics
                local_mean = np.mean(valid_pixels, axis=0)
                centered_data = valid_pixels - local_mean
                
                # Compute local covariance matrix
                local_cov = np.cov(centered_data.T)
                local_cov += self.regularization * np.eye(n_bands)
                
                # Compute inverse covariance matrix
                try:
                    local_cov_inv = linalg.inv(local_cov)
                except linalg.LinAlgError:
                    continue
                
                # Compute RX score for center pixel
                center_pixel = multispectral_data[row, col, :]
                rx_score = self._compute_rx_score(center_pixel, local_mean, local_cov_inv)
                rx_scores[row, col] = rx_score
        
        # Compute threshold and create detections
        valid_scores = rx_scores[mask]
        if len(valid_scores) > 0:
            threshold = np.percentile(valid_scores, 99.5)
            detections = (rx_scores > threshold) & mask
        else:
            detections = np.zeros((height, width), dtype=bool)
        
        logger.info("Adaptive RX detection completed")
        return rx_scores, detections


def post_process_detections(detections: np.ndarray, 
                           min_area: int = 25,
                           cleanup_size: int = 3) -> np.ndarray:
    """
    Post-process RX detections to remove noise and small objects.
    
    Args:
        detections: Binary detection mask
        min_area: Minimum area for valid detections
        cleanup_size: Size of morphological opening kernel
        
    Returns:
        Cleaned detection mask
    """
    # Morphological opening to remove noise
    if cleanup_size > 1:
        detections = opening(detections, footprint_rectangle((cleanup_size, cleanup_size)))
    
    # Remove small objects
    if min_area > 1:
        detections = remove_small_objects(detections, min_size=min_area)
    
    return detections


def create_multispectral_stack(bands: Dict[str, np.ndarray], 
                              band_order: Optional[list] = None) -> np.ndarray:
    """
    Create multispectral stack from individual bands.
    
    Args:
        bands: Dictionary mapping band names to arrays
        band_order: Optional list specifying band order
        
    Returns:
        3D array (height, width, bands)
    """
    if band_order is None:
        band_order = sorted(bands.keys())
    
    # Stack bands
    multispectral_stack = np.stack([bands[band] for band in band_order], axis=-1)
    
    return multispectral_stack
