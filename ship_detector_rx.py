"""
Ship Detection using RX (Reed-Xiaoli) Anomaly Detector

This module implements ship detection using the RX detector on multispectral
8-band satellite data. Ships appear as spectral anomalies against the water background.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

from icebreaker.utils.load import Sentinel2Preprocessor
from rx_detector import RXDetector, post_process_detections, create_multispectral_stack
from skimage.measure import label, regionprops

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_multispectral_stack_from_bands(processed_bands: Dict[str, np.ndarray], 
                                        band_order: List[str]) -> np.ndarray:
    """
    Create multispectral stack from processed bands.
    
    Args:
        processed_bands: Dictionary mapping band names to arrays
        band_order: List specifying band order
        
    Returns:
        3D array (height, width, n_bands)
    """
    logger.info("Creating multispectral stack...")
    
    # Create stack in wavelength order
    multispectral_stack = create_multispectral_stack(
        processed_bands, 
        band_order=band_order
    )
    
    logger.info(f"Multispectral stack shape: {multispectral_stack.shape}")
    logger.info(f"Data range: {multispectral_stack.min():.2f} - {multispectral_stack.max():.2f}")
    
    return multispectral_stack


def create_water_mask_from_scl(scl_band: np.ndarray) -> np.ndarray:
    """
    Create water mask from Scene Classification Layer.
    
    Args:
        scl_band: Scene Classification Layer band
        
    Returns:
        Boolean water mask
    """
    # Water classes in SCL
    water_classes = [6, 7, 10, 11]  # Water, water vapor, etc.
    
    # Create water mask
    water_mask = np.isin(scl_band, water_classes)
    
    # Add small dilation to include near-shore areas
    from skimage.morphology import binary_dilation, disk
    water_mask = binary_dilation(water_mask, disk(2))
    
    logger.info(f"Water pixels: {water_mask.sum()} / {water_mask.size} ({water_mask.mean()*100:.1f}%)")
    
    return water_mask


def create_fallback_water_mask(shape: tuple) -> np.ndarray:
    """
    Create a fallback water mask when SCL band is not available.
    
    Args:
        shape: Shape of the image (height, width)
        
    Returns:
        Boolean mask with all pixels set to True
    """
    logger.warning("No SCL band found, using full image mask")
    return np.ones(shape, dtype=bool)


def load_and_preprocess_data(config: Dict, multispectral_bands: List[str]) -> Dict[str, np.ndarray]:
    """
    Load and preprocess multispectral data.
    
    Args:
        config: Configuration dictionary
        multispectral_bands: List of band names to process
        
    Returns:
        Dictionary of processed bands
    """
    logger.info("Loading and preprocessing multispectral data...")
    
    # Initialize preprocessor
    safe_dir = Path(config["S2_DATA_ROOT"])
    preprocessor = Sentinel2Preprocessor(safe_dir, "icebreaker/config/settings.yaml")
    
    # Discover bands
    bands = preprocessor.discover_bands(multispectral_bands)
    logger.info(f"Discovered bands: {list(bands.keys())}")
    
    # Process bands to common resolution and clip to AOI
    processed_bands = preprocessor.process_bands(
        band_names=multispectral_bands,
        reference_band="B04",  # Use B04 as 10m reference
        aoi_geometry=config["AOI"],
        output_resolution="10m"
    )
    
    # Validate and fix any shape inconsistencies
    processed_bands = ensure_consistent_shapes(processed_bands)
    
    logger.info(f"Processed {len(processed_bands)} bands")
    logger.info(f"Band shapes: {[(k, v.shape) for k, v in processed_bands.items()]}")
    
    return processed_bands


def ensure_consistent_shapes(processed_bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure all processed bands have consistent shapes by resizing if necessary.
    
    Args:
        processed_bands: Dictionary of processed bands
        
    Returns:
        Dictionary of bands with consistent shapes
    """
    logger.info("Ensuring consistent band shapes...")
    
    # Find the most common shape (should be the target shape)
    shapes = {name: band.shape for name, band in processed_bands.items()}
    shape_counts = {}
    for shape in shapes.values():
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    # Get the most common shape
    target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
    logger.info(f"Target shape: {target_shape}")
    
    # Resize bands that don't match the target shape
    from scipy.ndimage import zoom
    
    consistent_bands = {}
    for band_name, band_array in processed_bands.items():
        if band_array.shape == target_shape:
            consistent_bands[band_name] = band_array
            logger.info(f"Band {band_name} already has target shape: {band_array.shape}")
        else:
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / band_array.shape[0], 
                          target_shape[1] / band_array.shape[1])
            
            # Resize the band
            resized_band = zoom(band_array, zoom_factors, order=1)  # Linear interpolation
            consistent_bands[band_name] = resized_band
            logger.info(f"Band {band_name} resized from {band_array.shape} to {resized_band.shape}")
    
    # Final validation
    final_shapes = {name: band.shape for name, band in consistent_bands.items()}
    unique_shapes = set(final_shapes.values())
    
    if len(unique_shapes) == 1:
        logger.info(f"✅ All bands now have consistent shape: {list(unique_shapes)[0]}")
    else:
        logger.error(f"❌ Shape consistency failed: {final_shapes}")
        raise ValueError(f"Failed to ensure consistent shapes: {final_shapes}")
    
    return consistent_bands


# Sentinel-2 multispectral bands configuration
SENTINEL2_MULTISPECTRAL_BANDS = [
    "B01",  # Coastal aerosol (443nm)
    "B02",  # Blue (490nm)
    "B03",  # Green (560nm)
    "B04",  # Red (665nm)
    "B05",  # Red Edge 1 (705nm)
    "B06",  # Red Edge 2 (740nm)
    "B07",  # Red Edge 3 (783nm)
    "B08",  # NIR (842nm)
]


class ShipDetectorRX:
    """
    Ship detector using RX anomaly detection on multispectral data.
    
    This detector leverages the spectral characteristics of ships vs water
    to identify ships as spectral anomalies in 8-band multispectral imagery.
    """
    
    def __init__(self, config_path: str = "icebreaker/config/settings.yaml"):
        """
        Initialize ship detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.preprocessor = None
        self.multispectral_stack = None
        self.water_mask = None
        self.rx_scores = None
        self.detections = None
        self.ships = []
        
        # RX detector parameters
        self.rx_detector = RXDetector(
            background_radius=15,
            guard_radius=3,
            min_valid_pixels=50,
            regularization=1e-6,
            fast_mode=True
        )
        
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    
    
    
    def detect_ships(self, 
                    multispectral_stack: np.ndarray,
                    water_mask: np.ndarray,
                    detection_mode: str = "fast",
                    threshold_percentile: float = 99.5,
                    min_area: int = 25,
                    cleanup_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ships using RX anomaly detection.
        
        Args:
            multispectral_stack: 3D array (height, width, bands)
            water_mask: 2D boolean mask for valid pixels
            detection_mode: "fast", "adaptive", or "pixel_wise"
            threshold_percentile: Percentile for detection threshold
            min_area: Minimum area for valid detections
            cleanup_size: Size of morphological cleanup
            
        Returns:
            Tuple of (rx_scores, binary_detections)
        """
        logger.info(f"Detecting ships using RX detector (mode: {detection_mode})...")
        
        # Run RX detection
        if detection_mode == "fast":
            self.rx_scores, self.detections = self.rx_detector.detect_anomalies_fast(
                multispectral_stack,
                mask=water_mask,
                threshold_percentile=threshold_percentile
            )
        elif detection_mode == "adaptive":
            self.rx_scores, self.detections = self.rx_detector.detect_anomalies_adaptive(
                multispectral_stack,
                mask=water_mask
            )
        else:  # pixel_wise
            self.rx_scores, self.detections = self.rx_detector.detect_anomalies(
                multispectral_stack,
                mask=water_mask
            )
        
        # Post-process detections
        self.detections = post_process_detections(
            self.detections,
            min_area=min_area,
            cleanup_size=cleanup_size
        )
        
        logger.info(f"Raw detections: {self.detections.sum()} pixels")
        
        # Analyze detections
        self._analyze_detections()
        
        return self.rx_scores, self.detections
    
    def _analyze_detections(self):
        """Analyze detected objects and filter for ships."""
        logger.info("Analyzing detections...")
        
        # Label connected components
        labeled_detections = label(self.detections, connectivity=2)
        
        # Extract properties
        self.ships = []
        for region in regionprops(labeled_detections):
            # Filter for ship-like characteristics
            if self._is_ship_like(region):
                ship_info = {
                    "centroid_rc": region.centroid,  # (row, col)
                    "area_px": int(region.area),
                    "eccentricity": float(region.eccentricity),
                    "solidity": float(region.solidity),
                    "major_axis_length": float(region.major_axis_length),
                    "minor_axis_length": float(region.minor_axis_length),
                    "aspect_ratio": float(region.major_axis_length / region.minor_axis_length) if region.minor_axis_length > 0 else 0,
                }
                self.ships.append(ship_info)
        
        logger.info(f"Ships detected: {len(self.ships)}")
        
        # Print ship summary
        if self.ships:
            areas = [ship["area_px"] for ship in self.ships]
            aspect_ratios = [ship["aspect_ratio"] for ship in self.ships]
            logger.info(f"   Area range: {min(areas)} - {max(areas)} pixels")
            logger.info(f"   Aspect ratio range: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f}")
    
    def _is_ship_like(self, region) -> bool:
        """
        Filter detections to identify ship-like objects.
        
        Args:
            region: Region properties from regionprops
            
        Returns:
            True if region is ship-like
        """
        # Ship filtering criteria
        min_area = 25
        max_area = 2000
        min_aspect_ratio = 1.2
        min_solidity = 0.6
        
        return (min_area <= region.area <= max_area and
                region.major_axis_length / region.minor_axis_length >= min_aspect_ratio and
                region.solidity >= min_solidity)
    
    def visualize_results(self, 
                         processed_bands: Dict[str, np.ndarray],
                         multispectral_stack: np.ndarray,
                         water_mask: np.ndarray,
                         save_path: str = "ship_detection_rx_results.png"):
        """
        Create comprehensive visualization of RX ship detection results.
        
        Args:
            processed_bands: Dictionary of processed bands
            multispectral_stack: 3D array (height, width, bands)
            water_mask: 2D boolean mask for valid pixels
            save_path: Path to save the visualization
        """
        logger.info("Creating visualization...")
        
        # Get bands for visualization
        red = processed_bands["B04"]
        green = processed_bands["B03"]
        blue = processed_bands["B02"]
        nir = processed_bands["B08"]
        
        # Create RGB composite
        rgb_composite = np.stack([red, green, blue], axis=-1) / 1000
        rgb_composite = np.clip(rgb_composite, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Row 1: Input data
        axes[0, 0].imshow(rgb_composite)
        axes[0, 0].set_title("RGB Composite")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(nir, cmap='gray')
        axes[0, 1].set_title("NIR Band")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(water_mask, cmap='Blues', alpha=0.7)
        axes[0, 2].set_title("Water Mask")
        axes[0, 2].axis('off')
        
        # Row 2: Multispectral bands
        axes[1, 0].imshow(processed_bands["B01"], cmap='gray')
        axes[1, 0].set_title("B01 (Coastal)")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(processed_bands["B05"], cmap='gray')
        axes[1, 1].set_title("B05 (Red Edge 1)")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(processed_bands["B08"], cmap='gray')
        axes[1, 2].set_title("B08 (NIR)")
        axes[1, 2].axis('off')
        
        # Row 3: Results
        # RX scores
        valid_scores = self.rx_scores[~np.isnan(self.rx_scores)]
        if len(valid_scores) > 0:
            vmin, vmax = np.percentile(valid_scores, [1, 99])
            axes[2, 0].imshow(self.rx_scores, cmap='hot', vmin=vmin, vmax=vmax)
        else:
            axes[2, 0].imshow(self.rx_scores, cmap='hot')
        axes[2, 0].set_title("RX Scores")
        axes[2, 0].axis('off')
        
        # Raw detections
        axes[2, 1].imshow(self.detections, cmap='Reds')
        axes[2, 1].set_title(f"Raw Detections ({self.detections.sum()} pixels)")
        axes[2, 1].axis('off')
        
        # Ships overlay
        rgb_overlay = rgb_composite.copy()
        rgb_overlay[self.detections] = [1, 0, 0]  # Red for detections
        
        axes[2, 2].imshow(rgb_overlay)
        axes[2, 2].set_title(f"Ships Detected ({len(self.ships)})")
        axes[2, 2].axis('off')
        
        # Mark ship centroids
        for ship in self.ships:
            row, col = ship["centroid_rc"]
            axes[2, 2].plot(col, row, 'yo', markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    def print_ship_summary(self):
        """Print detailed summary of detected ships."""
        print("\n" + "="*60)
        print("🚢 RX SHIP DETECTION SUMMARY")
        print("="*60)
        
        if not self.ships:
            print("No ships detected.")
            return
        
        for i, ship in enumerate(self.ships, 1):
            print(f"\nShip {i}:")
            print(f"  Position: ({ship['centroid_rc'][0]:.1f}, {ship['centroid_rc'][1]:.1f})")
            print(f"  Area: {ship['area_px']} pixels")
            print(f"  Aspect Ratio: {ship['aspect_ratio']:.2f}")
            print(f"  Solidity: {ship['solidity']:.2f}")
            print(f"  Length: {ship['major_axis_length']:.1f} pixels")
            print(f"  Width: {ship['minor_axis_length']:.1f} pixels")
    
    def run_complete_pipeline(self, 
                             detection_mode: str = "fast",
                             threshold_percentile: float = 99.5) -> None:
        """
        Run the complete RX ship detection pipeline.
        
        Args:
            detection_mode: RX detection mode ("fast", "adaptive", "pixel_wise")
            threshold_percentile: Detection threshold percentile
        """
        logger.info("🚀 Starting RX Ship Detection Pipeline")
        logger.info("="*50)
        
        try:
            # Step 1: Load and preprocess data
            processed_bands = load_and_preprocess_data(self.config, SENTINEL2_MULTISPECTRAL_BANDS)
            
            # Step 2: Create multispectral stack
            multispectral_stack = create_multispectral_stack_from_bands(
                processed_bands, 
                SENTINEL2_MULTISPECTRAL_BANDS
            )
            
            # Step 3: Create water mask
            if "SCL" in processed_bands:
                water_mask = create_water_mask_from_scl(processed_bands["SCL"])
            else:
                # Fallback: assume all pixels are valid
                water_mask = create_fallback_water_mask(multispectral_stack.shape[:2])
            
            # Step 4: Detect ships using RX
            self.detect_ships(
                multispectral_stack=multispectral_stack,
                water_mask=water_mask,
                detection_mode=detection_mode,
                threshold_percentile=threshold_percentile
            )
            
            # Step 5: Visualize results
            self.visualize_results(
                processed_bands=processed_bands,
                multispectral_stack=multispectral_stack,
                water_mask=water_mask
            )
            
            # Step 6: Print summary
            self.print_ship_summary()
            
            logger.info("✅ RX ship detection pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run RX ship detection."""
    detector = ShipDetectorRX()
    detector.run_complete_pipeline(detection_mode="fast", threshold_percentile=99.5)


if __name__ == "__main__":
    main()
