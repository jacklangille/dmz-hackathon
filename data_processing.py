"""
Data Processing Module for RX Ship Detection

This module contains all data processing functions for preparing multispectral
satellite data for ship detection using the RX anomaly detector.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import logging

from icebreaker.utils.load import Sentinel2Preprocessor
from rx_detector import create_multispectral_stack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def load_config(config_path: str = "icebreaker/config/settings.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def prepare_detection_data(config_path: str = "icebreaker/config/settings.yaml") -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare data for ship detection pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (multispectral_stack, water_mask, processed_bands)
    """
    logger.info("📊 Preparing detection data...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Step 1: Load and preprocess data
    processed_bands = load_and_preprocess_data(config, SENTINEL2_MULTISPECTRAL_BANDS)
    
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
    
    logger.info("✅ Data preparation completed")
    return multispectral_stack, water_mask, processed_bands
