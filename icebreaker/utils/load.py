"""
Sentinel-2 satellite data preprocessing module.

This module provides a class-based approach to processing Sentinel-2 satellite imagery,
including band discovery, resampling, clipping, and visualization capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import yaml
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry import shape

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sentinel2Preprocessor:
    """
    A comprehensive class for preprocessing Sentinel-2 satellite imagery.

    This class handles the complete preprocessing pipeline including:
    - Band discovery and validation
    - Resampling to common resolution
    - Area of Interest (AOI) clipping
    - Visualization of processing steps

    Attributes:
        safe_dir (Path): Path to the Sentinel-2 SAFE directory
        config (Dict): Configuration dictionary containing processing parameters
        bands (Dict[str, Path]): Dictionary mapping band names to file paths
    """

    SUPPORTED_BANDS = {
        "B01": ["*B01_20m.jp2", "*B01_60m.jp2"],  # Coastal aerosol
        "B02": ["*B02_10m.jp2", "*B02_20m.jp2", "*B02_60m.jp2"],  # Blue
        "B03": ["*B03_10m.jp2", "*B03_20m.jp2", "*B03_60m.jp2"],  # Green
        "B04": ["*B04_10m.jp2", "*B04_20m.jp2", "*B04_60m.jp2"],  # Red
        "B05": ["*B05_20m.jp2", "*B05_60m.jp2"],  # Red Edge 1
        "B06": ["*B06_20m.jp2", "*B06_60m.jp2"],  # Red Edge 2
        "B07": ["*B07_20m.jp2", "*B07_60m.jp2"],  # Red Edge 3
        "B08": ["*B08_10m.jp2"],  # NIR
        "B8A": ["*B8A_20m.jp2", "*B8A_60m.jp2"],  # NIR narrow
        "B09": ["*B09_60m.jp2"],  # Water vapour
        "B11": ["*B11_20m.jp2", "*B11_60m.jp2"],  # SWIR 1
        "B12": ["*B12_20m.jp2", "*B12_60m.jp2"],  # SWIR 2
        "SCL": ["*SCL_20m.jp2", "*SCL_60m.jp2"],  # Scene Classification
        "AOT": [
            "*AOT_10m.jp2",
            "*AOT_20m.jp2",
            "*AOT_60m.jp2",
        ],  # Aerosol Optical Thickness
        "WVP": ["*WVP_10m.jp2", "*WVP_20m.jp2", "*WVP_60m.jp2"],  # Water Vapour
        "TCI": ["*TCI_10m.jp2", "*TCI_20m.jp2", "*TCI_60m.jp2"],  # True Color Image
    }

    def __init__(
        self, safe_dir: Union[str, Path], config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the Sentinel2Preprocessor.

        Args:
            safe_dir: Path to the Sentinel-2 SAFE directory
            config_path: Optional path to YAML configuration file

        Raises:
            FileNotFoundError: If safe_dir doesn't exist
            ValueError: If GRANULE directory is not found in safe_dir
        """
        self.safe_dir = Path(safe_dir)
        self.config = self._load_config(config_path) if config_path else {}
        self.bands = {}

        self._validate_safe_directory()
        logger.info(f"Initialized Sentinel2Preprocessor for {self.safe_dir}")

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file

        Returns:
            Dictionary containing configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise

    def _validate_safe_directory(self) -> None:
        """
        Validate the SAFE directory structure.

        Raises:
            FileNotFoundError: If safe_dir doesn't exist
            ValueError: If required subdirectories are missing
        """
        if not self.safe_dir.exists():
            raise FileNotFoundError(f"SAFE directory not found: {self.safe_dir}")

        granule_dir = self.safe_dir / "GRANULE"
        if not granule_dir.exists():
            raise ValueError(f"GRANULE directory not found in {self.safe_dir}")

        granules = list(granule_dir.glob("*"))
        if not granules:
            raise ValueError(f"No granule subdirectories found in {granule_dir}")

    def discover_bands(
        self, bands_to_find: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Discover and validate Sentinel-2 band files.

        Args:
            bands_to_find: List of band names to find. If None, finds all supported bands.

        Returns:
            Dictionary mapping band names to their file paths

        Raises:
            ValueError: If no granules are found or required bands are missing
        """
        granules = list((self.safe_dir / "GRANULE").glob("*"))
        if not granules:
            raise ValueError("No GRANULE subdirectories found")

        # Use first granule (typically there's only one)
        granule = granules[0]

        # Define resolution directories
        resolution_dirs = [
            granule / "IMG_DATA" / "R10m",
            granule / "IMG_DATA" / "R20m",
            granule / "IMG_DATA" / "R60m",
        ]

        # Validate resolution directories exist
        existing_dirs = [d for d in resolution_dirs if d.exists()]
        if not existing_dirs:
            raise ValueError(f"No IMG_DATA resolution directories found in {granule}")

        bands_to_search = bands_to_find or list(self.SUPPORTED_BANDS.keys())
        discovered_bands = {}

        for band_name in bands_to_search:
            if band_name not in self.SUPPORTED_BANDS:
                logger.warning(f"Unknown band: {band_name}")
                continue

            band_path = self._find_band_file(
                existing_dirs, self.SUPPORTED_BANDS[band_name]
            )
            if band_path:
                discovered_bands[band_name] = band_path
                logger.debug(f"Found {band_name}: {band_path}")
            else:
                logger.warning(f"Band {band_name} not found")

        self.bands = discovered_bands
        logger.info(f"Discovered {len(discovered_bands)} bands")
        return discovered_bands

    def _find_band_file(
        self, resolution_dirs: List[Path], patterns: List[str]
    ) -> Optional[Path]:
        """
        Find a band file matching one of the given patterns in resolution directories.

        Args:
            resolution_dirs: List of resolution directories to search
            patterns: List of glob patterns to match

        Returns:
            Path to the found band file, or None if not found
        """
        for pattern in patterns:
            for res_dir in resolution_dirs:
                matches = list(res_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def resample_band_to_reference(
        self,
        band_path: Union[str, Path],
        reference_path: Union[str, Path],
        resampling_method: Resampling = Resampling.bilinear,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resample a band to match the resolution and extent of a reference band.

        Args:
            band_path: Path to the band to resample
            reference_path: Path to the reference band (typically 10m resolution)
            resampling_method: Resampling method to use

        Returns:
            Tuple of (resampled_array, reference_metadata)

        Raises:
            FileNotFoundError: If band files don't exist
            rasterio.errors.RasterioIOError: If files can't be read
        """
        band_path = Path(band_path)
        reference_path = Path(reference_path)

        if not band_path.exists():
            raise FileNotFoundError(f"Band file not found: {band_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        try:
            # Get reference properties
            with rio.open(reference_path) as ref:
                ref_meta = ref.meta.copy()
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_shape = (ref.height, ref.width)

            # Resample band to reference grid
            with rio.open(band_path) as src:
                resampled_data = np.empty(ref_shape, dtype=np.float32)
                rio.warp.reproject(
                    source=rio.band(src, 1),
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    dst_shape=ref_shape,
                    resampling=resampling_method,
                )

            logger.info(f"Resampled {band_path.name} to reference resolution")
            return resampled_data, ref_meta

        except Exception as e:
            logger.error(f"Error resampling {band_path}: {e}")
            raise

    def clip_to_area_of_interest(
        self, band_path: Union[str, Path], aoi_geometry: Dict
    ) -> Tuple[np.ndarray, rio.Affine, Dict]:
        """
        Clip a raster band to an Area of Interest (AOI).

        Args:
            band_path: Path to the band file to clip
            aoi_geometry: GeoJSON-like geometry dictionary (in EPSG:4326)

        Returns:
            Tuple of (clipped_array, transform, metadata)

        Raises:
            ValueError: If AOI geometry is invalid or empty
            FileNotFoundError: If band file doesn't exist
        """
        band_path = Path(band_path)
        if not band_path.exists():
            raise FileNotFoundError(f"Band file not found: {band_path}")

        try:
            # Parse and validate geometry
            geometry = self._parse_aoi_geometry(aoi_geometry)

            with rio.open(band_path) as src:
                raster_crs = src.crs

                # Reproject geometry to raster CRS
                gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
                gdf_projected = gdf.to_crs(raster_crs)

                # Convert to GeoJSON format for rasterio
                geometry_projected = [
                    json.loads(
                        gpd.GeoSeries(
                            [gdf_projected.iloc[0].geometry], crs=raster_crs
                        ).to_json()
                    )["features"][0]["geometry"]
                ]

                # Perform clipping
                clipped_image, clipped_transform = mask(
                    src, geometry_projected, crop=True
                )

                # Update metadata
                clipped_meta = src.meta.copy()
                clipped_meta.update(
                    {
                        "height": clipped_image.shape[1],
                        "width": clipped_image.shape[2],
                        "transform": clipped_transform,
                        "crs": raster_crs,
                    }
                )

            logger.info(f"Clipped {band_path.name} to AOI")
            return clipped_image, clipped_transform, clipped_meta

        except Exception as e:
            logger.error(f"Error clipping {band_path}: {e}")
            raise

    def _parse_aoi_geometry(self, aoi_geometry: Dict) -> shape:
        """
        Parse and validate AOI geometry from various GeoJSON formats.

        Args:
            aoi_geometry: GeoJSON-like geometry dictionary

        Returns:
            Shapely geometry object

        Raises:
            ValueError: If geometry format is invalid or geometry is empty
        """
        try:
            # Handle different GeoJSON formats
            if "features" in aoi_geometry:
                # FeatureCollection
                if not aoi_geometry["features"]:
                    raise ValueError("FeatureCollection has no features")
                geometry = shape(aoi_geometry["features"][0]["geometry"])
            elif "geometry" in aoi_geometry:
                # Feature
                geometry = shape(aoi_geometry["geometry"])
            elif "type" in aoi_geometry:
                # Direct geometry
                geometry = shape(aoi_geometry)
            else:
                raise ValueError("Invalid AOI geometry format")

            if geometry.is_empty:
                raise ValueError("AOI geometry is empty")

            return geometry

        except Exception as e:
            raise ValueError(f"Invalid AOI geometry: {e}")

    def visualize_processing_steps(
        self,
        original_path: Union[str, Path],
        resampled_array: np.ndarray,
        clipped_array: np.ndarray,
        figsize: Tuple[int, int] = (15, 5),
        cmap: str = "gray",
    ) -> None:
        """
        Visualize the processing steps side by side.

        Args:
            original_path: Path to the original band file
            resampled_array: Resampled array
            clipped_array: Clipped array (may be 3D)
            figsize: Figure size for matplotlib
            cmap: Colormap for visualization
        """
        try:
            # Read original image
            with rio.open(original_path) as src:
                original_array = src.read(1)

            # Handle 3D clipped array
            if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
                clipped_display = clipped_array[0]
            else:
                clipped_display = clipped_array

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            axes[0].imshow(original_array, cmap=cmap)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(resampled_array, cmap=cmap)
            axes[1].set_title("Resampled Image")
            axes[1].axis("off")

            axes[2].imshow(clipped_display, cmap=cmap)
            axes[2].set_title("Clipped Image")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

            logger.info("Generated processing visualization")

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise

    def get_band_info(self) -> Dict[str, Dict]:
        """
        Get information about discovered bands.

        Returns:
            Dictionary with band information including file paths and metadata
        """
        if not self.bands:
            logger.warning("No bands discovered. Run discover_bands() first.")
            return {}

        band_info = {}
        for band_name, band_path in self.bands.items():
            try:
                with rio.open(band_path) as src:
                    band_info[band_name] = {
                        "path": str(band_path),
                        "shape": (src.height, src.width),
                        "crs": str(src.crs),
                        "transform": src.transform,
                        "dtype": str(src.dtypes[0]),
                        "nodata": src.nodata,
                    }
            except Exception as e:
                logger.error(f"Error reading band {band_name}: {e}")
                band_info[band_name] = {"path": str(band_path), "error": str(e)}

        return band_info

    def process_bands(
        self,
        band_names: List[str],
        reference_band: str = "B04",
        aoi_geometry: Optional[Dict] = None,
        output_resolution: str = "10m",
    ) -> Dict[str, np.ndarray]:
        """
        Process multiple bands with resampling and optional clipping.

        Args:
            band_names: List of band names to process
            reference_band: Band to use as reference for resampling
            aoi_geometry: Optional AOI geometry for clipping
            output_resolution: Target resolution (for logging purposes)

        Returns:
            Dictionary mapping band names to processed arrays
        """
        if not self.bands:
            raise ValueError("No bands discovered. Run discover_bands() first.")

        if reference_band not in self.bands:
            raise ValueError(f"Reference band {reference_band} not found")

        processed_bands = {}
        reference_path = self.bands[reference_band]

        for band_name in band_names:
            if band_name not in self.bands:
                logger.warning(f"Band {band_name} not found, skipping")
                continue

            try:
                # Resample to reference
                resampled_array, _ = self.resample_band_to_reference(
                    self.bands[band_name], reference_path
                )

                # Clip if AOI provided
                if aoi_geometry:
                    clipped_array, _, _ = self.clip_to_area_of_interest(
                        self.bands[band_name], aoi_geometry
                    )
                    processed_bands[band_name] = clipped_array[0]  # Take first band
                else:
                    processed_bands[band_name] = resampled_array

                logger.info(f"Processed band {band_name}")

            except Exception as e:
                logger.error(f"Error processing band {band_name}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_bands)} bands")
        return processed_bands


def main():
    config_path = "/Users/jwl/Programming/dmz-hackathon/icebreaker/config/settings.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    safe_dir = Path(config["S2_DATA_ROOT"])
    preprocessor = Sentinel2Preprocessor(safe_dir)
    bands = preprocessor.discover_bands(["B02", "B03", "B04", "B08"])
    print(f"Discovered bands: {list(bands.keys())}")

    # Get band information
    band_info = preprocessor.get_band_info()
    for band, info in band_info.items():
        print(f"{band}: {info.get('shape', 'N/A')}")

    target_band = "B08"
    if target_band in bands:
        reference_band = "B04"  # 10m reference
        resampled_array, _ = preprocessor.resample_band_to_reference(
            bands[target_band], bands[reference_band]
        )

        aoi = config["AOI"]

        clipped_array, _, _ = preprocessor.clip_to_area_of_interest(
            bands[target_band], aoi
        )

        preprocessor.visualize_processing_steps(
            bands[target_band], resampled_array, clipped_array
        )


if __name__ == "__main__":
    main()
