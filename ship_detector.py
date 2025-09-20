#!/usr/bin/env python3
"""
Ship Detection Pipeline using CFAR on Sentinel-2 data.

This module provides a complete pipeline for detecting ships in satellite imagery
using the Constant False Alarm Rate (CFAR) algorithm on preprocessed Sentinel-2 data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

# Import your modules
from icebreaker.utils.load import Sentinel2Preprocessor
from masked_cfar import masked_cfar


class ShipDetector:
    """
    Ship detection using CFAR algorithm on Sentinel-2 data.
    
    This class handles the complete ship detection pipeline:
    1. Load and preprocess Sentinel-2 data
    2. Create water masks for ship detection
    3. Apply CFAR detection algorithm
    4. Analyze and filter detections
    5. Generate results and visualizations
    """
    
    def __init__(self, config_path="icebreaker/config/settings.yaml"):
        """
        Initialize the ship detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.preprocessor = None
        self.processed_bands = {}
        self.detections = None
        self.scores = None
        self.ships = []
        
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)
    
    def create_water_mask(self, scl_band, water_classes=[6, 7]):
        """
        Create water mask from Scene Classification Layer (SCL).
        
        Args:
            scl_band: SCL band array
            water_classes: List of SCL class values for water (6=water, 7=water_vapour)
        
        Returns:
            Boolean mask where True = water pixels
        """
        water_mask = np.isin(scl_band, water_classes)
        
        # Optional: dilate water mask to include near-shore areas
        # This helps catch ships that might be partially on land due to classification errors
        water_mask = binary_dilation(water_mask, disk(2))
        
        return water_mask
    
    def create_ship_intensity_image(self, bands):
        """
        Create intensity image optimized for ship detection.
        
        Args:
            bands: Dictionary of processed band arrays
            
        Returns:
            Intensity image optimized for ship detection
        """
        # Method 1: Use NIR band (B08) - ships often appear bright in NIR
        if "B08" in bands:
            intensity = bands["B08"].astype(np.float32)
        else:
            # Fallback: use red band
            intensity = bands["B04"].astype(np.float32)
        
        # Method 2: Use NDVI-like index (uncomment to try)
        # if "B08" in bands and "B04" in bands:
        #     nir = bands["B08"].astype(np.float32)
        #     red = bands["B04"].astype(np.float32)
        #     intensity = (nir - red) / (nir + red + 1e-6)
        #     intensity = np.clip(intensity, 0, 1) * 1000  # Scale for CFAR
        
        # Method 3: Use RGB composite (uncomment to try)
        # if all(b in bands for b in ["B04", "B03", "B02"]):
        #     red = bands["B04"].astype(np.float32)
        #     green = bands["B03"].astype(np.float32)
        #     blue = bands["B02"].astype(np.float32)
        #     intensity = (red + green + blue) / 3.0
        
        return intensity
    
    def load_and_preprocess_data(self):
        """Load and preprocess Sentinel-2 data."""
        print("🛰️  Loading Sentinel-2 data...")
        
        # Initialize preprocessor
        safe_dir = Path(self.config["S2_DATA_ROOT"])
        self.preprocessor = Sentinel2Preprocessor(safe_dir)
        
        # Discover bands needed for ship detection
        bands_to_process = ["B02", "B03", "B04", "B08", "SCL"]
        bands = self.preprocessor.discover_bands(bands_to_process)
        print(f"✅ Discovered bands: {list(bands.keys())}")
        
        # Process bands to common resolution
        print("🔄 Processing bands to common resolution...")
        self.processed_bands = self.preprocessor.process_bands(
            band_names=bands_to_process,
            reference_band="B04",  # 10m reference
            aoi_geometry=self.config["AOI"]
        )
        
        print(f"📊 Band shapes: {[(k, v.shape) for k, v in self.processed_bands.items()]}")
        return self.processed_bands
    
    def detect_ships(self, 
                    bg_radius=20,      # Larger background for ships
                    guard_radius=5,    # Larger guard to avoid ship edges
                    k=2.5,             # Moderate threshold
                    min_valid=100,     # Need more valid pixels for ships
                    min_area=25,       # Ships are larger than small targets
                    cleanup_open=5):   # Larger cleanup for ship shapes
        """
        Detect ships using CFAR algorithm.
        
        Args:
            bg_radius: Background window radius
            guard_radius: Guard window radius
            k: Threshold multiplier
            min_valid: Minimum valid pixels for statistics
            min_area: Minimum ship area in pixels
            cleanup_open: Morphological opening size
        """
        print("🚢 Detecting ships with CFAR...")
        
        # Create water mask
        water_mask = self.create_water_mask(self.processed_bands["SCL"])
        print(f"🌊 Water pixels: {water_mask.sum()} / {water_mask.size} ({water_mask.mean()*100:.1f}%)")
        
        # Create intensity image
        intensity_img = self.create_ship_intensity_image(self.processed_bands)
        print(f"📈 Intensity range: {intensity_img.min():.2f} - {intensity_img.max():.2f}")
        
        # Run CFAR detection
        self.detections, self.scores = masked_cfar(
            img=intensity_img,
            mask=water_mask,
            bg_radius=bg_radius,
            guard_radius=guard_radius,
            k=k,
            min_valid=min_valid,
            min_area=min_area,
            use_log=False,  # Optical data, not SAR
            cleanup_open=cleanup_open
        )
        
        print(f"🎯 Raw detections: {self.detections.sum()}")
        
        # Analyze detections
        self._analyze_detections(intensity_img)
        
        return self.detections, self.scores
    
    def _analyze_detections(self, intensity_img):
        """Analyze detected objects and filter for ships."""
        print("📊 Analyzing detections...")
        
        # Label connected components
        labeled_detections = label(self.detections, connectivity=2)
        
        # Extract properties
        self.ships = []
        for region in regionprops(labeled_detections, intensity_image=intensity_img):
            # Filter for ship-like characteristics
            if self._is_ship_like(region):
                ship_info = {
                    "centroid_rc": region.centroid,  # (row, col)
                    "area_px": int(region.area),
                    "mean_intensity": float(region.mean_intensity),
                    "max_intensity": float(region.max_intensity),
                    "eccentricity": float(region.eccentricity),
                    "solidity": float(region.solidity),
                    "major_axis_length": float(region.major_axis_length),
                    "minor_axis_length": float(region.minor_axis_length),
                    "aspect_ratio": float(region.major_axis_length / region.minor_axis_length) if region.minor_axis_length > 0 else 0,
                }
                self.ships.append(ship_info)
        
        print(f"🚢 Ships detected: {len(self.ships)}")
        
        # Print ship summary
        if self.ships:
            areas = [ship["area_px"] for ship in self.ships]
            intensities = [ship["mean_intensity"] for ship in self.ships]
            print(f"   Area range: {min(areas)} - {max(areas)} pixels")
            print(f"   Intensity range: {min(intensities):.2f} - {max(intensities):.2f}")
    
    def _is_ship_like(self, region):
        """
        Filter detections to identify ship-like objects.
        
        Args:
            region: Region properties from regionprops
            
        Returns:
            True if the region is likely a ship
        """
        # Ship characteristics:
        # - Reasonable size (not too small, not too large)
        # - Elongated shape (ships are longer than wide)
        # - Solid shape (not too fragmented)
        
        min_area = 25      # Minimum ship size
        max_area = 2000    # Maximum ship size
        min_aspect_ratio = 1.2  # Ships are elongated
        min_solidity = 0.6      # Ships have solid shapes
        
        return (min_area <= region.area <= max_area and
                region.major_axis_length / region.minor_axis_length >= min_aspect_ratio and
                region.solidity >= min_solidity)
    
    def visualize_results(self, save_path="ship_detection_results.png"):
        """
        Create comprehensive visualization of ship detection results.
        
        Args:
            save_path: Path to save the visualization
        """
        print("📊 Creating visualization...")
        
        # Get bands for visualization
        red = self.processed_bands["B04"]
        green = self.processed_bands["B03"]
        blue = self.processed_bands["B02"]
        nir = self.processed_bands["B08"]
        
        # Create RGB composite
        rgb_composite = np.stack([red, green, blue], axis=-1) / 1000
        rgb_composite = np.clip(rgb_composite, 0, 1)
        
        # Create water mask for visualization
        water_mask = self.create_water_mask(self.processed_bands["SCL"])
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Input data
        axes[0, 0].imshow(rgb_composite)
        axes[0, 0].set_title("RGB Composite")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(nir, cmap='gray')
        axes[0, 1].set_title("NIR Band (Intensity)")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(water_mask, cmap='Blues', alpha=0.7)
        axes[0, 2].set_title("Water Mask")
        axes[0, 2].axis('off')
        
        # Row 2: Results
        axes[1, 0].imshow(self.scores, cmap='RdBu_r', vmin=-5, vmax=5)
        axes[1, 0].set_title("CFAR Scores")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.detections, cmap='Reds')
        axes[1, 1].set_title(f"Raw Detections ({self.detections.sum()} pixels)")
        axes[1, 1].axis('off')
        
        # Overlay detections on RGB
        rgb_overlay = rgb_composite.copy()
        rgb_overlay[self.detections] = [1, 0, 0]  # Red for detections
        
        # Mark ship centroids
        for ship in self.ships:
            row, col = ship["centroid_rc"]
            axes[1, 2].plot(col, row, 'yo', markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        axes[1, 2].imshow(rgb_overlay)
        axes[1, 2].set_title(f"Ships Detected ({len(self.ships)})")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {save_path}")
    
    def print_ship_summary(self):
        """Print detailed summary of detected ships."""
        print("\n" + "="*60)
        print("🚢 SHIP DETECTION SUMMARY")
        print("="*60)
        
        if not self.ships:
            print("No ships detected.")
            return
        
        for i, ship in enumerate(self.ships, 1):
            print(f"Ship {i}:")
            print(f"  Position: ({ship['centroid_rc'][0]:.1f}, {ship['centroid_rc'][1]:.1f})")
            print(f"  Area: {ship['area_px']} pixels")
            print(f"  Mean Intensity: {ship['mean_intensity']:.2f}")
            print(f"  Aspect Ratio: {ship['aspect_ratio']:.2f}")
            print(f"  Solidity: {ship['solidity']:.3f}")
            print(f"  Length: {ship['major_axis_length']:.1f} pixels")
            print()
    
    def run_complete_pipeline(self):
        """Run the complete ship detection pipeline."""
        print("🚀 Starting Ship Detection Pipeline")
        print("="*50)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Detect ships
            self.detect_ships()
            
            # Step 3: Visualize results
            self.visualize_results()
            
            # Step 4: Print summary
            self.print_ship_summary()
            
            print("✅ Ship detection pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run ship detection."""
    detector = ShipDetector()
    detector.run_complete_pipeline()


if __name__ == "__main__":
    main()
