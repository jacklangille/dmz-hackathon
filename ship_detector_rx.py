"""
Ship Detection using RX (Reed-Xiaoli) Anomaly Detector

This module implements ship detection using the RX detector on multispectral
8-band satellite data. Ships appear as spectral anomalies against the water background.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from rx_detector import RXDetector, post_process_detections
from skimage.measure import label, regionprops
from data_processing import prepare_detection_data


class ShipDetectorRX:
    """
    Ship detector using RX anomaly detection on multispectral data.
    
    This detector leverages the spectral characteristics of ships vs water
    to identify ships as spectral anomalies in 8-band multispectral imagery.
    """
    
    def __init__(self):
        """
        Initialize ship detector.
        """
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
        
        # Analyze detections
        self._analyze_detections()
        
        return self.rx_scores, self.detections
    
    def _analyze_detections(self):
        """Analyze detected objects and filter for ships."""
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
    
    def create_ship_mask(self) -> np.ndarray:
        """
        Create a binary mask for detected ships.
        
        Returns:
            Binary mask where 1 indicates a detected ship pixel and 0 otherwise
        """
        if self.detections is None:
            raise ValueError("No detections available. Run detect_ships() first.")
        
        # Start with the raw detections
        ship_mask = self.detections.copy().astype(np.uint8)
        
        # If we have analyzed ships, create a mask only for ship-like objects
        if self.ships:
            # Create a new mask initialized to zeros
            ship_mask = np.zeros_like(self.detections, dtype=np.uint8)
            
            # Label connected components in the raw detections
            labeled_detections = label(self.detections, connectivity=2)
            
            # Get properties for all regions
            regions = regionprops(labeled_detections)
            
            # Fill the mask only for ship-like regions
            for i, region in enumerate(regions):
                if self._is_ship_like(region):
                    # Get the coordinates of this region
                    coords = region.coords
                    ship_mask[coords[:, 0], coords[:, 1]] = 1
        
        return ship_mask
    
    def save_ship_mask(self, filepath: str = "ship_mask.png") -> None:
        """
        Save the ship mask to a file.
        
        Args:
            filepath: Path where to save the ship mask image
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        
        ship_mask = self.create_ship_mask()
        
        # Create discrete colormap for binary values
        colors = ['white', 'red']  # 0 = white (background), 1 = red (ship)
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(ship_mask, cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Ship Detection Mask ({ship_mask.sum()} pixels)")
        
        # Create legend with labeled patches
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Background'),
            Patch(facecolor='red', label='Ship')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_ship_coordinates(self) -> list:
        """
        Get the coordinates of detected ships.
        
        Returns:
            List of dictionaries containing ship coordinates and properties.
            Each dictionary contains:
            - 'centroid': (row, col) coordinates of ship center
            - 'pixel_coordinates': List of (row, col) tuples for all ship pixels
            - 'area': Number of pixels in the ship
            - 'bounding_box': (min_row, min_col, max_row, max_col) bounding box
        """
        if self.ships is None or len(self.ships) == 0:
            return []
        
        ship_coordinates = []
        
        # Label connected components in the raw detections
        labeled_detections = label(self.detections, connectivity=2)
        
        # Get properties for all regions
        regions = regionprops(labeled_detections)
        
        # Extract coordinates for ship-like regions
        for i, region in enumerate(regions):
            if self._is_ship_like(region):
                # Get all pixel coordinates for this ship
                pixel_coords = region.coords  # (row, col) coordinates
                
                # Get bounding box
                min_row, min_col, max_row, max_col = region.bbox
                
                ship_info = {
                    'centroid': region.centroid,  # (row, col) of center
                    'pixel_coordinates': pixel_coords.tolist(),  # All ship pixels
                    'area': int(region.area),
                    'bounding_box': (min_row, min_col, max_row, max_col)
                }
                ship_coordinates.append(ship_info)
        
        return ship_coordinates
    
    def get_ship_centroids(self) -> list:
        """
        Get just the centroid coordinates of detected ships.
        
        Returns:
            List of (row, col) tuples representing ship centroids
        """
        ship_coords = self.get_ship_coordinates()
        return [ship['centroid'] for ship in ship_coords]
    
    def get_ship_bounding_boxes(self) -> list:
        """
        Get the bounding boxes of detected ships.
        
        Returns:
            List of (min_row, min_col, max_row, max_col) tuples
        """
        ship_coords = self.get_ship_coordinates()
        return [ship['bounding_box'] for ship in ship_coords]
    
    def save_ship_coordinates(self, filepath: str = "ship_coordinates.json") -> None:
        """
        Save ship coordinates to a JSON file.
        
        Args:
            filepath: Path where to save the coordinates file
        """
        import json
        
        ship_coords = self.get_ship_coordinates()
        
        # Convert numpy arrays to lists for JSON serialization
        json_coords = []
        for ship in ship_coords:
            json_ship = {
                'centroid': [float(ship['centroid'][0]), float(ship['centroid'][1])],
                'pixel_coordinates': ship['pixel_coordinates'],
                'area': ship['area'],
                'bounding_box': ship['bounding_box']
            }
            json_coords.append(json_ship)
        
        with open(filepath, 'w') as f:
            json.dump(json_coords, f, indent=2)
    
    def visualize_results(self, 
                         multispectral_stack: np.ndarray,
                         water_mask: np.ndarray,
                         save_path: str = "ship_detection_rx_results.png"):
        """
        Create comprehensive visualization of RX ship detection results.
        
        Args:
            multispectral_stack: 3D array (height, width, bands) in order [B01, B02, B03, B04, B05, B06, B07, B08]
            water_mask: 2D boolean mask for valid pixels
            save_path: Path to save the visualization
        """
        # Extract bands from multispectral stack (order: B01, B02, B03, B04, B05, B06, B07, B08)
        b01 = multispectral_stack[:, :, 0]  # Coastal aerosol
        b02 = multispectral_stack[:, :, 1]  # Blue
        b03 = multispectral_stack[:, :, 2]  # Green
        b04 = multispectral_stack[:, :, 3]  # Red
        b05 = multispectral_stack[:, :, 4]  # Red Edge 1
        b06 = multispectral_stack[:, :, 5]  # Red Edge 2
        b07 = multispectral_stack[:, :, 6]  # Red Edge 3
        b08 = multispectral_stack[:, :, 7]  # NIR
        
        # Get bands for visualization
        red = b04
        green = b03
        blue = b02
        nir = b08
        
        # Create RGB composite
        rgb_composite = np.stack([red, green, blue], axis=-1) / 1000
        rgb_composite = np.clip(rgb_composite, 0, 1)
        
        # Create visualization
        _, axes = plt.subplots(3, 3, figsize=(18, 18))
        
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
        axes[1, 0].imshow(b01, cmap='gray')
        axes[1, 0].set_title("B01 (Coastal)")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(b05, cmap='gray')
        axes[1, 1].set_title("B05 (Red Edge 1)")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(b08, cmap='gray')
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
        
        # Ship mask
        ship_mask = self.create_ship_mask()
        axes[2, 1].imshow(ship_mask, cmap='Reds')
        axes[2, 1].set_title(f"Ship Mask ({ship_mask.sum()} pixels)")
        axes[2, 1].axis('off')
        
        # Ships overlay
        rgb_overlay = rgb_composite.copy()
        rgb_overlay[ship_mask.astype(bool)] = [1, 0, 0]  # Red for ships
        
        axes[2, 2].imshow(rgb_overlay)
        axes[2, 2].set_title(f"Ships Detected ({len(self.ships)})")
        axes[2, 2].axis('off')
        
        # Mark ship centroids
        for ship in self.ships:
            row, col = ship["centroid_rc"]
            axes[2, 2].plot(col, row, 'yo', markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
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
                             multispectral_stack: np.ndarray,
                             water_mask: np.ndarray,
                             detection_mode: str = "fast",
                             threshold_percentile: float = 99.5) -> None:
        """
        Run the complete RX ship detection pipeline.
        
        Args:
            multispectral_stack: Preprocessed multispectral data stack
            water_mask: Water mask for filtering detections
            detection_mode: RX detection mode ("fast", "adaptive", "pixel_wise")
            threshold_percentile: Detection threshold percentile
        """
        # Step 1: Detect ships using RX
        self.detect_ships(
            multispectral_stack=multispectral_stack,
            water_mask=water_mask,
            detection_mode=detection_mode,
            threshold_percentile=threshold_percentile
        )
        
        # Step 2: Visualize results
        self.visualize_results(
            multispectral_stack=multispectral_stack,
            water_mask=water_mask
        )
        
        # Step 3: Save ship mask
        self.save_ship_mask("ship_mask.png")
        
        # Step 4: Save ship coordinates
        self.save_ship_coordinates("ship_coordinates.json")
        
        # Step 5: Print summary
        self.print_ship_summary()


def main():
    """Main function to run RX ship detection."""
    # Prepare data
    multispectral_stack, water_mask = prepare_detection_data()
    
    # Run detection pipeline
    detector = ShipDetectorRX()
    detector.run_complete_pipeline(
        multispectral_stack=multispectral_stack,
        water_mask=water_mask,
        detection_mode="fast", 
        threshold_percentile=99.5
    )


if __name__ == "__main__":
    main()
