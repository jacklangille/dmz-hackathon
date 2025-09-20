#!/usr/bin/env python3
"""
Test script for ship detection pipeline.
"""

import numpy as np
from ship_detector import ShipDetector


def test_ship_detector():
    """Test the ship detector with synthetic data."""
    print("🧪 Testing Ship Detector...")
    
    # Create synthetic test data
    print("Creating synthetic test data...")
    
    # Create a simple test image (100x100)
    img = np.ones((100, 100), dtype=np.float32) * 10.0  # Background
    
    # Add a "ship" (bright elongated object)
    img[40:50, 30:60] = 50.0  # Ship-like shape
    
    # Add some noise
    img += np.random.normal(0, 1, img.shape)
    
    # Create water mask (most of the image is water)
    water_mask = np.ones_like(img, dtype=bool)
    water_mask[0:20, :] = False  # Some land at top
    water_mask[:, 0:10] = False  # Some land at left
    
    print(f"Test image shape: {img.shape}")
    print(f"Ship intensity: {img[45, 45]}")
    print(f"Background intensity: {img[20, 20]}")
    print(f"Water pixels: {water_mask.sum()}")
    
    # Test CFAR detection
    from masked_cfar import masked_cfar
    
    detections, scores = masked_cfar(
        img=img,
        mask=water_mask,
        bg_radius=15,
        guard_radius=3,
        k=2.0,
        min_valid=50,
        min_area=20,
        cleanup_open=3
    )
    
    print(f"Detections: {detections.sum()}")
    print(f"Ship detected: {detections[40:50, 30:60].any()}")
    
    # Test ship-like filtering
    from skimage.measure import label, regionprops
    
    labeled = label(detections)
    regions = regionprops(labeled, intensity_image=img)
    
    ships = []
    for region in regions:
        if (25 <= region.area <= 2000 and
            region.major_axis_length / region.minor_axis_length >= 1.2 and
            region.solidity >= 0.6):
            ships.append(region)
    
    print(f"Ships found: {len(ships)}")
    
    if ships:
        ship = ships[0]
        print(f"Ship area: {ship.area}")
        print(f"Ship aspect ratio: {ship.major_axis_length / ship.minor_axis_length:.2f}")
        print(f"Ship solidity: {ship.solidity:.3f}")
    
    print("✅ Test completed successfully!")
    return len(ships) > 0


if __name__ == "__main__":
    test_ship_detector()
