#!/usr/bin/env python3
"""
Test script to demonstrate water mask format.
"""

import numpy as np
from data_processing import create_water_mask_from_scl, create_fallback_water_mask

def demonstrate_water_mask_format():
    print("🌊 Water Mask Format Demonstration")
    print("=" * 50)
    
    # Create a sample SCL band (Scene Classification Layer)
    # SCL values: 0=no_data, 1=saturated, 2=dark_area, 3=cloud_shadow, 
    # 4=vegetation, 5=bare_soil, 6=water, 7=cloud_low, 8=cloud_medium, 
    # 9=cloud_high, 10=thin_cirrus, 11=snow
    height, width = 10, 15
    scl_band = np.array([
        [4, 4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4],  # Row 0: vegetation, water, vegetation
        [4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4],  # Row 1: vegetation, water, vegetation
        [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4],  # Row 2: vegetation, water, vegetation
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4],  # Row 3: water, vegetation
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4],  # Row 4: water, vegetation
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4],  # Row 5: water, vegetation
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # Row 6: all water
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # Row 7: all water
        [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4],  # Row 8: vegetation, water, vegetation
        [4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4],  # Row 9: vegetation, water, vegetation
    ])
    
    print("📊 Sample SCL Band (Scene Classification Layer):")
    print("Values: 4=vegetation, 6=water, 7=cloud_low, 10=thin_cirrus, 11=snow")
    print(f"Shape: {scl_band.shape}")
    print(scl_band)
    print()
    
    # Create water mask from SCL
    water_mask = create_water_mask_from_scl(scl_band)
    
    print("🌊 Water Mask (from SCL):")
    print(f"Data type: {water_mask.dtype}")
    print(f"Shape: {water_mask.shape}")
    print(f"Values: {np.unique(water_mask)} (True=water, False=land)")
    print(f"Water pixels: {water_mask.sum()} / {water_mask.size} ({water_mask.mean()*100:.1f}%)")
    print()
    
    # Show the mask as 1s and 0s for clarity
    print("Water Mask (1=water, 0=land):")
    print(water_mask.astype(int))
    print()
    
    # Create fallback mask
    fallback_mask = create_fallback_water_mask((height, width))
    
    print("🔄 Fallback Water Mask (when no SCL available):")
    print(f"Data type: {fallback_mask.dtype}")
    print(f"Shape: {fallback_mask.shape}")
    print(f"Values: {np.unique(fallback_mask)}")
    print(f"All pixels: {fallback_mask.sum()} / {fallback_mask.size} ({fallback_mask.mean()*100:.1f}%)")
    print()
    
    # Show how it's used in RX detection
    print("🔍 How Water Mask is Used in RX Detection:")
    print("1. Reshape multispectral data: (height, width, bands) → (height*width, bands)")
    print("2. Reshape water mask: (height, width) → (height*width,)")
    print("3. Filter valid pixels: data_2d[mask_1d]")
    print()
    
    # Demonstrate reshaping
    multispectral_data = np.random.rand(height, width, 8)  # 8-band data
    print(f"Multispectral data shape: {multispectral_data.shape}")
    
    data_2d = multispectral_data.reshape(-1, 8)
    mask_1d = water_mask.reshape(-1)
    
    print(f"Reshaped data: {data_2d.shape}")
    print(f"Reshaped mask: {mask_1d.shape}")
    print(f"Valid pixels: {mask_1d.sum()} out of {mask_1d.size}")
    
    valid_pixels = data_2d[mask_1d]
    print(f"Valid pixels shape: {valid_pixels.shape}")
    print(f"Valid pixels are water pixels only!")

if __name__ == "__main__":
    demonstrate_water_mask_format()
