#!/usr/bin/env python3
"""
Test script for Maxar dataset support in RX ship detection.
"""

from ship_detector_rx import main

if __name__ == "__main__":
    print("Testing Maxar dataset support...")
    print("Note: This will fail if Maxar data is not available at the configured path.")
    print("Update icebreaker/config/maxar_settings.yaml with your actual data path.")
    print()
    
    try:
        # Test with Maxar dataset
        main(dataset_type="maxar")
    except Exception as e:
        print(f"Error testing Maxar dataset: {e}")
        print("\nThis is expected if Maxar data is not available.")
        print("Please update the configuration file with your actual data path.")
