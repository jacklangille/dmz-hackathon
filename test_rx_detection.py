"""
Test script for RX ship detection on multispectral 8-band data.

This script tests the RX detector implementation and provides examples
of different detection modes and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from ship_detector_rx import ShipDetectorRX
from rx_detector import RXDetector, create_multispectral_stack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rx_detector_basic():
    """Test basic RX detector functionality with synthetic data."""
    logger.info("Testing basic RX detector...")
    
    # Create synthetic multispectral data
    height, width, n_bands = 100, 100, 8
    
    # Background: water-like spectra (low values, similar across bands)
    background = np.random.normal(100, 10, (height, width, n_bands))
    
    # Add some anomalies (ships): higher values, different spectral signature
    ship_positions = [(30, 30), (70, 70), (20, 80)]
    for row, col in ship_positions:
        # Ships have different spectral signature
        background[row-2:row+3, col-2:col+3, :] = np.random.normal(200, 20, (5, 5, n_bands))
    
    # Create mask (all pixels valid for this test)
    mask = np.ones((height, width), dtype=bool)
    
    # Test RX detector
    rx_detector = RXDetector(
        background_radius=10,
        guard_radius=2,
        min_valid_pixels=20,
        fast_mode=True
    )
    
    # Run detection
    rx_scores, detections = rx_detector.detect_anomalies_fast(
        background, mask=mask, threshold_percentile=95.0
    )
    
    logger.info(f"Detected {detections.sum()} anomalous pixels")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show first band
    axes[0, 0].imshow(background[:, :, 0], cmap='gray')
    axes[0, 0].set_title("Band 1 (Synthetic Data)")
    axes[0, 0].axis('off')
    
    # Show RX scores
    axes[0, 1].imshow(rx_scores, cmap='hot')
    axes[0, 1].set_title("RX Scores")
    axes[0, 1].axis('off')
    
    # Show detections
    axes[1, 0].imshow(detections, cmap='Reds')
    axes[1, 0].set_title("Detections")
    axes[1, 0].axis('off')
    
    # Overlay detections
    overlay = background[:, :, 0].copy()
    overlay[detections] = overlay.max()
    axes[1, 1].imshow(overlay, cmap='gray')
    axes[1, 1].set_title("Detections Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_rx_synthetic.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("Basic RX detector test completed")


def test_different_detection_modes():
    """Test different RX detection modes."""
    logger.info("Testing different RX detection modes...")
    
    # Create synthetic data with known anomalies
    height, width, n_bands = 50, 50, 8
    data = np.random.normal(100, 10, (height, width, n_bands))
    
    # Add anomalies
    data[20:25, 20:25, :] = np.random.normal(200, 20, (5, 5, n_bands))
    data[35:40, 35:40, :] = np.random.normal(180, 15, (5, 5, n_bands))
    
    mask = np.ones((height, width), dtype=bool)
    
    # Test different modes
    modes = ["fast", "adaptive"]
    results = {}
    
    for mode in modes:
        logger.info(f"Testing {mode} mode...")
        
        rx_detector = RXDetector(fast_mode=True)
        
        if mode == "fast":
            scores, detections = rx_detector.detect_anomalies_fast(
                data, mask=mask, threshold_percentile=95.0
            )
        else:  # adaptive
            scores, detections = rx_detector.detect_anomalies_adaptive(
                data, mask=mask
            )
        
        results[mode] = {
            'scores': scores,
            'detections': detections,
            'count': detections.sum()
        }
        
        logger.info(f"{mode} mode detected {detections.sum()} pixels")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    axes[0, 0].imshow(data[:, :, 0], cmap='gray')
    axes[0, 0].set_title("Original Data (Band 1)")
    axes[0, 0].axis('off')
    
    # Fast mode results
    axes[0, 1].imshow(results['fast']['scores'], cmap='hot')
    axes[0, 1].set_title(f"Fast Mode Scores")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['fast']['detections'], cmap='Reds')
    axes[0, 2].set_title(f"Fast Mode Detections ({results['fast']['count']})")
    axes[0, 2].axis('off')
    
    # Adaptive mode results
    axes[1, 0].imshow(results['adaptive']['scores'], cmap='hot')
    axes[1, 0].set_title("Adaptive Mode Scores")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['adaptive']['detections'], cmap='Reds')
    axes[1, 1].set_title(f"Adaptive Mode Detections ({results['adaptive']['count']})")
    axes[1, 1].axis('off')
    
    # Comparison
    comparison = np.zeros((height, width, 3))
    comparison[results['fast']['detections']] = [1, 0, 0]  # Red for fast
    comparison[results['adaptive']['detections']] = [0, 1, 0]  # Green for adaptive
    comparison[np.logical_and(results['fast']['detections'], 
                             results['adaptive']['detections'])] = [1, 1, 0]  # Yellow for both
    
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title("Comparison (Red=Fast, Green=Adaptive)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_rx_modes_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("Detection modes comparison completed")


def test_ship_detector_integration():
    """Test the complete ship detector integration."""
    logger.info("Testing ship detector integration...")
    
    # Check if data exists
    data_path = Path("s2_data.SAFE")
    if not data_path.exists():
        logger.warning("s2_data.SAFE not found. Skipping integration test.")
        logger.info("To run integration test, ensure s2_data.SAFE is in the project directory.")
        return
    
    try:
        # Initialize detector
        detector = ShipDetectorRX()
        
        # Test data loading
        logger.info("Testing data loading...")
        processed_bands = detector.load_and_preprocess_data()
        logger.info(f"Loaded {len(processed_bands)} bands")
        
        # Test multispectral stack creation
        logger.info("Testing multispectral stack creation...")
        multispectral_stack = detector.create_multispectral_stack()
        logger.info(f"Created stack with shape: {multispectral_stack.shape}")
        
        # Test ship detection (fast mode for testing)
        logger.info("Testing ship detection...")
        rx_scores, detections = detector.detect_ships(
            detection_mode="fast",
            threshold_percentile=99.0,
            min_area=20
        )
        
        logger.info(f"Detection completed. Found {len(detector.ships)} ships.")
        
        # Test visualization
        logger.info("Testing visualization...")
        detector.visualize_results("test_rx_integration_results.png")
        
        logger.info("Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def run_performance_benchmark():
    """Run performance benchmark on different image sizes."""
    logger.info("Running performance benchmark...")
    
    import time
    
    # Test different image sizes
    sizes = [(50, 50), (100, 100), (200, 200)]
    n_bands = 8
    
    results = {}
    
    for height, width in sizes:
        logger.info(f"Testing {height}x{width} image...")
        
        # Create synthetic data
        data = np.random.normal(100, 10, (height, width, n_bands))
        mask = np.ones((height, width), dtype=bool)
        
        # Test fast mode
        rx_detector = RXDetector(fast_mode=True)
        
        start_time = time.time()
        scores, detections = rx_detector.detect_anomalies_fast(
            data, mask=mask, threshold_percentile=95.0
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        results[(height, width)] = {
            'time': processing_time,
            'pixels': height * width,
            'detections': detections.sum()
        }
        
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Detections: {detections.sum()}")
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*50)
    for (height, width), result in results.items():
        pixels_per_sec = result['pixels'] / result['time']
        print(f"{height}x{width}: {result['time']:.2f}s, {pixels_per_sec:.0f} pixels/sec")
    
    logger.info("Performance benchmark completed")


def main():
    """Run all tests."""
    logger.info("Starting RX detector tests...")
    
    try:
        # Test 1: Basic functionality
        test_rx_detector_basic()
        
        # Test 2: Different detection modes
        test_different_detection_modes()
        
        # Test 3: Performance benchmark
        run_performance_benchmark()
        
        # Test 4: Integration test (if data available)
        test_ship_detector_integration()
        
        logger.info("All tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
