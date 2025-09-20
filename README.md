# 🚢 Ship Detection using RX (Reed-Xiaoli) Anomaly Detector

A complete pipeline for detecting ships in multispectral satellite imagery using the RX (Reed-Xiaoli) anomaly detection algorithm on 8-band Sentinel-2 data.

## 📋 Table of Contents

- [Overview](#overview)
- [RX Algorithm](#rx-algorithm)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Detection Modes](#detection-modes)
- [Output](#output)
- [Testing](#testing)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a ship detection system that:

1. **Loads and preprocesses** 8-band multispectral Sentinel-2 satellite imagery
2. **Creates multispectral stacks** from individual bands
3. **Applies RX anomaly detection** to identify ships as spectral anomalies
4. **Filters detections** based on ship-like characteristics (size, shape, solidity)
5. **Generates comprehensive results** with visualizations and detailed reports

### Key Features

- ✅ **RX Anomaly Detection** optimized for multispectral data
- ✅ **8-band multispectral processing** (B01-B08)
- ✅ **Multiple detection modes** (fast, adaptive, pixel-wise)
- ✅ **Robust filtering** to reduce false positives
- ✅ **Comprehensive visualization** and reporting
- ✅ **Configurable parameters** for different scenarios
- ✅ **Production-ready** with error handling and logging

## 🔬 RX Algorithm

### What is RX Detection?

The Reed-Xiaoli (RX) detector is a statistical anomaly detection algorithm originally developed for hyperspectral imagery. It works by:

1. **Computing background statistics** (mean and covariance) from surrounding pixels
2. **Calculating Mahalanobis distance** between each pixel and background
3. **Identifying anomalies** as pixels that are statistically different from background

### Why RX for Ship Detection?

- **Spectral anomalies**: Ships have different spectral signatures than water
- **Multispectral advantage**: Uses all 8 bands for better discrimination
- **Statistical robustness**: Handles varying background conditions
- **No training required**: Unsupervised detection method

### Mathematical Foundation

For each pixel **x**, the RX score is calculated as:

```
RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Where:
- **x**: Pixel spectral vector (8 bands)
- **μ**: Background mean vector
- **Σ⁻¹**: Inverse of background covariance matrix

Higher RX scores indicate more anomalous (ship-like) pixels.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sentinel-2    │───▶│   Data Pipeline  │───▶│  RX Detection   │
│   8-Band Data   │    │   (load.py)      │    │   Algorithm     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Multispectral   │    │   Ship Results  │
                       │     Stack        │    │  & Visualization│
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Sentinel-2 SAFE format data

### Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python test_rx_detection.py
   ```

## 📖 Usage

### Quick Start

Run the complete RX ship detection pipeline:

```bash
python ship_detector_rx.py
```

This will:
1. Load your Sentinel-2 data from `s2_data.SAFE`
2. Process 8 bands to common resolution
3. Create multispectral stack
4. Detect ships using RX algorithm
5. Generate visualization and summary report

### Programmatic Usage

```python
from ship_detector_rx import ShipDetectorRX

# Initialize detector
detector = ShipDetectorRX("icebreaker/config/settings.yaml")

# Run complete pipeline
detector.run_complete_pipeline(
    detection_mode="fast",
    threshold_percentile=99.5
)

# Or run steps individually
detector.load_and_preprocess_data()
detector.create_multispectral_stack()
rx_scores, detections = detector.detect_ships(detection_mode="fast")
detector.visualize_results("my_results.png")
detector.print_ship_summary()
```

### Custom Parameters

```python
# Customize detection parameters
rx_scores, detections = detector.detect_ships(
    detection_mode="adaptive",      # Detection mode
    threshold_percentile=99.0,      # Detection threshold
    min_area=30,                   # Minimum ship area
    cleanup_size=5                 # Morphological cleanup
)
```

## ⚙️ Configuration

### Data Configuration (`icebreaker/config/settings.yaml`)

```yaml
S2_DATA_ROOT: s2_data.SAFE  # Path to Sentinel-2 SAFE directory
AOI:                        # Area of Interest (GeoJSON format)
  type: Polygon
  coordinates:
    - - [-53.209534, 69.252743]
      - [-53.732758, 69.252743]
      - [-53.732758, 69.040093]
      - [-53.209534, 69.040093]
      - [-53.209534, 69.252743]
```

### RX Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `background_radius` | 15 | Background window radius (pixels) |
| `guard_radius` | 3 | Guard window radius (pixels) |
| `min_valid_pixels` | 50 | Minimum valid pixels for statistics |
| `regularization` | 1e-6 | Covariance matrix regularization |
| `fast_mode` | True | Use fast approximation methods |

### Ship Filtering Criteria

| Criterion | Value | Description |
|-----------|-------|-------------|
| **Size** | 25-2000 pixels | Realistic ship dimensions |
| **Aspect Ratio** | ≥ 1.2 | Ships are elongated |
| **Solidity** | ≥ 0.6 | Ships have solid shapes |

## 🔍 Detection Modes

### 1. Fast Mode (Default)
```python
detector.detect_ships(detection_mode="fast")
```

**Characteristics:**
- Uses global background statistics
- ~10-50x faster than pixel-wise
- Good for initial screening
- Less accurate for varying backgrounds

**Best for:** Large images, initial testing, real-time applications

### 2. Adaptive Mode
```python
detector.detect_ships(detection_mode="adaptive")
```

**Characteristics:**
- Uses local background statistics
- ~5-10x faster than pixel-wise
- Better adaptation to varying backgrounds
- Good balance of speed and accuracy

**Best for:** Production use, varying background conditions

### 3. Pixel-wise Mode
```python
detector.detect_ships(detection_mode="pixel_wise")
```

**Characteristics:**
- Computes statistics for each pixel
- Highest accuracy
- Slowest processing
- Best for research and validation

**Best for:** Small images, maximum accuracy required

## 📊 Output

### Visualization

The pipeline generates a comprehensive 9-panel visualization (3×3 grid):

#### **Row 1: Input Data**
1. **RGB Composite** - True color image (B04, B03, B02)
2. **NIR Band** - Near-infrared band (B08)
3. **Water Mask** - Valid detection regions from SCL

#### **Row 2: Multispectral Bands**
4. **B01 (Coastal)** - Coastal aerosol band
5. **B05 (Red Edge 1)** - Red edge band
6. **B08 (NIR)** - Near-infrared band

#### **Row 3: Results**
7. **RX Scores** - Anomaly detection scores
8. **Raw Detections** - Binary detection mask
9. **Ships Detected** - Detected ships marked on RGB

### Ship Summary Report

```
🚢 RX SHIP DETECTION SUMMARY
============================================================

Ship 1:
  Position: (245.3, 156.7)
  Area: 89 pixels
  Aspect Ratio: 2.34
  Solidity: 0.78
  Length: 18.2 pixels
  Width: 7.8 pixels
```

### Data Files

- **`ship_detection_rx_results.png`** - Comprehensive visualization
- **RX scores array** - Available programmatically as NumPy array
- **Detection arrays** - Binary detection masks

## 🧪 Testing

### Run All Tests
```bash
python test_rx_detection.py
```

### Test Components

1. **Basic RX Detector Test**
   - Tests core RX algorithm with synthetic data
   - Validates anomaly detection functionality

2. **Detection Modes Comparison**
   - Compares fast vs adaptive modes
   - Shows performance vs accuracy trade-offs

3. **Performance Benchmark**
   - Tests processing speed on different image sizes
   - Provides performance metrics

4. **Integration Test**
   - Tests complete pipeline with real data
   - Validates end-to-end functionality

### Synthetic Data Testing

The test script includes synthetic data generation for testing without real satellite data:

```python
# Create synthetic multispectral data
height, width, n_bands = 100, 100, 8
background = np.random.normal(100, 10, (height, width, n_bands))

# Add ship anomalies
ship_positions = [(30, 30), (70, 70)]
for row, col in ship_positions:
    background[row-2:row+3, col-2:col+3, :] = np.random.normal(200, 20, (5, 5, n_bands))
```

## ⚡ Performance

### Typical Performance

| Image Size | Fast Mode | Adaptive Mode | Pixel-wise Mode |
|------------|-----------|---------------|-----------------|
| 100×100    | 0.1s      | 0.5s          | 5s              |
| 500×500    | 0.5s      | 2s            | 60s             |
| 1000×1000  | 2s        | 8s            | 300s            |

### Memory Usage

- **Multispectral stack**: ~32MB for 1000×1000×8 (float32)
- **RX scores**: ~4MB for 1000×1000 (float32)
- **Background statistics**: ~1MB for 8×8 covariance matrix

### Optimization Tips

- **Use fast mode** for initial testing and large images
- **Process in tiles** for very large images
- **Adjust threshold** based on false positive rate
- **Use adaptive mode** for production with varying backgrounds

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Not Found**
   ```bash
   # Check that s2_data.SAFE exists
   ls -la s2_data.SAFE/
   
   # Verify config path
   cat icebreaker/config/settings.yaml
   ```

3. **No Detections**
   - Lower the threshold: `threshold_percentile=95.0`
   - Reduce minimum area: `min_area=10`
   - Check water mask coverage
   - Verify multispectral data range

4. **Too Many False Positives**
   - Increase threshold: `threshold_percentile=99.8`
   - Increase minimum area: `min_area=50`
   - Tighten ship filtering criteria
   - Increase morphological cleanup: `cleanup_size=7`

5. **Memory Issues**
   - Process smaller AOI regions
   - Use fast mode for large images
   - Reduce image resolution

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

detector = ShipDetectorRX()
detector.run_complete_pipeline()
```

### Performance Issues

- **Large images**: Use fast mode or process in tiles
- **Slow processing**: Reduce background window size
- **Memory errors**: Process bands individually

## 📚 References

- **RX Algorithm**: Reed, I.S., Yu, X. (1990) "Adaptive multiple-band CFAR detection of an optical pattern with unknown spectral distribution"
- **Sentinel-2**: ESA's optical Earth observation mission
- **Multispectral Analysis**: Principles of remote sensing and image processing
- **Anomaly Detection**: Statistical methods for outlier detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the DMZ Hackathon and follows the event's licensing terms.

---

**Happy Ship Detecting with RX! 🚢✨**