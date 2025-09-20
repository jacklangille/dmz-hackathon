# 🚢 Ship Detection Pipeline using CFAR on Sentinel-2 Data

A complete pipeline for detecting ships in satellite imagery using the Constant False Alarm Rate (CFAR) algorithm on preprocessed Sentinel-2 data.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Algorithm Details](#algorithm-details)
- [Output](#output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a ship detection system that:

1. **Loads and preprocesses** Sentinel-2 satellite imagery using the existing `load.py` pipeline
2. **Creates water masks** from the Scene Classification Layer (SCL)
3. **Applies CFAR detection** to identify bright targets (ships) against water backgrounds
4. **Filters detections** based on ship-like characteristics (size, shape, solidity)
5. **Generates comprehensive results** with visualizations and detailed reports

### Key Features

- ✅ **Integrated** with existing Sentinel-2 preprocessing pipeline
- ✅ **Optimized** for ship detection in optical imagery
- ✅ **Robust filtering** to reduce false positives
- ✅ **Comprehensive visualization** and reporting
- ✅ **Configurable parameters** for different scenarios
- ✅ **Production-ready** with error handling and logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sentinel-2    │───▶│   Data Pipeline  │───▶│  Ship Detection │
│   SAFE Data     │    │   (load.py)      │    │   (CFAR)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Preprocessed    │    │   Ship Results  │
                       │     Bands        │    │  & Visualization│
                       └──────────────────┘    └─────────────────┘
```

### Components

1. **`masked_cfar.py`** - Core CFAR detection algorithm
2. **`ship_detector.py`** - Complete ship detection pipeline
3. **`icebreaker/utils/load.py`** - Sentinel-2 data preprocessing
4. **`icebreaker/config/settings.yaml`** - Configuration file

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Sentinel-2 SAFE format data

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/dmz-hackathon
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python test_ship_detection.py
   ```

## 📖 Usage

### Quick Start

Run the complete ship detection pipeline:

```bash
python ship_detector.py
```

This will:
1. Load your Sentinel-2 data from `s2_data.SAFE`
2. Process bands to common resolution
3. Detect ships using CFAR
4. Generate visualization and summary report

### Programmatic Usage

```python
from ship_detector import ShipDetector

# Initialize detector
detector = ShipDetector("icebreaker/config/settings.yaml")

# Run complete pipeline
detector.run_complete_pipeline()

# Or run steps individually
detector.load_and_preprocess_data()
detections, scores = detector.detect_ships()
detector.visualize_results("my_results.png")
detector.print_ship_summary()
```

### Custom Parameters

```python
# Customize detection parameters
detections, scores = detector.detect_ships(
    bg_radius=25,      # Larger background window
    guard_radius=7,    # Larger guard window  
    k=3.0,            # Higher threshold (fewer false positives)
    min_area=50,      # Larger minimum ship size
    cleanup_open=5    # Larger morphological cleanup
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

### CFAR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bg_radius` | 20 | Background window radius (pixels) |
| `guard_radius` | 5 | Guard window radius (pixels) |
| `k` | 2.5 | Threshold multiplier |
| `min_valid` | 100 | Minimum valid pixels for statistics |
| `min_area` | 25 | Minimum ship area (pixels) |
| `cleanup_open` | 5 | Morphological opening size |

### Ship Filtering Criteria

| Criterion | Value | Description |
|-----------|-------|-------------|
| **Size** | 25-2000 pixels | Realistic ship dimensions |
| **Aspect Ratio** | ≥ 1.2 | Ships are elongated |
| **Solidity** | ≥ 0.6 | Ships have solid shapes |

## 🔬 Algorithm Details

### CFAR (Constant False Alarm Rate) Detection

The CFAR algorithm works by:

1. **Ring Kernel Construction**: Creates a donut-shaped window around each pixel
   - Outer ring: Background region for statistics
   - Inner ring: Guard region (excluded from background)
   - Center: Test pixel

2. **Local Statistics**: Computes mean and standard deviation from background ring
   - Only uses valid (water) pixels
   - Requires minimum number of valid pixels

3. **Threshold Detection**: Detects pixels that exceed:
   ```
   threshold = mean + k × std
   ```

4. **Post-processing**: 
   - Morphological opening to remove noise
   - Small object removal
   - Ship-like filtering

### Intensity Image Selection

The pipeline uses the **NIR band (B08)** for ship detection because:
- Ships appear bright in NIR due to metal surfaces
- Good contrast against water
- Less affected by atmospheric conditions

Alternative methods available:
- **NDVI-like index**: `(NIR - Red) / (NIR + Red)`
- **RGB composite**: `(Red + Green + Blue) / 3`

### Water Mask Creation

Uses the Scene Classification Layer (SCL) to identify water pixels:
- **Class 6**: Water
- **Class 7**: Water vapor
- **Dilation**: Includes near-shore areas to catch ships partially on land

## 📊 Output

### Visualization

The pipeline generates a 6-panel visualization:

1. **RGB Composite** - True color image
2. **NIR Band** - Intensity image used for detection
3. **Water Mask** - Valid detection regions
4. **CFAR Scores** - Detection confidence scores
5. **Raw Detections** - Binary detection mask
6. **Ships Overlay** - Detected ships marked on RGB image

### Ship Summary Report

```
🚢 SHIP DETECTION SUMMARY
============================================================
Ship 1:
  Position: (245.3, 156.7)
  Area: 89 pixels
  Mean Intensity: 1247.32
  Aspect Ratio: 2.34
  Solidity: 0.78
  Length: 18.2 pixels
```

### Data Files

- **`ship_detection_results.png`** - Visualization
- **Detection arrays** - Available programmatically as NumPy arrays

## 🛠️ Customization

### Different Target Types

For detecting other objects, modify the filtering criteria:

```python
def _is_ship_like(self, region):
    # For aircraft detection
    return (10 <= region.area <= 100 and
            region.solidity >= 0.8)
    
    # For building detection  
    return (50 <= region.area <= 5000 and
            region.aspect_ratio <= 3.0)
```

### Different Intensity Images

```python
def create_ship_intensity_image(self, bands):
    # Method 1: NIR (default)
    intensity = bands["B08"].astype(np.float32)
    
    # Method 2: NDVI-like
    nir = bands["B08"].astype(np.float32)
    red = bands["B04"].astype(np.float32)
    intensity = (nir - red) / (nir + red + 1e-6)
    
    # Method 3: RGB composite
    red = bands["B04"].astype(np.float32)
    green = bands["B03"].astype(np.float32)
    blue = bands["B02"].astype(np.float32)
    intensity = (red + green + blue) / 3.0
    
    return intensity
```

### Different CFAR Parameters

```python
# For high-resolution imagery
detections, scores = detector.detect_ships(
    bg_radius=30,      # Larger background
    guard_radius=8,    # Larger guard
    k=2.0,            # Lower threshold
    min_area=15       # Smaller minimum area
)

# For noisy imagery
detections, scores = detector.detect_ships(
    bg_radius=15,      # Smaller background
    guard_radius=3,    # Smaller guard
    k=4.0,            # Higher threshold
    cleanup_open=7    # Larger cleanup
)
```

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
   - Lower the threshold: `k=1.5`
   - Reduce minimum area: `min_area=10`
   - Check water mask coverage
   - Verify intensity image range

4. **Too Many False Positives**
   - Increase threshold: `k=3.5`
   - Increase minimum area: `min_area=50`
   - Tighten ship filtering criteria
   - Increase morphological cleanup: `cleanup_open=7`

### Performance Optimization

- **Large images**: Process in tiles
- **Memory issues**: Reduce image size or process bands separately
- **Speed**: Use smaller background windows for faster processing

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

detector = ShipDetector()
detector.run_complete_pipeline()
```

## 📚 References

- **CFAR Algorithm**: Constant False Alarm Rate detection for radar/sonar
- **Sentinel-2**: ESA's optical Earth observation mission
- **Scene Classification Layer**: Automated land/water classification
- **Morphological Operations**: Image processing for noise removal

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the DMZ Hackathon and follows the event's licensing terms.

---

**Happy Ship Detecting! 🚢✨**