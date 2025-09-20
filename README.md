# 🚢 RX Ship Detection Pipeline using Multispectral Satellite Data

A complete pipeline for detecting ships in multispectral satellite imagery using the RX (Reed-Xiaoli) anomaly detector on preprocessed Sentinel-2 data.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [RX Detection Algorithm](#rx-detection-algorithm)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a ship detection system that:

1. **Loads and preprocesses** Sentinel-2 multispectral imagery using the `load.py` pipeline
2. **Applies the RX anomaly detector** to identify statistical outliers (potential ships) in the multispectral data
3. **Filters detections** based on ship-like characteristics (size, shape, solidity)
4. **Generates comprehensive results** with visualizations and detailed reports

### Key Features

- **8-band multispectral processing** (B01-B08 from Sentinel-2)
- **Three detection modes**: Fast, Adaptive, and Pixel-wise
- **Automatic shape consistency** handling for different band resolutions
- **Comprehensive visualization** with 9-panel results
- **Configurable parameters** for different scenarios
- **Robust post-processing** to reduce false positives

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sentinel-2    │───▶│  Data Pipeline   │───▶│  RX Detector    │
│   .SAFE Data    │    │   (load.py)      │    │  (rx_detector)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Multispectral    │    │ Ship Detection  │
                       │ Stack (8 bands)  │    │ & Analysis      │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Visualization   │
                                               │ & Results       │
                                               └─────────────────┘
```

## 🔬 RX Detection Algorithm

### Mathematical Foundation

The **Reed-Xiaoli (RX) Detector** is a statistical anomaly detection algorithm that identifies spectral outliers in multispectral imagery. It computes the **Mahalanobis distance** between each pixel's spectral signature and the local background statistics:

```
RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Where:
- `x` = pixel spectral vector (8-dimensional)
- `μ` = background mean vector
- `Σ⁻¹` = inverse background covariance matrix

### Algorithm Inputs

#### Primary Input:
```python
multispectral_data: np.ndarray
# Shape: (height, width, n_bands)
# Type: float32/float64
# Range: 0.0 - 14034.0 (reflectance values)
```

#### Secondary Inputs:
```python
mask: Optional[np.ndarray]
# Shape: (height, width)
# Type: bool
# Purpose: Valid pixels for detection (water mask)

# Detection Parameters:
background_radius: int = 15      # Background sampling radius
guard_radius: int = 3            # Exclusion zone around test pixel
min_valid_pixels: int = 50       # Minimum pixels for background stats
regularization: float = 1e-6     # Covariance matrix regularization
threshold_percentile: float = 99.5  # Detection threshold percentile
```

### Algorithm Outputs

#### Primary Outputs:
```python
rx_scores: np.ndarray
# Shape: (height, width)
# Type: float32
# Values: Mahalanobis distance scores (higher = more anomalous)

detections: np.ndarray
# Shape: (height, width)
# Type: bool
# Values: True = detected anomaly, False = background
```

#### Secondary Outputs:
```python
ships: List[Dict]
# List of detected ship objects with properties:
# - position: (x, y) coordinates
# - area: pixel count
# - aspect_ratio: length/width
# - solidity: shape compactness
# - length, width: bounding box dimensions
```

### Three Detection Modes

#### A. Fast Mode (Default)
```python
detect_anomalies_fast()
```
- **Global background statistics** computed once
- **Vectorized operations** for speed
- **Suitable for**: Large images, initial screening
- **Speed**: ~100x faster than pixel-wise
- **Accuracy**: Good for homogeneous backgrounds

#### B. Adaptive Mode
```python
detect_anomalies_adaptive()
```
- **Local background statistics** in sliding windows
- **Better adaptation** to varying backgrounds
- **Suitable for**: Heterogeneous scenes
- **Speed**: ~10x faster than pixel-wise
- **Accuracy**: Better than fast mode

#### C. Pixel-wise Mode
```python
detect_anomalies()
```
- **Individual background statistics** for each pixel
- **Ring-based sampling** around each pixel
- **Suitable for**: Maximum accuracy
- **Speed**: Slowest
- **Accuracy**: Highest

### Background Sampling Strategy

#### Ring-based Sampling:
```python
def _create_ring_mask():
    distances = sqrt((rows - center_row)² + (cols - center_col)²)
    ring_mask = (distances <= background_radius) & (distances > guard_radius)
```

- **Outer radius**: `background_radius` (default: 15 pixels)
- **Inner radius**: `guard_radius` (default: 3 pixels)
- **Purpose**: Exclude target pixel and immediate neighbors

### Post-processing Pipeline

```python
def post_process_detections():
    # 1. Morphological opening (noise removal)
    detections = opening(detections, footprint_rectangle((3, 3)))
    
    # 2. Remove small objects
    detections = remove_small_objects(detections, min_size=25)
    
    # 3. Ship analysis
    ships = analyze_ship_properties(detections)
```

## 📊 Data Processing Pipeline

### Data Flow

1. **Raw Sentinel-2 bands** → **Resampled & clipped** → **Consistent shapes**
2. **Individual bands** → **Stacked** → **3D multispectral array**
3. **Multispectral array** → **RX detector** → **Anomaly scores**

### Band Processing

The pipeline handles 8 Sentinel-2 bands with different native resolutions:

| Band | Wavelength | Native Resolution | Purpose |
|------|------------|-------------------|---------|
| B01  | 443nm      | 60m               | Coastal aerosol |
| B02  | 490nm      | 10m               | Blue |
| B03  | 560nm      | 10m               | Green |
| B04  | 665nm      | 10m               | Red |
| B05  | 705nm      | 20m               | Red Edge 1 |
| B06  | 740nm      | 20m               | Red Edge 2 |
| B07  | 783nm      | 20m               | Red Edge 3 |
| B08  | 842nm      | 10m               | NIR |

### Shape Consistency

The pipeline automatically ensures all bands have consistent shapes:

```python
def _ensure_consistent_shapes(self, processed_bands):
    # Find the most common shape (target shape)
    target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
    
    # Resize bands that don't match using scipy.ndimage.zoom
    for band_name, band_array in processed_bands.items():
        if band_array.shape != target_shape:
            zoom_factors = (target_shape[0] / band_array.shape[0], 
                          target_shape[1] / band_array.shape[1])
            resized_band = zoom(band_array, zoom_factors, order=1)
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd dmz-hackathon
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
matplotlib>=3.5.0
rasterio>=1.2.0
geopandas>=0.10.0
PyYAML>=6.0.0
shapely>=2.0.0
```

## 📖 Usage

### Basic Usage

```bash
# Run the complete pipeline
python ship_detector_rx.py
```

### Programmatic Usage

```python
from ship_detector_rx import ShipDetectorRX

# Initialize detector
detector = ShipDetectorRX("icebreaker/config/settings.yaml")

# Run complete pipeline
detector.run_complete_pipeline()

# Access results
ships = detector.ships
rx_scores = detector.rx_scores
detections = detector.detections
```

### Custom Detection Parameters

```python
# Initialize with custom parameters
detector = ShipDetectorRX()

# Run detection with custom settings
detector.detect_ships(
    detection_mode="adaptive",      # "fast", "adaptive", or "pixel_wise"
    threshold_percentile=99.0,      # Detection threshold
    min_area=30,                    # Minimum ship area
    cleanup_size=3                  # Morphological cleanup size
)
```

## ⚙️ Configuration

### Configuration File (`icebreaker/config/settings.yaml`)

```yaml
S2_DATA_ROOT: s2_data.SAFE
AOI:
  type: Polygon
  coordinates:
    - - [-53.209534, 69.252743]
      - [-53.732758, 69.252743]
      - [-53.732758, 69.040093]
      - [-53.209534, 69.040093]
      - [-53.209534, 69.252743]
```

### RX Detector Parameters

```python
# Default parameters
RXDetector(
    background_radius=15,      # Background sampling radius
    guard_radius=3,            # Exclusion zone around test pixel
    min_valid_pixels=50,       # Minimum pixels for background stats
    regularization=1e-6,       # Covariance matrix regularization
    fast_mode=True             # Use fast approximation methods
)
```

## 📈 Output

### Visualization

The pipeline generates a comprehensive 9-panel visualization (`ship_detection_rx_results.png`):

1. **RGB Composite** - Natural color visualization
2. **NIR Composite** - Near-infrared visualization
3. **Multispectral Stack** - All 8 bands combined
4. **RX Scores** - Anomaly detection scores
5. **Binary Detections** - Raw detection mask
6. **Post-processed Detections** - Cleaned detections
7. **Ship Analysis** - Detected ships with bounding boxes
8. **Ship Properties** - Size and shape analysis
9. **Detection Summary** - Statistical overview

### Console Output

```
🚢 RX SHIP DETECTION SUMMARY
============================================================

Ship 1:
  Position: (43.9, 384.5)
  Area: 45 pixels
  Aspect Ratio: 2.09
  Solidity: 0.94
  Length: 11.1 pixels
  Width: 5.3 pixels

Ship 2:
  Position: (65.4, 860.1)
  Area: 27 pixels
  Aspect Ratio: 1.36
  Solidity: 0.96
  Length: 6.8 pixels
  Width: 5.0 pixels

... (additional ships)
```

### Performance Metrics

```
INFO:__main__:Multispectral stack shape: (1244, 1107, 8)
INFO:__main__:Data range: 0.00 - 14034.00
INFO:rx_detector:Fast RX detection completed. Threshold: 338.28
INFO:rx_detector:Detected 6886 anomalous pixels
INFO:__main__:Raw detections: 2366 pixels
INFO:__main__:Ships detected: 19
```

## ⚡ Performance Tuning

### Speed Optimization Parameters

| Parameter | Current | Range | Impact on Speed | Impact on Accuracy |
|-----------|---------|-------|-----------------|-------------------|
| `detection_mode` | "fast" | fast/adaptive/pixel_wise | 100x/10x/1x | Good/Fair/Best |
| `background_radius` | 15 | 5-30 | Smaller = Faster | Smaller = Less robust |
| `min_valid_pixels` | 50 | 20-100 | Smaller = Faster | Smaller = Less stable |
| `threshold_percentile` | 99.5 | 95-99.9 | N/A | Higher = Fewer FPs |

### Accuracy Optimization Parameters

| Parameter | Current | Range | Impact on Accuracy | Impact on Speed |
|-----------|---------|-------|-------------------|-----------------|
| `regularization` | 1e-6 | 1e-8 to 1e-4 | Higher = More stable | N/A |
| `guard_radius` | 3 | 1-5 | Larger = Better separation | N/A |
| `min_area` | 25 | 10-50 | Larger = Fewer FPs | N/A |
| `cleanup_size` | 3 | 1-5 | Larger = Less noise | N/A |

### Recommended Configurations

#### For Maximum Speed:
```python
detector = RXDetector(
    background_radius=10,      # Smaller radius
    min_valid_pixels=30,       # Fewer pixels needed
    fast_mode=True
)
```

#### For Maximum Accuracy:
```python
detector = RXDetector(
    background_radius=20,      # Larger radius
    guard_radius=5,            # Larger guard zone
    min_valid_pixels=100,      # More pixels for stats
    regularization=1e-8        # Less regularization
)
```

#### For Balanced Performance:
```python
detector = RXDetector(
    background_radius=15,      # Default
    guard_radius=3,            # Default
    min_valid_pixels=50,       # Default
    regularization=1e-6        # Default
)
```

### Scene-Specific Tuning

#### Open Ocean (Homogeneous Background):
```python
detection_mode = "fast"
background_radius = 20
threshold_percentile = 99.8
```

#### Coastal Areas (Heterogeneous Background):
```python
detection_mode = "adaptive"
window_size = 41
threshold_percentile = 99.5
```

#### High-Resolution Imagery:
```python
background_radius = 25
guard_radius = 5
min_area = 50  # Larger minimum ship size
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Shape Mismatch Error
```
ValueError: Failed to standardize band shapes
```
**Solution**: The pipeline automatically handles this with the `_ensure_consistent_shapes()` method.

#### 2. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solutions**:
- Use `detection_mode="fast"` for large images
- Reduce `background_radius`
- Process image in tiles

#### 3. No Ships Detected
**Solutions**:
- Lower `threshold_percentile` (e.g., 99.0)
- Reduce `min_area`
- Check if water mask is correct
- Verify multispectral data quality

#### 4. Too Many False Positives
**Solutions**:
- Increase `threshold_percentile` (e.g., 99.8)
- Increase `min_area`
- Increase `cleanup_size`
- Use `detection_mode="adaptive"` for heterogeneous scenes

### Performance Issues

#### Slow Processing
- Use `detection_mode="fast"` for initial testing
- Reduce `background_radius`
- Process smaller AOI regions

#### Low Detection Accuracy
- Use `detection_mode="adaptive"` or `"pixel_wise"`
- Increase `background_radius`
- Adjust `threshold_percentile`

### Data Quality Issues

#### Poor Spectral Contrast
- Check band alignment and registration
- Verify radiometric calibration
- Ensure proper atmospheric correction

#### Inconsistent Band Shapes
- The pipeline automatically handles this
- Check if all bands are properly resampled

## 📚 References

- Reed, I. S., & Yu, X. (1990). Adaptive multiple-band CFAR detection of an optical pattern with unknown spectral distribution. IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(10), 1760-1770.
- Manolakis, D., & Shaw, G. (2002). Detection algorithms for hyperspectral imaging applications. IEEE Signal Processing Magazine, 19(1), 29-43.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on the GitHub repository or contact the development team.