from data_processing import prepare_detection_data
from ship_detector_rx import ShipDetectorRX


multispectral_stack, water_mask = prepare_detection_data()

print("✅ Preprocessing done")

detector = ShipDetectorRX()
print(
    detector.detect_ships(
        multispectral_stack=multispectral_stack,
        water_mask=water_mask,
        detection_mode="fast", 
        threshold_percentile=99.5
    )
)

print("✅ Detection done")