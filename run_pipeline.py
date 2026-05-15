import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import UASDetectionPipeline

print("Starting pipeline...")

p = UASDetectionPipeline(
    uas_threshold=0.30,
    output_dir='logs',
    library_path='library/uas_signatures.jsonl',
    verbose=True,
)

summary = p.run('test_data/uas_test.npz')

print("\n=== PIPELINE DONE ===")
print(f"Segments processed : {summary['detection_stats']['segments_processed']}")
print(f"UAS detections     : {summary['detection_stats']['uas_detections']}")
print(f"Unknown detections : {summary['detection_stats']['unknown_detections']}")
print(f"Library signatures : {summary['run_info']['library_size']}")
print(f"\nAlert log saved to : {summary['run_info']['alert_log']}")