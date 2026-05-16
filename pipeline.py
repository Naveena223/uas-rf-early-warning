"""
UAS RF Early-Warning Pipeline
Main orchestrator: parse → extract → classify → alert → log.
"""

import time
import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import asdict
from datetime import datetime, timezone

from parsers import auto_detect_and_parse, RFSegment
from feature_extractor import RFFeatureExtractor
from classifier import UASClassifier, LABEL_UAS, LABEL_UNKNOWN
from alert_engine import AlertEngine
from library import SignatureLibrary

logger = logging.getLogger(__name__)


class UASDetectionPipeline:
    """
    End-to-end passive RF monitoring pipeline.

    Processing chain per segment (target ≤2 s):
    1. Parse segment (IQ or spectrogram)
    2. Extract features (≈10–50 ms on laptop CPU)
    3. Classify (≈2–5 ms rule-based, ≈10–30 ms ML)
    4. Library lookup + update
    5. Emit alert if warranted (sync, ≤2 s total)
    6. Write JSON log line
    """

    def __init__(self,
                 uas_threshold: float = 0.55,
                 output_dir: str = 'logs',
                 library_path: str = 'library/uas_signatures.jsonl',
                 unknown_band_alerts: bool = True,
                 sample_rate: float = 1e6,
                 verbose: bool = False):

        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.alert_log_path = os.path.join(output_dir, f'alerts_{ts}.json')
        self.summary_path = os.path.join(output_dir, f'summary_{ts}.json')

        self.extractor = RFFeatureExtractor(sample_rate=sample_rate)
        self.classifier = UASClassifier(uas_threshold=uas_threshold)
        self.alert_engine = AlertEngine(
            log_path=self.alert_log_path,
            unknown_band_alerts=unknown_band_alerts,
        )
        self.library = SignatureLibrary(library_path=library_path)

        self.verbose = verbose
        self._stats = {
            'segments_processed': 0,
            'uas_detections': 0,
            'unknown_detections': 0,
            'non_uas_detections': 0,
            'processing_times_ms': [],
        }

    # ─── Public API ──────────────────────────────────────────────────────────

    def run(self, input_path: str) -> Dict:
        """
        Run the full pipeline on an input file/directory.
        Returns summary dict.
        """
        logger.info(f"Starting UAS RF detection on: {input_path}")
        run_start = time.perf_counter()

        for segment in auto_detect_and_parse(input_path):
            self._process_segment(segment)

        # Flush and merge pending alerts into events
        events = self.alert_engine.flush_events()

        run_duration = time.perf_counter() - run_start
        summary = self._build_summary(input_path, run_duration)
        self._write_summary(summary)

        logger.info(
            f"Pipeline complete in {run_duration:.2f}s | "
            f"Segments: {self._stats['segments_processed']} | "
            f"UAS alerts: {self._stats['uas_detections']} | "
            f"Library: {self.library.count()} signatures"
        )
        return summary

    def set_threshold(self, threshold: float):
        self.classifier.set_threshold(threshold)

    # ─── Private processing ──────────────────────────────────────────────────

    def _process_segment(self, segment: RFSegment):
        t0 = time.perf_counter()

        try:
            # 1. Extract features
            if segment.iq_samples is not None:
                features = self.extractor.extract_from_iq(
                    segment.iq_samples,
                    center_freq=segment.center_freq_hz,
                    metadata=segment.metadata,
                )
            elif segment.spectrogram is not None:
                features = self.extractor.extract_from_spectrogram(
                    segment.spectrogram,
                    time_axis=segment.time_axis,
                    freq_axis=segment.freq_axis,
                )
                # Use stored center freq if spectrum-based peak isn't reliable
                if segment.center_freq_hz > 0 and features.get('peak_freq_hz', 0) == 0:
                    features['peak_freq_hz'] = segment.center_freq_hz
                features['uas_band_match'] = self.extractor._band_match_score(
                    segment.center_freq_hz / 1e6,
                    features.get('bandwidth_3db_hz', 0) / 1e6
                )
            else:
                logger.warning(f"Empty segment: {segment.segment_id}")
                return

            # 2. Classify
            result = self.classifier.classify(features)

            # 3. Library lookup
            match_id = None
            if result.label in (LABEL_UAS, LABEL_UNKNOWN):
                match_id, similarity = self.library.lookup(
                    features, segment.center_freq_hz
                )

            # 4. Alert
            alert = self.alert_engine.process(
                result=result,
                segment_id=segment.segment_id,
                source_file=segment.source_file,
                t_start=segment.timestamp_start,
                t_end=segment.timestamp_end,
                center_freq_hz=segment.center_freq_hz,
                process_start_time=t0,
            )

            # 5. Update library for UAS-like detections
            if alert and result.label == LABEL_UAS:
                sig_id, is_new, _ = self.library.add_or_update(
                    features=features,
                    source_file=segment.source_file,
                    segment_id=segment.segment_id,
                    center_freq_hz=segment.center_freq_hz,
                    bandwidth_hz=features.get('bandwidth_3db_hz', 0),
                    confidence=result.confidence,
                    label=result.label,
                    time_start=segment.timestamp_start,
                    time_end=segment.timestamp_end,
                )
                if alert:
                    alert.library_match_id = sig_id

            # Stats
            self._stats['segments_processed'] += 1
            if result.label == LABEL_UAS:
                self._stats['uas_detections'] += 1
            elif result.label == LABEL_UNKNOWN:
                self._stats['unknown_detections'] += 1
            else:
                self._stats['non_uas_detections'] += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stats['processing_times_ms'].append(elapsed_ms)

            if self.verbose:
                logger.debug(
                    f"  {segment.segment_id}: {result.label} "
                    f"conf={result.confidence:.3f} "
                    f"[{elapsed_ms:.1f}ms]"
                )

        except Exception as e:
            logger.error(f"Error processing {segment.segment_id}: {e}", exc_info=True)

    def _build_summary(self, input_path: str, run_duration: float) -> Dict:
        alert_stats = self.alert_engine.get_summary_stats()
        pt = self._stats['processing_times_ms']
        events = self.alert_engine.get_all_events()

        return {
            'run_info': {
                'input': input_path,
                'run_duration_s': round(run_duration, 3),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'alert_log': self.alert_log_path,
                'library_size': self.library.count(),
            },
            'detection_stats': {
                'segments_processed': self._stats['segments_processed'],
                'uas_detections': self._stats['uas_detections'],
                'unknown_detections': self._stats['unknown_detections'],
                'non_uas_detections': self._stats['non_uas_detections'],
                'detection_rate_pct': round(
                    100 * self._stats['uas_detections'] /
                    max(self._stats['segments_processed'], 1), 2
                ),
            },
            'alert_stats': alert_stats,
            'performance': {
                'mean_segment_time_ms': round(sum(pt)/len(pt), 2) if pt else 0,
                'max_segment_time_ms': round(max(pt), 2) if pt else 0,
                'p95_segment_time_ms': round(sorted(pt)[int(0.95*len(pt))], 2) if len(pt) >= 20 else (round(max(pt), 2) if pt else 0),
                'throughput_segs_per_s': round(len(pt) / run_duration, 2) if run_duration > 0 and pt else 0,
            },
            'events': [
                {
                    'event_id': e.event_id,
                    'label': e.label,
                    'start_epoch': e.start_epoch,
                    'duration_s': e.duration_s,
                    'center_freq_hz': e.center_freq_hz,
                    'peak_confidence': e.peak_confidence,
                    'alert_count': len(e.alerts),
                }
                for e in events
            ],
        }

    def _write_summary(self, summary: Dict):
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary written to {self.summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to RF input file or directory")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--output-dir", default="logs")
    args = parser.parse_args()

    p = UASDetectionPipeline(
        uas_threshold=args.threshold,
        output_dir=args.output_dir,
    )
    summary = p.run(args.input)
    print(f"\nAlerts: {summary['detection_stats']['uas_detections']}")
    print(f"Log: {summary['run_info']['alert_log']}")