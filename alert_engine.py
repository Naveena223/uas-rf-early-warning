"""
Alert Engine
Generates, deduplicates, and logs UAS detection alerts.
Enforces ≤2 second alert latency requirement.
Merges consecutive detections into events.
"""

import json
import time
import uuid
import logging
import os
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Deque
from datetime import datetime, timezone

from classifier import ClassificationResult, LABEL_UAS, LABEL_UNKNOWN

logger = logging.getLogger(__name__)


# ─── Alert data model ─────────────────────────────────────────────────────────

@dataclass
class RFAlert:
    alert_id: str
    alert_type: str                  # 'UAS-LIKE' | 'UNKNOWN'
    severity: str                    # 'HIGH' | 'MEDIUM' | 'LOW'
    timestamp_utc: str               # ISO-8601
    epoch_s: float                   # Unix epoch
    signal_time_start: float         # seconds relative to file start
    signal_time_end: float
    center_freq_hz: float
    bandwidth_hz: float
    confidence: float                # 0–1
    uas_score: float
    non_uas_score: float
    unknown_score: float
    detection_latency_ms: float      # from segment onset to alert emit
    source_file: str
    segment_ids: List[str]
    features: Dict
    library_match_id: Optional[str]  # if matched to library entry
    summary: str


@dataclass
class AlertEvent:
    """Merged event spanning multiple consecutive UAS detections."""
    event_id: str
    alerts: List[RFAlert]
    start_epoch: float
    end_epoch: float
    peak_confidence: float
    mean_confidence: float
    center_freq_hz: float
    bandwidth_hz: float
    duration_s: float
    label: str


# ─── Alert Engine ─────────────────────────────────────────────────────────────

class AlertEngine:
    """
    Processes ClassificationResults into actionable alerts.

    Latency guarantee:
    - Alerts are emitted synchronously within the processing pipeline.
    - The alert_emit_time - segment_start must be ≤ 2000 ms.
    - If processing is too slow, the engine logs a warning.

    Deduplication / event merging:
    - Consecutive UAS-LIKE segments within merge_gap_s are merged into one event.
    """

    LATENCY_BUDGET_MS = 2000.0

    def __init__(self,
                 log_path: str = 'logs/alerts.json',
                 unknown_band_alerts: bool = True,
                 min_confidence: float = 0.0,
                 merge_gap_s: float = 1.0):
        self.log_path = log_path
        self.unknown_band_alerts = unknown_band_alerts
        self.min_confidence = min_confidence
        self.merge_gap_s = merge_gap_s

        self._alerts: List[RFAlert] = []
        self._events: List[AlertEvent] = []
        self._pending: Deque[RFAlert] = deque()
        self._latencies: List[float] = []

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        self._log_handle = open(log_path, 'a')

        logger.info(f"AlertEngine initialised. Log: {log_path}")

    def process(self,
                result: ClassificationResult,
                segment_id: str,
                source_file: str,
                t_start: float,
                t_end: float,
                center_freq_hz: float,
                process_start_time: float) -> Optional[RFAlert]:
        """
        Process a classification result and emit an alert if warranted.
        process_start_time: time.perf_counter() at segment onset.
        """
        should_alert = (
            result.label == LABEL_UAS or
            (self.unknown_band_alerts and result.label == LABEL_UNKNOWN)
        ) and result.confidence >= self.min_confidence

        if not should_alert:
            return None

        emit_time = time.perf_counter()
        latency_ms = (emit_time - process_start_time) * 1000.0
        self._latencies.append(latency_ms)

        if latency_ms > self.LATENCY_BUDGET_MS:
            logger.warning(
                f"⚠️  Alert latency {latency_ms:.1f} ms exceeds {self.LATENCY_BUDGET_MS} ms budget "
                f"for segment {segment_id}"
            )

        bw = result.features_summary.get('bandwidth_3db_hz', 0.0)
        severity = self._severity(result.confidence)
        now_utc = datetime.now(timezone.utc)

        alert = RFAlert(
            alert_id=str(uuid.uuid4())[:8].upper(),
            alert_type=result.label,
            severity=severity,
            timestamp_utc=now_utc.isoformat(),
            epoch_s=now_utc.timestamp(),
            signal_time_start=t_start,
            signal_time_end=t_end,
            center_freq_hz=center_freq_hz,
            bandwidth_hz=bw,
            confidence=round(result.confidence, 4),
            uas_score=round(result.uas_score, 4),
            non_uas_score=round(result.non_uas_score, 4),
            unknown_score=round(result.unknown_score, 4),
            detection_latency_ms=round(latency_ms, 2),
            source_file=os.path.basename(source_file),
            segment_ids=[segment_id],
            features=result.features_summary,
            library_match_id=None,
            summary=self._summary(result, center_freq_hz, bw, t_start),
        )

        self._alerts.append(alert)
        self._pending.append(alert)
        self._write_alert(alert)

        logger.info(
            f"🚨 ALERT [{alert.severity}] {alert.alert_type} | "
            f"freq={center_freq_hz/1e6:.2f} MHz | "
            f"conf={result.confidence:.2f} | "
            f"lat={latency_ms:.0f}ms | id={alert.alert_id}"
        )
        return alert

    def flush_events(self) -> List[AlertEvent]:
        """Merge pending alerts into events based on temporal proximity."""
        if not self._pending:
            return []

        pending = sorted(self._pending, key=lambda a: a.signal_time_start)
        self._pending.clear()

        events = []
        current_group: List[RFAlert] = [pending[0]]

        for alert in pending[1:]:
            last = current_group[-1]
            if (alert.signal_time_start - last.signal_time_end) <= self.merge_gap_s:
                current_group.append(alert)
            else:
                events.append(self._make_event(current_group))
                current_group = [alert]
        events.append(self._make_event(current_group))

        self._events.extend(events)
        return events

    def get_summary_stats(self) -> Dict:
        """Return alert statistics for the summary report."""
        n = len(self._alerts)
        uas_alerts = [a for a in self._alerts if a.alert_type == LABEL_UAS]
        unk_alerts = [a for a in self._alerts if a.alert_type == LABEL_UNKNOWN]
        lats = self._latencies

        return {
            'total_alerts': n,
            'uas_like_alerts': len(uas_alerts),
            'unknown_alerts': len(unk_alerts),
            'total_events': len(self._events),
            'mean_confidence': round(float(
                sum(a.confidence for a in self._alerts) / n
            ), 4) if n > 0 else 0.0,
            'latency_mean_ms': round(float(sum(lats) / len(lats)), 2) if lats else 0.0,
            'latency_max_ms': round(float(max(lats)), 2) if lats else 0.0,
            'latency_p95_ms': round(float(sorted(lats)[int(0.95 * len(lats))]), 2) if len(lats) >= 20 else (round(max(lats), 2) if lats else 0.0),
            'latency_budget_met_pct': round(
                100 * sum(1 for l in lats if l <= self.LATENCY_BUDGET_MS) / len(lats), 1
            ) if lats else 100.0,
            'alerts_above_0_8_confidence': sum(1 for a in self._alerts if a.confidence >= 0.8),
        }

    def get_all_alerts(self) -> List[RFAlert]:
        return list(self._alerts)

    def get_all_events(self) -> List[AlertEvent]:
        return list(self._events)

    def close(self):
        self._log_handle.flush()
        self._log_handle.close()

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _write_alert(self, alert: RFAlert):
        line = json.dumps(asdict(alert), default=str)
        self._log_handle.write(line + '\n')
        self._log_handle.flush()

    def _severity(self, confidence: float) -> str:
        if confidence >= 0.80:
            return 'HIGH'
        elif confidence >= 0.60:
            return 'MEDIUM'
        return 'LOW'

    def _summary(self, result: ClassificationResult,
                  freq: float, bw: float, t: float) -> str:
        f_mhz = freq / 1e6
        bw_khz = bw / 1e3
        dc = result.features_summary.get('duty_cycle', 0)
        ir = result.features_summary.get('ibi_regularity', 0)
        parts = [f"Detected {result.label} signal at {f_mhz:.3f} MHz"]
        if bw_khz > 0:
            parts.append(f"BW≈{bw_khz:.1f} kHz")
        if dc > 0:
            parts.append(f"duty={dc*100:.0f}%")
        if ir > 0.5:
            parts.append(f"periodic-bursts")
        parts.append(f"conf={result.confidence:.0%}")
        return ", ".join(parts) + "."

    def _make_event(self, alerts: List[RFAlert]) -> AlertEvent:
        freqs = [a.center_freq_hz for a in alerts]
        confs = [a.confidence for a in alerts]
        bws = [a.bandwidth_hz for a in alerts]
        return AlertEvent(
            event_id='EVT-' + str(uuid.uuid4())[:6].upper(),
            alerts=alerts,
            start_epoch=alerts[0].epoch_s,
            end_epoch=alerts[-1].epoch_s,
            peak_confidence=round(max(confs), 4),
            mean_confidence=round(sum(confs)/len(confs), 4),
            center_freq_hz=round(sum(freqs)/len(freqs), 1),
            bandwidth_hz=round(max(bws), 1),
            duration_s=round(alerts[-1].signal_time_end - alerts[0].signal_time_start, 4),
            label=alerts[0].alert_type,
        )
