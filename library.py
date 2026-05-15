"""
UAS Signature Library
Stores detected UAS-like RF signatures for re-occurrence matching.
Each entry is a feature fingerprint with metadata.
"""

import json
import os
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

LIBRARY_PATH_DEFAULT = 'library/uas_signatures.jsonl'


@dataclass
class SignatureEntry:
    """A stored UAS RF signature."""
    signature_id: str
    created_at: str
    source_file: str
    segment_id: str
    center_freq_hz: float
    bandwidth_hz: float
    confidence: float
    label: str
    # Feature fingerprint (subset of most stable features)
    fingerprint: Dict
    # Event metadata
    time_start_s: float
    time_end_s: float
    duration_s: float
    # Occurrence count (incremented on re-match)
    occurrence_count: int = 1
    last_seen_at: str = ''
    summary: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SignatureEntry':
        return cls(**d)


class SignatureLibrary:
    """
    Persistent UAS signature library.
    Stores fingerprints and supports similarity-based matching.

    Fingerprinting uses stable, modulation-independent features:
    - Center frequency band
    - Bandwidth class
    - Burst pattern (duty cycle, IBI)
    - Spectral shape (flatness, roll-off ratio)
    """

    # Features used in fingerprint comparison
    FINGERPRINT_FEATURES = [
        'uas_band_match', 'bandwidth_3db_hz', 'duty_cycle',
        'ibi_regularity', 'spectral_flatness', 'periodicity_score',
        'am_modulation_index', 'cyclic_peak_ratio',
    ]

    # Weights for similarity score
    FEATURE_WEIGHTS = {
        'uas_band_match': 0.25,
        'bandwidth_3db_hz': 0.15,
        'duty_cycle': 0.15,
        'ibi_regularity': 0.15,
        'spectral_flatness': 0.10,
        'periodicity_score': 0.10,
        'am_modulation_index': 0.05,
        'cyclic_peak_ratio': 0.05,
    }

    # Normalisation ranges for feature comparison
    FEATURE_RANGES = {
        'uas_band_match': (0, 1),
        'bandwidth_3db_hz': (0, 40e6),
        'duty_cycle': (0, 1),
        'ibi_regularity': (-1, 1),
        'spectral_flatness': (0, 1),
        'periodicity_score': (0, 1),
        'am_modulation_index': (0, 2),
        'cyclic_peak_ratio': (0, 100),
    }

    MATCH_THRESHOLD = 0.80  # similarity score above which → match

    def __init__(self, library_path: str = LIBRARY_PATH_DEFAULT):
        self.library_path = library_path
        self._entries: Dict[str, SignatureEntry] = {}
        os.makedirs(os.path.dirname(library_path) if os.path.dirname(library_path) else '.', exist_ok=True)
        self._load()

    def add_or_update(self,
                       features: Dict,
                       source_file: str,
                       segment_id: str,
                       center_freq_hz: float,
                       bandwidth_hz: float,
                       confidence: float,
                       label: str,
                       time_start: float,
                       time_end: float) -> Tuple[str, bool, float]:
        """
        Add a new signature or update an existing one if similar.
        Returns (signature_id, is_new, similarity_score).
        """
        fingerprint = self._make_fingerprint(features)
        match_id, sim = self._find_match(fingerprint, center_freq_hz)

        now_str = self._now_str()

        if match_id and sim >= self.MATCH_THRESHOLD:
            # Update existing
            entry = self._entries[match_id]
            entry.occurrence_count += 1
            entry.last_seen_at = now_str
            entry.confidence = max(entry.confidence, confidence)
            logger.info(f"Library match: {match_id} (sim={sim:.3f}, count={entry.occurrence_count})")
            self._save_entry(entry)
            return match_id, False, sim
        else:
            # New signature
            sig_id = 'SIG-' + str(uuid.uuid4())[:8].upper()
            entry = SignatureEntry(
                signature_id=sig_id,
                created_at=now_str,
                source_file=os.path.basename(source_file),
                segment_id=segment_id,
                center_freq_hz=round(center_freq_hz, 1),
                bandwidth_hz=round(bandwidth_hz, 1),
                confidence=round(confidence, 4),
                label=label,
                fingerprint=fingerprint,
                time_start_s=round(time_start, 4),
                time_end_s=round(time_end, 4),
                duration_s=round(time_end - time_start, 4),
                last_seen_at=now_str,
                summary=self._entry_summary(center_freq_hz, bandwidth_hz, features),
            )
            self._entries[sig_id] = entry
            self._save_entry(entry)
            logger.info(f"New library entry: {sig_id} @ {center_freq_hz/1e6:.2f} MHz")
            return sig_id, True, 0.0

    def lookup(self, features: Dict, center_freq_hz: float) -> Tuple[Optional[str], float]:
        """Find best matching library entry. Returns (id, similarity) or (None, 0)."""
        fp = self._make_fingerprint(features)
        return self._find_match(fp, center_freq_hz)

    def get_all(self) -> List[SignatureEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def export_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(
                [e.to_dict() for e in self._entries.values()],
                f, indent=2, default=str
            )
        logger.info(f"Library exported to {path} ({self.count()} entries)")

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _make_fingerprint(self, features: Dict) -> Dict:
        fp = {}
        for feat in self.FINGERPRINT_FEATURES:
            val = features.get(feat, 0.0)
            lo, hi = self.FEATURE_RANGES[feat]
            # Normalised to [0, 1]
            fp[feat] = float(np.clip((val - lo) / (hi - lo + 1e-12), 0, 1))
        return fp

    def _find_match(self, fingerprint: Dict,
                     center_freq_hz: float) -> Tuple[Optional[str], float]:
        best_id = None
        best_sim = 0.0

        for sig_id, entry in self._entries.items():
            # Quick frequency pre-filter (±20% bandwidth tolerance)
            freq_ratio = abs(entry.center_freq_hz - center_freq_hz) / (center_freq_hz + 1e-9)
            if freq_ratio > 0.05:  # >5% frequency difference = different band
                continue

            sim = self._cosine_weighted_similarity(fingerprint, entry.fingerprint)
            if sim > best_sim:
                best_sim = sim
                best_id = sig_id

        return (best_id, best_sim) if best_id else (None, 0.0)

    def _cosine_weighted_similarity(self, fp1: Dict, fp2: Dict) -> float:
        """Weighted similarity between two fingerprints."""
        total_weight = 0.0
        weighted_sim = 0.0

        for feat, weight in self.FEATURE_WEIGHTS.items():
            v1 = fp1.get(feat, 0.0)
            v2 = fp2.get(feat, 0.0)
            # 1 - absolute normalised difference
            sim = 1.0 - abs(v1 - v2)
            weighted_sim += weight * sim
            total_weight += weight

        return float(weighted_sim / (total_weight + 1e-9))

    def _entry_summary(self, freq: float, bw: float, features: Dict) -> str:
        dc = features.get('duty_cycle', 0)
        band = features.get('uas_band_match', 0)
        band_str = "in known UAS band" if band > 0.7 else "near UAS band" if band > 0.3 else "outside UAS bands"
        return (f"UAS-like signal at {freq/1e6:.3f} MHz, BW≈{bw/1e3:.1f} kHz, "
                f"duty={dc*100:.0f}%, {band_str}.")

    def _save_entry(self, entry: SignatureEntry):
        # Rewrite entire library (small size expected)
        with open(self.library_path, 'w') as f:
            for e in self._entries.values():
                f.write(json.dumps(e.to_dict(), default=str) + '\n')

    def _load(self):
        if not os.path.exists(self.library_path):
            return
        with open(self.library_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    e = SignatureEntry.from_dict(d)
                    self._entries[e.signature_id] = e
                except Exception as ex:
                    logger.warning(f"Bad library entry: {ex}")
        logger.info(f"Loaded {len(self._entries)} signatures from library.")

    def _now_str(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
