"""
UAS RF Classifier
Ensemble of Random Forest + Gradient Boosting with calibrated confidence scores.
Supports threshold tuning for FAR control and binary/ternary output.
"""

import numpy as np
import pickle
import logging
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ─── Output labels ────────────────────────────────────────────────────────────
LABEL_UAS     = "UAS-LIKE"
LABEL_NON_UAS = "NON-UAS"
LABEL_UNKNOWN = "UNKNOWN"

@dataclass
class ClassificationResult:
    label: str               # UAS-LIKE / NON-UAS / UNKNOWN
    confidence: float        # 0.0 – 1.0 probability of UAS-LIKE
    uas_score: float         # raw ensemble score
    non_uas_score: float
    unknown_score: float
    threshold_used: float
    features_summary: Dict   # key features driving the decision
    model_version: str = "v1.0-ensemble"

    def to_dict(self) -> Dict:
        return asdict(self)


class UASClassifier:
    """
    Lightweight ensemble classifier for UAS RF detection.

    Architecture:
    - Random Forest (primary): robust to noisy features, interpretable
    - Gradient Boosted Trees (secondary): captures non-linear interactions
    - Rule-based band/burst oracle: physics-informed override layer
    - Calibrated Platt scaling for confidence scores

    Threshold strategy:
    - uas_threshold: probability above which signal → UAS-LIKE (tune for FAR control)
    - unknown_band: [low, high] around threshold for UNKNOWN output
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'uas_classifier.pkl')

    def __init__(self, uas_threshold: float = 0.55, unknown_band: float = 0.10):
        self.uas_threshold = uas_threshold
        self.unknown_band = unknown_band  # ± around threshold → UNKNOWN
        self._rf = None
        self._gb = None
        self._scaler = None
        self._feature_names: List[str] = []
        self._is_trained = False
        self._load_or_init()

    # ─── Public API ──────────────────────────────────────────────────────────

    def classify(self, features: Dict) -> ClassificationResult:
        """Classify a feature dict, returning a ClassificationResult."""
        vec = self._features_to_vector(features)

        if self._is_trained:
            uas_prob = self._ensemble_predict_proba(vec)
        else:
            uas_prob = self._rule_based_score(features)

        # Apply unknown band around threshold
        low = max(0.0, self.uas_threshold - self.unknown_band)
        high = min(1.0, self.uas_threshold + self.unknown_band)

        if uas_prob >= high:
            label = LABEL_UAS
        elif uas_prob < low:
            label = LABEL_NON_UAS
        else:
            label = LABEL_UNKNOWN

        non_uas_prob = 1.0 - uas_prob
        # For ternary, redistribute unknown confidence
        if label == LABEL_UNKNOWN:
            mid = (uas_prob + (1 - uas_prob)) / 2
            uas_s = uas_prob
            non_uas_s = non_uas_prob
            unk_s = 1.0 - abs(uas_prob - 0.5) * 2
        else:
            unk_s = 0.0
            uas_s = uas_prob
            non_uas_s = non_uas_prob

        return ClassificationResult(
            label=label,
            confidence=float(uas_prob),
            uas_score=float(uas_s),
            non_uas_score=float(non_uas_s),
            unknown_score=float(unk_s),
            threshold_used=self.uas_threshold,
            features_summary=self._top_features(features, vec),
        )

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: Optional[List[str]] = None):
        """Train the ensemble on labelled data (y: 1=UAS, 0=Non-UAS)."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import Pipeline

            self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._rf = CalibratedClassifierCV(
                RandomForestClassifier(n_estimators=200, max_depth=12,
                                       class_weight='balanced', random_state=42),
                cv=3, method='sigmoid'
            )
            self._gb = CalibratedClassifierCV(
                GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                           learning_rate=0.05, random_state=42),
                cv=3, method='sigmoid'
            )
            self._rf.fit(X_scaled, y)
            self._gb.fit(X_scaled, y)
            self._is_trained = True
            self._save()
            logger.info("Classifier trained and saved.")
        except ImportError:
            logger.warning("scikit-learn not available; using rule-based fallback.")

    def set_threshold(self, threshold: float):
        """Tune detection threshold for FAR/Pd trade-off."""
        assert 0 < threshold < 1, "Threshold must be in (0, 1)"
        self.uas_threshold = threshold
        logger.info(f"Detection threshold set to {threshold:.3f}")

    def feature_importance(self) -> Dict[str, float]:
        """Return RF feature importances (if trained)."""
        if self._rf is None or not self._feature_names:
            return {}
        try:
            base = self._rf.estimator
            imp = base.feature_importances_
            return dict(sorted(zip(self._feature_names, imp),
                                key=lambda x: -x[1]))
        except Exception:
            return {}

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _ensemble_predict_proba(self, vec: np.ndarray) -> float:
        """Weighted ensemble probability (RF 60%, GB 40%)."""
        try:
            vec_scaled = self._scaler.transform(vec.reshape(1, -1))
            p_rf = self._rf.predict_proba(vec_scaled)[0][1]
            p_gb = self._gb.predict_proba(vec_scaled)[0][1]
            # Blend with rule-based signal
            p_rule = self._rule_based_score_from_vec(vec)
            return float(0.45 * p_rf + 0.35 * p_gb + 0.20 * p_rule)
        except Exception as e:
            logger.debug(f"Ensemble prediction error: {e}, using rules")
            return self._rule_based_score_from_vec(vec)

    def _rule_based_score(self, features: Dict) -> float:
        vec = self._features_to_vector(features)
        return self._rule_based_score_from_vec(vec)

    def _rule_based_score_from_vec(self, vec: np.ndarray) -> float:
        """
        Physics-informed scoring based on UAS RF characteristics.
        Higher scores = more UAS-like.
        """
        score = 0.0
        names = self._feature_names or self._default_feature_names()
        feat = dict(zip(names, vec))

        # Band match — strong indicator
        bm = feat.get('uas_band_match', 0.0)
        score += 0.30 * bm

        # Duty cycle: UAS RC links ~2–50% (not continuous, not silent)
        dc = feat.get('duty_cycle', 0.5)
        if 0.02 <= dc <= 0.60:
            score += 0.15 * (1 - abs(dc - 0.25) / 0.25)

        # Burst regularity: control links are highly periodic
        ir = feat.get('ibi_regularity', 0.0)
        if ir > 0.7:
            score += 0.15 * ir

        # Bandwidth: typical UAS links 1 kHz – 40 MHz
        bw = feat.get('bandwidth_3db_hz', 0.0)
        if 1e3 <= bw <= 40e6:
            score += 0.10

        # Moderate AM: spread spectrum / FHSS
        am = feat.get('am_modulation_index', 0.0)
        if 0.1 <= am <= 0.8:
            score += 0.08

        # Periodicity (cyclostationary)
        ps = feat.get('periodicity_score', 0.0)
        score += 0.12 * ps

        # Non-negligible bursts
        nb = feat.get('num_bursts', 0)
        if nb >= 2:
            score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        names = self._feature_names or self._default_feature_names()
        return np.array([features.get(n, 0.0) for n in names], dtype=np.float32)

    def _default_feature_names(self) -> List[str]:
        return [
            'signal_power_db', 'peak_freq_hz', 'bandwidth_3db_hz', 'bandwidth_99pct_hz',
            'spectral_centroid_hz', 'spectral_flatness', 'spectral_rolloff_hz',
            'peak_psd_db', 'freq_offset_from_center_hz',
            'num_bursts', 'mean_burst_duration_s', 'duty_cycle',
            'mean_inter_burst_interval_s', 'ibi_regularity',
            'kurtosis_real', 'kurtosis_imag', 'skewness_real', 'skewness_imag',
            'crest_factor', 'papr_db', 'iq_imbalance',
            'am_modulation_index', 'fm_deviation_hz', 'phase_discontinuity_rate',
            'envelope_variance', 'cyclic_peak_ratio', 'periodicity_score',
            'uas_band_match',
        ]

    def _top_features(self, features: Dict, vec: np.ndarray) -> Dict:
        """Return top-5 most diagnostic features for explainability."""
        names = self._feature_names or self._default_feature_names()
        importance = self.feature_importance()
        if importance:
            top_names = list(importance.keys())[:5]
        else:
            # Heuristic important features
            top_names = ['uas_band_match', 'duty_cycle', 'ibi_regularity',
                         'periodicity_score', 'bandwidth_3db_hz']
        return {k: float(features.get(k, 0.0)) for k in top_names if k in features}

    def _save(self):
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump({
                'rf': self._rf, 'gb': self._gb,
                'scaler': self._scaler,
                'feature_names': self._feature_names,
                'uas_threshold': self.uas_threshold,
            }, f)

    def _load_or_init(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self._rf = data['rf']
                self._gb = data['gb']
                self._scaler = data['scaler']
                self._feature_names = data['feature_names']
                self._is_trained = True
                logger.info("Pre-trained classifier loaded.")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using rule-based fallback.")
        else:
            logger.info("No pre-trained model found. Using rule-based classifier.")
