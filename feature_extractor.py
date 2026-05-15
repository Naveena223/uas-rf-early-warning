"""
RF Feature Extraction Engine
Extracts spectral, temporal, and modulation features from IQ/SigMF and spectrogram data.
"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RFFeatureExtractor:
    """
    Extracts a rich feature vector from RF data for UAS classification.

    Feature categories:
    - Spectral: PSD shape, bandwidth, center frequency, spectral flatness
    - Temporal: burst duration, duty cycle, inter-burst interval
    - Statistical: kurtosis, skewness, crest factor, AM/FM modulation indices
    - Cyclostationary: cyclic autocorrelation features (UAS links often burst-periodic)
    - SigMF metadata: annotated frequency, sample rate
    """

    # Known UAS control/telemetry frequency bands (MHz)
    UAS_BANDS = [
        (2400, 2483),   # 2.4 GHz ISM - DJI, Parrot, generic RC
        (5725, 5850),   # 5.8 GHz ISM - video downlink, telemetry
        (902, 928),     # 900 MHz ISM - long-range RC (ELRS, Crossfire)
        (433.05, 434.79),  # 433 MHz - EU RC, telemetry
        (858, 860),     # 860 MHz - some RC systems
        (1240, 1300),   # 1.2 GHz - FPV video
    ]

    def __init__(self, sample_rate: float = 1e6, nfft: int = 1024):
        self.sample_rate = sample_rate
        self.nfft = nfft

    def extract_from_iq(self, iq_samples: np.ndarray,
                        center_freq: float = 0.0,
                        metadata: Optional[Dict] = None) -> Dict:
        """Extract full feature vector from complex IQ samples."""
        features = {}

        # Basic signal power
        power = np.mean(np.abs(iq_samples) ** 2)
        features['signal_power_db'] = 10 * np.log10(power + 1e-12)

        # PSD via Welch
        freqs, psd = signal.welch(iq_samples, fs=self.sample_rate,
                                  nperseg=self.nfft, return_onesided=False)
        freqs = np.fft.fftshift(freqs) + center_freq
        psd = np.fft.fftshift(psd)
        features.update(self._spectral_features(freqs, psd, center_freq))

        # Temporal / burst features
        features.update(self._temporal_features(iq_samples))

        # Statistical moments
        features.update(self._statistical_features(iq_samples))

        # Modulation features
        features.update(self._modulation_features(iq_samples))

        # Cyclostationary indicator (burst repetition)
        features.update(self._cyclostationary_features(iq_samples))

        # Band match score — center_freq is in Hz, convert to MHz
        features['uas_band_match'] = self._band_match_score(
            center_freq / 1e6, features.get('bandwidth_3db_hz', 0) / 1e6
        )

        # SigMF metadata features
        if metadata:
            features.update(self._metadata_features(metadata, center_freq))

        return features

    def extract_from_spectrogram(self, spectrogram: np.ndarray,
                                  time_axis: Optional[np.ndarray] = None,
                                  freq_axis: Optional[np.ndarray] = None) -> Dict:
        """Extract features from a 2D spectrogram array (freq x time)."""
        features = {}
        spec_db = 10 * np.log10(np.abs(spectrogram) + 1e-12)

        # Mean spectrum
        mean_spectrum = np.mean(spec_db, axis=1)
        freqs = freq_axis if freq_axis is not None else np.arange(spectrogram.shape[0])
        center_freq = freqs[np.argmax(mean_spectrum)] if freq_axis is not None else 0.0

        features.update(self._spectral_features_from_spectrum(freqs, mean_spectrum, center_freq))
        features.update(self._temporal_features_from_spectrogram(spec_db, time_axis))
        features['uas_band_match'] = self._band_match_score(
            center_freq / 1e6, features.get('bandwidth_3db_hz', 0) / 1e6
        )
        features['spec_snr_db'] = float(np.max(mean_spectrum) - np.percentile(mean_spectrum, 10))
        features['spec_duty_cycle'] = self._spec_duty_cycle(spec_db)

        return features

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _spectral_features(self, freqs: np.ndarray, psd: np.ndarray,
                            center_freq: float) -> Dict:
        psd_norm = psd / (np.sum(psd) + 1e-12)
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # 3dB bandwidth
        half_power = psd[peak_idx] / 2.0
        above = psd >= half_power
        if np.any(above):
            idxs = np.where(above)[0]
            bw = freqs[idxs[-1]] - freqs[idxs[0]]
        else:
            bw = 0.0

        # Spectral centroid
        centroid = np.sum(freqs * psd_norm)

        # Spectral flatness (Wiener entropy) — low for narrowband bursts
        geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
        arith_mean = np.mean(psd) + 1e-12
        flatness = geo_mean / arith_mean

        # Occupied bandwidth (99% power)
        cum_psd = np.cumsum(psd_norm)
        low_idx = np.searchsorted(cum_psd, 0.005)
        high_idx = np.searchsorted(cum_psd, 0.995)
        occ_bw = freqs[min(high_idx, len(freqs)-1)] - freqs[max(low_idx, 0)]

        # Spectral roll-off (85%)
        rolloff_idx = np.searchsorted(cum_psd, 0.85)
        rolloff_freq = freqs[min(rolloff_idx, len(freqs)-1)]

        return {
            'peak_freq_hz': float(peak_freq),
            'bandwidth_3db_hz': float(abs(bw)),
            'bandwidth_99pct_hz': float(abs(occ_bw)),
            'spectral_centroid_hz': float(centroid),
            'spectral_flatness': float(np.clip(flatness, 0, 1)),
            'spectral_rolloff_hz': float(rolloff_freq),
            'peak_psd_db': float(10 * np.log10(psd[peak_idx] + 1e-12)),
            'freq_offset_from_center_hz': float(peak_freq - center_freq),
        }

    def _spectral_features_from_spectrum(self, freqs: np.ndarray,
                                          spectrum_db: np.ndarray,
                                          center_freq: float) -> Dict:
        psd = 10 ** (spectrum_db / 10)
        return self._spectral_features(freqs, psd, center_freq)

    def _temporal_features(self, iq: np.ndarray) -> Dict:
        envelope = np.abs(iq)
        threshold = np.mean(envelope) + np.std(envelope)
        bursts = envelope > threshold

        # Run-length encoding for burst detection
        transitions = np.diff(bursts.astype(int))
        burst_starts = np.where(transitions == 1)[0]
        burst_ends = np.where(transitions == -1)[0]

        if len(burst_starts) > 0 and len(burst_ends) > 0:
            if burst_ends[0] < burst_starts[0]:
                burst_ends = burst_ends[1:]
            min_len = min(len(burst_starts), len(burst_ends))
            burst_starts = burst_starts[:min_len]
            burst_ends = burst_ends[:min_len]
            burst_durations = (burst_ends - burst_starts) / self.sample_rate
            num_bursts = len(burst_durations)
            mean_burst_dur = float(np.mean(burst_durations)) if num_bursts > 0 else 0.0
            duty_cycle = float(np.sum(burst_durations) / (len(iq) / self.sample_rate))

            if num_bursts > 1:
                ibi = np.diff(burst_starts) / self.sample_rate
                mean_ibi = float(np.mean(ibi))
                ibi_regularity = float(1.0 - np.std(ibi) / (np.mean(ibi) + 1e-9))
            else:
                mean_ibi = 0.0
                ibi_regularity = 0.0
        else:
            num_bursts = 0
            mean_burst_dur = 0.0
            duty_cycle = float(np.mean(bursts))
            mean_ibi = 0.0
            ibi_regularity = 0.0

        return {
            'num_bursts': int(num_bursts),
            'mean_burst_duration_s': mean_burst_dur,
            'duty_cycle': float(np.clip(duty_cycle, 0, 1)),
            'mean_inter_burst_interval_s': mean_ibi,
            'ibi_regularity': float(np.clip(ibi_regularity, -1, 1)),
        }

    def _temporal_features_from_spectrogram(self, spec_db: np.ndarray,
                                              time_axis: Optional[np.ndarray]) -> Dict:
        # Use peak column power over time
        power_vs_time = np.max(spec_db, axis=0)
        threshold = np.mean(power_vs_time) + np.std(power_vs_time)
        active = power_vs_time > threshold
        duty = float(np.mean(active))

        transitions = np.diff(active.astype(int))
        n_bursts = int(np.sum(transitions == 1))

        if time_axis is not None and n_bursts > 1:
            starts = np.where(transitions == 1)[0]
            ibi_samples = np.diff(starts)
            dt = np.mean(np.diff(time_axis)) if len(time_axis) > 1 else 1.0
            mean_ibi = float(np.mean(ibi_samples) * dt)
            ibi_reg = float(1.0 - np.std(ibi_samples) / (np.mean(ibi_samples) + 1e-9))
        else:
            mean_ibi = 0.0
            ibi_reg = 0.0

        return {
            'num_bursts': n_bursts,
            'duty_cycle': duty,
            'mean_inter_burst_interval_s': mean_ibi,
            'ibi_regularity': float(np.clip(ibi_reg, -1, 1)),
            'mean_burst_duration_s': 0.0,
        }

    def _spec_duty_cycle(self, spec_db: np.ndarray) -> float:
        noise_floor = np.percentile(spec_db, 10)
        active = spec_db > (noise_floor + 6)  # 6dB above noise
        return float(np.mean(active))

    def _statistical_features(self, iq: np.ndarray) -> Dict:
        real, imag = iq.real, iq.imag
        envelope = np.abs(iq)

        crest = float(np.max(envelope) / (np.mean(envelope) + 1e-12))
        papr = float(10 * np.log10(np.max(envelope**2) / (np.mean(envelope**2) + 1e-12)))

        return {
            'kurtosis_real': float(kurtosis(real)),
            'kurtosis_imag': float(kurtosis(imag)),
            'skewness_real': float(skew(real)),
            'skewness_imag': float(skew(imag)),
            'crest_factor': float(np.clip(crest, 0, 50)),
            'papr_db': float(np.clip(papr, 0, 40)),
            'iq_imbalance': float(np.abs(np.var(real) - np.var(imag)) / (np.var(envelope) + 1e-12)),
        }

    def _modulation_features(self, iq: np.ndarray) -> Dict:
        # Instantaneous amplitude, phase, frequency
        envelope = np.abs(iq)
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase) / (2 * np.pi) * self.sample_rate

        am_index = float(np.std(envelope) / (np.mean(envelope) + 1e-12))
        fm_deviation = float(np.std(inst_freq))

        # Phase continuity (low = FSK/GFSK bursts common in UAS RC links)
        phase_jumps = np.sum(np.abs(np.diff(phase)) > np.pi) / len(phase)

        return {
            'am_modulation_index': float(np.clip(am_index, 0, 5)),
            'fm_deviation_hz': float(np.clip(fm_deviation, 0, self.sample_rate / 2)),
            'phase_discontinuity_rate': float(phase_jumps),
            'envelope_variance': float(np.var(envelope)),
        }

    def _cyclostationary_features(self, iq: np.ndarray) -> Dict:
        """
        Simplified cyclostationary analysis.
        UAS control links (e.g. FHSS, DSSS) exhibit periodic non-stationarity.
        """
        n = len(iq)
        if n < 256:
            return {'cyclic_peak_ratio': 0.0, 'periodicity_score': 0.0}

        # Squared envelope spectrum — reveals cyclic frequencies
        sq_env = np.abs(iq) ** 2
        sq_env -= np.mean(sq_env)
        spec = np.abs(np.fft.fft(sq_env, n=min(n, 8192))) ** 2
        spec = spec[:len(spec)//2]

        # Ratio of peak to mean (high → strong periodicity)
        peak_ratio = float(np.max(spec) / (np.mean(spec) + 1e-12))

        # Periodicity score: energy in harmonic peaks vs. total
        sorted_spec = np.sort(spec)[::-1]
        top5_energy = np.sum(sorted_spec[:5])
        total_energy = np.sum(spec) + 1e-12
        periodicity = float(top5_energy / total_energy)

        return {
            'cyclic_peak_ratio': float(np.clip(peak_ratio, 0, 1000)),
            'periodicity_score': float(np.clip(periodicity, 0, 1)),
        }

    def _band_match_score(self, center_mhz: float, bw_mhz: float) -> float:
        """Score 0–1 based on proximity to known UAS frequency bands."""
        if center_mhz <= 0:
            return 0.0
        for low, high in self.UAS_BANDS:
            band_center = (low + high) / 2
            band_half = (high - low) / 2 + max(bw_mhz / 2, 5)  # tolerance
            if abs(center_mhz - band_center) <= band_half:
                # Score based on how centered we are
                score = 1.0 - abs(center_mhz - band_center) / (band_half + 1e-9)
                return float(np.clip(score, 0, 1))
        # Soft score for proximity
        min_dist = min(
            abs(center_mhz - (l + h) / 2) / ((h - l) / 2 + 1)
            for l, h in self.UAS_BANDS
        )
        return float(np.clip(1.0 / (1.0 + min_dist * 0.1), 0, 0.4))

    def _metadata_features(self, metadata: Dict, center_freq: float) -> Dict:
        feats = {}
        if 'annotations' in metadata:
            ann = metadata['annotations']
            if ann:
                first = ann[0]
                feats['annotated_bw_hz'] = float(first.get('core:freq_upper_edge', 0) -
                                                   first.get('core:freq_lower_edge', 0))
        if 'global' in metadata:
            hw = metadata['global'].get('core:hw', '').lower()
            known_uas_hw = ['dji', 'parrot', 'phantom', 'mavic', 'fpv', 'elrs', 'crossfire', 'tbs']
            feats['hw_uas_keyword'] = float(any(k in hw for k in known_uas_hw))
        return feats

    def feature_names(self) -> list:
        """Return ordered list of feature names matching extract_from_iq output."""
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
