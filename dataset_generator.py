"""
Synthetic RF Dataset Generator
Produces realistic IQ datasets with UAS-like signals and RF clutter.
Generates SigMF-compatible .sigmf-meta + .sigmf-data files.

Signal types:
  UAS-like:
    - FHSS burst (DJI-style 2.4 GHz)
    - GFSK burst with periodic IBI (RC control link)
    - DSSS spread-spectrum (video downlink)
    - 5.8 GHz FPV telemetry bursts
  Clutter:
    - Wi-Fi OFDM (overlapping ISM band)
    - Bluetooth (2.4 GHz FHSS)
    - Continuous narrowband CW interference
    - Broadband AWGN
    - LTE-like OFDM
"""

import numpy as np
import json
import os
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    label: str          # 'uas' | 'clutter'
    signal_type: str    # specific type string
    start_sample: int
    end_sample: int
    center_freq_offset_hz: float
    power_db: float


class RFDatasetGenerator:
    """Generates synthetic RF datasets with ground-truth labels."""

    SAMPLE_RATE = 20e6   # 20 MSPS
    CENTER_FREQ = 2.44e9  # 2.44 GHz ISM centre

    def __init__(self, sample_rate: float = None, seed: int = 42):
        self.fs = sample_rate or self.SAMPLE_RATE
        self.rng = np.random.default_rng(seed)

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate_dataset(self,
                          output_dir: str,
                          duration_s: float = 10.0,
                          uas_density: float = 0.35,
                          clutter_density: float = 0.50,
                          snr_db_range: Tuple[float, float] = (5, 25),
                          name: str = 'test_dataset') -> Dict:
        """
        Generate a complete SigMF dataset file.
        Returns dict with file paths and ground truth events.
        """
        os.makedirs(output_dir, exist_ok=True)
        n_samples = int(duration_s * self.fs)
        iq = np.zeros(n_samples, dtype=np.complex64)
        events: List[SignalEvent] = []

        # Add AWGN floor
        noise_power = 0.01
        iq += (self.rng.standard_normal(n_samples) +
               1j * self.rng.standard_normal(n_samples)).astype(np.complex64) * np.sqrt(noise_power / 2)

        # Place UAS signals
        if uas_density > 0:
            uas_events = self._place_uas_signals(iq, n_samples, uas_density, snr_db_range)
            events.extend(uas_events)

        # Place clutter
        if clutter_density > 0:
            clutter_events = self._place_clutter(iq, n_samples, clutter_density)
            events.extend(clutter_events)

        # Save SigMF
        meta_path = os.path.join(output_dir, f'{name}.sigmf-meta')
        data_path = os.path.join(output_dir, f'{name}.sigmf-data')
        annotations = self._events_to_annotations(events)
        self._save_sigmf(iq, meta_path, data_path, annotations)

        # Ground truth JSON
        gt_path = os.path.join(output_dir, f'{name}_groundtruth.json')
        ground_truth = {
            'dataset': name,
            'sample_rate_hz': self.fs,
            'center_freq_hz': self.CENTER_FREQ,
            'duration_s': duration_s,
            'n_samples': n_samples,
            'events': [asdict(e) for e in events],
            'n_uas_events': sum(1 for e in events if e.label == 'uas'),
            'n_clutter_events': sum(1 for e in events if e.label == 'clutter'),
        }
        with open(gt_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)

        logger.info(
            f"Generated {name}: {duration_s}s, "
            f"{ground_truth['n_uas_events']} UAS / "
            f"{ground_truth['n_clutter_events']} clutter events"
        )
        return {
            'meta_path': meta_path,
            'data_path': data_path,
            'gt_path': gt_path,
            'ground_truth': ground_truth,
        }

    def generate_spectrogram_bundle(self,
                                     output_dir: str,
                                     duration_s: float = 5.0,
                                     name: str = 'spec_test') -> Dict:
        """Generate an NPZ spectrogram bundle."""
        os.makedirs(output_dir, exist_ok=True)
        n_samples = int(duration_s * self.fs)
        iq = np.zeros(n_samples, dtype=np.complex64)
        iq += (self.rng.standard_normal(n_samples) +
               1j * self.rng.standard_normal(n_samples)).astype(np.complex64) * 0.05
        events = self._place_uas_signals(iq, n_samples, 0.40, (8, 20))

        # STFT spectrogram
        from scipy import signal as sig
        nfft = 512
        _, t_axis, spec = sig.spectrogram(
            iq, fs=self.fs, nperseg=nfft, noverlap=nfft//2, return_onesided=False
        )
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/self.fs)) + self.CENTER_FREQ
        spec = np.fft.fftshift(np.abs(spec), axes=0)

        npz_path = os.path.join(output_dir, f'{name}.npz')
        np.savez(npz_path,
                 spectrogram=spec.astype(np.float32),
                 freq_axis=freqs.astype(np.float64),
                 time_axis=t_axis.astype(np.float64),
                 metadata=np.array({'center_freq_hz': self.CENTER_FREQ,
                                    'sample_rate_hz': self.fs}, dtype=object))
        return {'npz_path': npz_path, 'n_uas_events': len(events)}

    # ─── Signal synthesis helpers ─────────────────────────────────────────────

    def _place_uas_signals(self, iq: np.ndarray, n: int,
                            density: float, snr_range: Tuple) -> List[SignalEvent]:
        events = []
        signal_types = ['dji_fhss', 'gfsk_rc', 'dsss_video', 'fpv_telemetry']
        weights = [0.35, 0.30, 0.20, 0.15]

        t_pos = int(n * 0.05)
        while t_pos < int(n * 0.95):
            sig_type = self.rng.choice(signal_types, p=weights)
            gap = int(self.rng.uniform(0.05, 0.3) * self.fs)
            snr = float(self.rng.uniform(*snr_range))

            if sig_type == 'dji_fhss':
                burst, dur = self._dji_fhss_burst(snr)
            elif sig_type == 'gfsk_rc':
                burst, dur = self._gfsk_rc_burst(snr)
            elif sig_type == 'dsss_video':
                burst, dur = self._dsss_signal(snr)
            else:
                burst, dur = self._fpv_telemetry_burst(snr)

            end = min(t_pos + len(burst), n)
            actual_len = end - t_pos
            iq[t_pos:end] += burst[:actual_len]

            events.append(SignalEvent(
                label='uas',
                signal_type=sig_type,
                start_sample=t_pos,
                end_sample=end,
                center_freq_offset_hz=float(self.rng.uniform(-5e6, 5e6)),
                power_db=snr,
            ))

            t_pos = end + gap
            if self.rng.random() > density:
                t_pos += int(self.rng.uniform(0.1, 0.5) * self.fs)

        return events

    def _place_clutter(self, iq: np.ndarray, n: int,
                        density: float) -> List[SignalEvent]:
        events = []
        clutter_types = ['wifi_ofdm', 'bluetooth', 'cw_interference', 'lte_like']

        t_pos = int(n * 0.02)
        while t_pos < int(n * 0.98):
            c_type = self.rng.choice(clutter_types)
            gap = int(self.rng.uniform(0.02, 0.15) * self.fs)

            if c_type == 'wifi_ofdm':
                burst, _ = self._wifi_ofdm(snr_db=float(self.rng.uniform(15, 30)))
            elif c_type == 'bluetooth':
                burst, _ = self._bluetooth_burst(snr_db=float(self.rng.uniform(10, 25)))
            elif c_type == 'cw_interference':
                burst, _ = self._cw_tone(snr_db=float(self.rng.uniform(5, 20)))
            else:
                burst, _ = self._lte_like(snr_db=float(self.rng.uniform(10, 25)))

            end = min(t_pos + len(burst), n)
            actual_len = end - t_pos
            iq[t_pos:end] += burst[:actual_len]

            events.append(SignalEvent(
                label='clutter',
                signal_type=c_type,
                start_sample=t_pos,
                end_sample=end,
                center_freq_offset_hz=float(self.rng.uniform(-8e6, 8e6)),
                power_db=float(self.rng.uniform(10, 30)),
            ))

            t_pos = end + gap

        return events

    # ── UAS signal synthesizers ───────────────────────────────────────────────

    def _dji_fhss_burst(self, snr_db: float) -> Tuple[np.ndarray, float]:
        """DJI-style FHSS: 10 ms bursts at ~10 Hz, OFDM-like."""
        burst_len = int(0.010 * self.fs)
        n_bursts = int(self.rng.integers(5, 15))
        ibi = int(0.090 * self.fs)  # ~10 Hz repetition
        total = n_bursts * (burst_len + ibi)
        sig_arr = np.zeros(total, dtype=np.complex64)

        amplitude = self._snr_to_amplitude(snr_db)
        for i in range(n_bursts):
            t0 = i * (burst_len + ibi)
            # OFDM-like: sum of subcarriers
            n_sc = 64
            sc_freqs = np.linspace(-5e6, 5e6, n_sc)
            burst = np.zeros(burst_len, dtype=np.complex64)
            for f in sc_freqs[::4]:
                phase = self.rng.uniform(0, 2 * np.pi)
                t = np.arange(burst_len) / self.fs
                burst += np.exp(1j * (2 * np.pi * f * t + phase)).astype(np.complex64)
            burst /= (n_sc // 4)
            sig_arr[t0:t0 + burst_len] = burst * amplitude

        return sig_arr, total / self.fs

    def _gfsk_rc_burst(self, snr_db: float) -> Tuple[np.ndarray, float]:
        """GFSK RC link: periodic 22 ms bursts, ~45 Hz repetition."""
        burst_len = int(0.022 * self.fs)
        n_bursts = int(self.rng.integers(8, 25))
        ibi = int(0.0222 * self.fs)  # 22 ms IBI → 45 Hz
        total = n_bursts * (burst_len + ibi)
        sig_arr = np.zeros(total, dtype=np.complex64)

        amplitude = self._snr_to_amplitude(snr_db)
        bit_rate = 100e3
        f_dev = 50e3
        bits_per_burst = int(bit_rate * burst_len / self.fs)

        for i in range(n_bursts):
            t0 = i * (burst_len + ibi)
            bits = self.rng.integers(0, 2, bits_per_burst)
            # Simple FM: ±f_dev
            freq_seq = np.repeat(2 * bits - 1, int(self.fs / bit_rate) + 1)[:burst_len]
            phase = np.cumsum(freq_seq * f_dev / self.fs * 2 * np.pi)
            sig_arr[t0:t0 + burst_len] = amplitude * np.exp(1j * phase).astype(np.complex64)

        return sig_arr, total / self.fs

    def _dsss_signal(self, snr_db: float) -> Tuple[np.ndarray, float]:
        """DSSS spread spectrum video downlink: continuous wideband."""
        dur = float(self.rng.uniform(0.1, 0.5))
        n = int(dur * self.fs)
        amplitude = self._snr_to_amplitude(snr_db) * 0.5
        # PN sequence spread
        chip_rate = 10e6
        chips = 2 * self.rng.integers(0, 2, int(chip_rate * dur)) - 1
        chips_upsampled = np.repeat(chips, int(self.fs / chip_rate) + 1)[:n]
        carrier_phase = self.rng.uniform(0, 2 * np.pi)
        t = np.arange(n) / self.fs
        carrier = np.exp(1j * (2 * np.pi * 1e6 * t + carrier_phase))
        sig_arr = (amplitude * chips_upsampled * carrier).astype(np.complex64)
        return sig_arr, dur

    def _fpv_telemetry_burst(self, snr_db: float) -> Tuple[np.ndarray, float]:
        """FPV telemetry: short bursts at 5.8 GHz offset (simulated via freq offset)."""
        burst_len = int(0.005 * self.fs)
        n_bursts = int(self.rng.integers(3, 10))
        ibi = int(self.rng.uniform(0.010, 0.050) * self.fs)
        total = n_bursts * (burst_len + ibi)
        sig_arr = np.zeros(total, dtype=np.complex64)
        amplitude = self._snr_to_amplitude(snr_db)

        for i in range(n_bursts):
            t0 = i * (burst_len + ibi)
            f_offset = self.rng.uniform(-2e6, 2e6)
            t = np.arange(burst_len) / self.fs
            sig_arr[t0:t0 + burst_len] = amplitude * np.exp(
                1j * 2 * np.pi * f_offset * t
            ).astype(np.complex64)

        return sig_arr, total / self.fs

    # ── Clutter synthesizers ──────────────────────────────────────────────────

    def _wifi_ofdm(self, snr_db: float) -> Tuple[np.ndarray, float]:
        dur = float(self.rng.uniform(0.002, 0.020))
        n = int(dur * self.fs)
        # 802.11 OFDM: 52 data subcarriers over 20 MHz
        amplitude = self._snr_to_amplitude(snr_db)
        sig_arr = np.zeros(n, dtype=np.complex64)
        sc_freqs = np.linspace(-9e6, 9e6, 52)
        for f in sc_freqs:
            phase = self.rng.uniform(0, 2 * np.pi)
            t = np.arange(n) / self.fs
            sig_arr += amplitude / 52 * np.exp(1j * (2 * np.pi * f * t + phase)).astype(np.complex64)
        return sig_arr, dur

    def _bluetooth_burst(self, snr_db: float) -> Tuple[np.ndarray, float]:
        """BT GFSK: very short 625 µs slots, no periodic UAS-like IBI."""
        dur = float(self.rng.uniform(0.000625, 0.005))
        n = int(dur * self.fs)
        amplitude = self._snr_to_amplitude(snr_db)
        f_dev = 150e3
        bits = self.rng.integers(0, 2, int(1e6 * dur))
        freq_seq = np.repeat(2 * bits - 1, int(self.fs / 1e6) + 1)[:n]
        phase = np.cumsum(freq_seq * f_dev / self.fs * 2 * np.pi)
        return amplitude * np.exp(1j * phase).astype(np.complex64), dur

    def _cw_tone(self, snr_db: float) -> Tuple[np.ndarray, float]:
        dur = float(self.rng.uniform(0.05, 0.5))
        n = int(dur * self.fs)
        amplitude = self._snr_to_amplitude(snr_db)
        f = self.rng.uniform(-8e6, 8e6)
        t = np.arange(n) / self.fs
        return (amplitude * np.exp(1j * 2 * np.pi * f * t)).astype(np.complex64), dur

    def _lte_like(self, snr_db: float) -> Tuple[np.ndarray, float]:
        dur = float(self.rng.uniform(0.01, 0.1))
        n = int(dur * self.fs)
        amplitude = self._snr_to_amplitude(snr_db)
        # LTE-like: continuous OFDM with 15 kHz subcarrier spacing
        n_sc = 100
        sc_freqs = np.arange(-n_sc//2, n_sc//2) * 15e3
        sig_arr = np.zeros(n, dtype=np.complex64)
        for f in sc_freqs:
            phase = self.rng.uniform(0, 2 * np.pi)
            t = np.arange(n) / self.fs
            sig_arr += amplitude / n_sc * np.exp(1j * (2 * np.pi * f * t + phase)).astype(np.complex64)
        return sig_arr, dur

    # ── SigMF I/O ──────────────────────────────────────────────────────────────

    def _save_sigmf(self, iq: np.ndarray, meta_path: str, data_path: str,
                     annotations: list):
        # Save IQ as interleaved float32
        interleaved = np.empty(2 * len(iq), dtype=np.float32)
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag
        interleaved.tofile(data_path)

        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self.fs,
                "core:frequency": self.CENTER_FREQ,
                "core:version": "0.0.2",
                "core:hw": "Synthetic RF generator v1.0",
                "core:description": "Synthetic UAS/clutter RF dataset",
            },
            "captures": [{"core:sample_start": 0, "core:frequency": self.CENTER_FREQ}],
            "annotations": annotations,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _events_to_annotations(self, events: List[SignalEvent]) -> list:
        anns = []
        for ev in events:
            ann = {
                "core:sample_start": ev.start_sample,
                "core:sample_count": ev.end_sample - ev.start_sample,
                "core:label": ev.label,
                "core:comment": ev.signal_type,
                "rf:power_db": ev.power_db,
            }
            anns.append(ann)
        return anns

    def _snr_to_amplitude(self, snr_db: float) -> float:
        noise_power = 0.01
        signal_power = noise_power * 10 ** (snr_db / 10)
        return float(np.sqrt(signal_power))
