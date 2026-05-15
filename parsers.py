"""
RF Data Parsers
Supports SigMF (.sigmf-meta + .sigmf-data) and spectrogram bundles (NPZ/CSV/PNG).
"""

import numpy as np
import json
import os
import logging
from typing import Dict, Tuple, Optional, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RFSegment:
    """A time-windowed RF segment ready for feature extraction."""
    segment_id: str
    source_file: str
    input_type: str                    # 'sigmf' | 'spectrogram'
    timestamp_start: float             # Unix epoch or relative seconds
    timestamp_end: float
    center_freq_hz: float
    sample_rate_hz: float
    iq_samples: Optional[np.ndarray]   # complex64, None for spectrogram input
    spectrogram: Optional[np.ndarray]  # freq×time float32, None for IQ input
    freq_axis: Optional[np.ndarray]
    time_axis: Optional[np.ndarray]
    metadata: Dict


class SigMFParser:
    """
    Parses SigMF datasets.
    Supports .sigmf-meta (JSON) + .sigmf-data (binary IQ) or .sigmf (zip) files.
    """

    SUPPORTED_DTYPES = {
        'cf32_le': np.complex64,
        'cf32_be': np.complex64,
        'cf64_le': np.complex128,
        'cf64_be': np.complex128,
        'ci16_le': np.int16,
        'ci8': np.int8,
        'ri16_le': np.int16,
    }

    def __init__(self, window_size: int = 65536, overlap: float = 0.25):
        """
        window_size: IQ samples per segment (default ~65k @ 1 MSPS = 65 ms)
        overlap: fractional overlap between windows
        """
        self.window_size = window_size
        self.overlap = overlap
        self.hop = int(window_size * (1 - overlap))

    def parse(self, meta_path: str) -> Generator[RFSegment, None, None]:
        """Yield RFSegments from a SigMF dataset."""
        meta = self._load_meta(meta_path)
        data_path = self._find_data_file(meta_path)

        sample_rate = float(meta['global'].get('core:sample_rate', 1e6))
        center_freq = float(meta['global'].get('core:frequency', 0.0))
        dtype_str = meta['global'].get('core:datatype', 'cf32_le')
        dtype = self.SUPPORTED_DTYPES.get(dtype_str, np.complex64)

        raw = self._load_raw(data_path, dtype)
        iq = self._to_complex(raw, dtype_str)

        # Walk sliding windows
        n = len(iq)
        window_idx = 0
        pos = 0
        while pos + self.window_size <= n:
            segment_iq = iq[pos:pos + self.window_size]
            t_start = pos / sample_rate
            t_end = (pos + self.window_size) / sample_rate

            # Find annotations overlapping this window
            anns = self._annotations_in_range(meta, pos, pos + self.window_size)

            yield RFSegment(
                segment_id=f"{os.path.basename(meta_path)}_seg{window_idx:04d}",
                source_file=meta_path,
                input_type='sigmf',
                timestamp_start=t_start,
                timestamp_end=t_end,
                center_freq_hz=center_freq,
                sample_rate_hz=sample_rate,
                iq_samples=segment_iq,
                spectrogram=None,
                freq_axis=None,
                time_axis=None,
                metadata={'global': meta['global'], 'annotations': anns},
            )
            pos += self.hop
            window_idx += 1

        # Handle trailing samples
        if pos < n and (n - pos) > self.window_size // 4:
            segment_iq = iq[pos:]
            yield RFSegment(
                segment_id=f"{os.path.basename(meta_path)}_seg{window_idx:04d}_tail",
                source_file=meta_path,
                input_type='sigmf',
                timestamp_start=pos / sample_rate,
                timestamp_end=n / sample_rate,
                center_freq_hz=center_freq,
                sample_rate_hz=sample_rate,
                iq_samples=segment_iq,
                spectrogram=None,
                freq_axis=None,
                time_axis=None,
                metadata={'global': meta['global'], 'annotations': []},
            )

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _load_meta(self, meta_path: str) -> Dict:
        with open(meta_path, 'r') as f:
            return json.load(f)

    def _find_data_file(self, meta_path: str) -> str:
        # Try .sigmf-data first, then .data, then same basename
        base = meta_path.replace('.sigmf-meta', '').replace('.json', '')
        candidates = [
            base + '.sigmf-data',
            base + '.data',
            base,
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Cannot find data file for {meta_path}")

    def _load_raw(self, path: str, dtype) -> np.ndarray:
        if dtype in (np.int16,):
            return np.fromfile(path, dtype=np.int16)
        return np.fromfile(path, dtype=np.float32)

    def _to_complex(self, raw: np.ndarray, dtype_str: str) -> np.ndarray:
        if 'cf32' in dtype_str or 'cf64' in dtype_str:
            if raw.dtype in [np.float32, np.float64]:
                return raw[0::2] + 1j * raw[1::2]
            return raw.astype(np.complex64)
        elif 'ci16' in dtype_str or 'ri16' in dtype_str:
            norm = raw.astype(np.float32) / 32768.0
            return norm[0::2] + 1j * norm[1::2]
        elif 'ci8' in dtype_str:
            norm = raw.astype(np.float32) / 128.0
            return norm[0::2] + 1j * norm[1::2]
        return raw.astype(np.complex64)

    def _annotations_in_range(self, meta: Dict, start: int, end: int) -> list:
        anns = meta.get('annotations', [])
        result = []
        for ann in anns:
            ann_start = ann.get('core:sample_start', 0)
            ann_len = ann.get('core:sample_count', 0)
            ann_end = ann_start + ann_len
            if ann_start < end and ann_end > start:
                result.append(ann)
        return result


class SpectrogramParser:
    """
    Parses spectrogram bundles.
    Supported formats:
    - NPZ: keys 'spectrogram' (freq×time), optionally 'freq_axis', 'time_axis', 'metadata'
    - CSV: comma-separated spectrogram (freq bins × time columns)
    - PNG/JPEG: grayscale image as spectrogram proxy (each row = freq bin)
    - Directory of NPZ/CSV files: processed in sorted order
    """

    def __init__(self, window_cols: int = 256, overlap: float = 0.25):
        self.window_cols = window_cols
        self.hop = int(window_cols * (1 - overlap))

    def parse(self, path: str) -> Generator[RFSegment, None, None]:
        if os.path.isdir(path):
            yield from self._parse_directory(path)
        elif path.endswith('.npz'):
            yield from self._parse_npz(path)
        elif path.endswith('.csv'):
            yield from self._parse_csv(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            yield from self._parse_image(path)
        else:
            raise ValueError(f"Unsupported spectrogram format: {path}")

    def _parse_directory(self, dir_path: str) -> Generator[RFSegment, None, None]:
        files = sorted([
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.endswith(('.npz', '.csv', '.png', '.jpg'))
        ])
        for f in files:
            yield from self.parse(f)

    def _parse_npz(self, path: str) -> Generator[RFSegment, None, None]:
        data = np.load(path, allow_pickle=True)
        spec = data['spectrogram'] if 'spectrogram' in data else list(data.values())[0]
        freq_axis = data['freq_axis'] if 'freq_axis' in data else None
        time_axis = data['time_axis'] if 'time_axis' in data else None
        meta = {}
        if 'metadata' in data:
            try:
                meta = data['metadata'].item()
            except Exception:
                pass

        center_freq = meta.get('center_freq_hz', 0.0)
        sample_rate = meta.get('sample_rate_hz', 1e6)

        yield from self._window_spectrogram(
            spec, freq_axis, time_axis, path, center_freq, sample_rate, meta
        )

    def _parse_csv(self, path: str) -> Generator[RFSegment, None, None]:
        spec = np.loadtxt(path, delimiter=',')
        yield from self._window_spectrogram(spec, None, None, path, 0.0, 1e6, {})

    def _parse_image(self, path: str) -> Generator[RFSegment, None, None]:
        try:
            from PIL import Image
            img = Image.open(path).convert('L')
            spec = np.array(img, dtype=np.float32)
            # Flip so low freq = index 0 (images are top-down)
            spec = spec[::-1, :]
            yield from self._window_spectrogram(spec, None, None, path, 0.0, 1e6, {})
        except ImportError:
            logger.error("Pillow not installed; cannot parse image spectrograms.")

    def _window_spectrogram(self, spec: np.ndarray,
                             freq_axis: Optional[np.ndarray],
                             time_axis: Optional[np.ndarray],
                             source: str,
                             center_freq: float,
                             sample_rate: float,
                             meta: Dict) -> Generator[RFSegment, None, None]:
        n_time = spec.shape[1] if spec.ndim == 2 else 1
        if spec.ndim == 1:
            spec = spec.reshape(-1, 1)

        window_idx = 0
        pos = 0
        while pos + self.window_cols <= n_time:
            window_spec = spec[:, pos:pos + self.window_cols]
            if time_axis is not None:
                t_start = float(time_axis[pos])
                t_end = float(time_axis[min(pos + self.window_cols - 1, len(time_axis)-1)])
            else:
                t_start = pos / sample_rate
                t_end = (pos + self.window_cols) / sample_rate

            yield RFSegment(
                segment_id=f"{os.path.basename(source)}_seg{window_idx:04d}",
                source_file=source,
                input_type='spectrogram',
                timestamp_start=t_start,
                timestamp_end=t_end,
                center_freq_hz=center_freq,
                sample_rate_hz=sample_rate,
                iq_samples=None,
                spectrogram=window_spec,
                freq_axis=freq_axis,
                time_axis=time_axis[pos:pos + self.window_cols] if time_axis is not None else None,
                metadata=meta,
            )
            pos += self.hop
            window_idx += 1


def auto_detect_and_parse(input_path: str, **kwargs) -> Generator[RFSegment, None, None]:
    """
    Auto-detects input format and returns a segment generator.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.endswith('.sigmf-meta') or input_path.endswith('.json'):
        logger.info(f"Detected SigMF format: {input_path}")
        yield from SigMFParser(**kwargs).parse(input_path)
    elif os.path.isdir(input_path):
        # Check if directory contains SigMF or spectrogram files
        files = os.listdir(input_path)
        if any(f.endswith('.sigmf-meta') for f in files):
            logger.info(f"Detected SigMF directory: {input_path}")
            for f in sorted(files):
                if f.endswith('.sigmf-meta'):
                    yield from SigMFParser(**kwargs).parse(os.path.join(input_path, f))
        else:
            logger.info(f"Detected spectrogram directory: {input_path}")
            yield from SpectrogramParser().parse(input_path)
    elif input_path.endswith('.npz'):
        yield from SpectrogramParser().parse(input_path)
    elif input_path.endswith('.csv'):
        yield from SpectrogramParser().parse(input_path)
    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        yield from SpectrogramParser().parse(input_path)
    else:
        raise ValueError(f"Unrecognised input format: {input_path}")
