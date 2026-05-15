import numpy as np
import os

os.makedirs("test_data", exist_ok=True)

spec = np.random.randn(50, 512).astype(np.float32) * 0.1
spec[20:25, 100:120] = 5.0
spec[20:25, 250:270] = 5.0
spec[20:25, 400:420] = 5.0

freq_axis = np.linspace(2400e6, 2500e6, 50)
time_axis = np.linspace(0, 1.0, 512)
metadata  = {'center_freq_hz': 2440e6, 'sample_rate_hz': 100e6}

np.savez("test_data/uas_test.npz",
         spectrogram=spec,
         freq_axis=freq_axis,
         time_axis=time_axis,
         metadata=metadata)

print("Done! Created test_data/uas_test.npz")