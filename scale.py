import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import sounddevice as sd

# Parameters
fs = 44100  # Sampling frequency
duration = 5  # seconds
f = 500  # Sine frequency, Hz
input_device_index = 1
output_device_index = 22
start_freq = 500
end_freq = 750

# Set the default input and output devices
sd.default.device = [input_device_index, output_device_index]

# Generate sine wave
# Time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Calculate instantaneous frequency at each point in time
instantaneous_freq = np.linspace(start_freq, end_freq, t.shape[0])

# Generate sine wave with varying frequency
phase = np.cumsum(2 * np.pi * instantaneous_freq / fs)  # Integrate frequency to get phase
sine_wave = 0.5 * np.sin(phase)

# Play and record simultaneously with specified input and output devices
recorded_audio = sd.playrec(sine_wave[:, np.newaxis], samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Wait until recording is finished

# FFT analysis
fft_result = fft(recorded_audio.flatten())
fft_freq = np.fft.fftfreq(len(fft_result), 1/fs)

# Convert the FFT result to magnitude
fft_magnitude = np.abs(fft_result)

# Find indices corresponding to the frequency range 100 Hz to 1000 Hz
freq_indices = np.where((fft_freq >= 100) & (fft_freq <= 1000))
exclusion_indices = np.where((fft_freq < 500) | (fft_freq > 750))

# Calculate the average magnitude within this frequency range
average_magnitude = np.mean(fft_magnitude[freq_indices])
average_magnitude_excluded = np.mean(fft_magnitude[exclusion_indices])

print(f"Average magnitude in the 100 Hz to 1000 Hz range: {average_magnitude_excluded}")

# Plot FFT results
plt.figure(figsize=(10, 6))
plt.plot(fft_freq, np.abs(fft_result))
plt.title('FFT of Recorded Audio')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 2000)  # Focus on the 100Hz to 1000Hz range
plt.grid(True)
plt.show()