import essentia.standard as es
import numpy as np
filename = './results/audio/dark_minimal_techno/0aAzuHdakBx3QziGAayHJh.mp3'
sample_rate = 44100

# Spectral Flux function
def calculate_spectral_flux(audio, sample_rate):
    # Initialize the necessary Essentia algorithms
    frame_size = 1024
    hop_size = 512
    window_function = es.Windowing(type='hann')
    spectrum = es.Spectrum(size=frame_size)
    spectral_flux = es.Flux()
    # Frame the audio
    frame_generator = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
    
    flux_values = []

    # Compute the spectral flux
    for frame in frame_generator:
        windowed_frame = window_function(frame)
        current_spectrum = spectrum(windowed_frame)
        flux = spectral_flux(current_spectrum)
        flux_values.append(flux)

    return np.mean(flux_values)

# Attack time functon
def calculate_attack_time(audio, sample_rate):
    attack_times = []
    onset_detector = es.OnsetDetectionGlobal()
    onsets = es.Onsets()([onset_detector(audio)],[1.0])
 
    envelope = es.Envelope()

    for index, onset in enumerate(onsets):
        # Extract a short segment around the onset
        start_sample = int(onset * sample_rate)
        end_sample = start_sample + int(5 * sample_rate)  # e.g., 100ms window
        segment = audio[start_sample:end_sample]

        # Calculate the amplitude envelope
        env = envelope(segment)

        peak_amplitude = 0
        peak_index = 0
        for i in range(1, len(env) - 1):
            if env[i] > env[i-1] and env[i] > env[i+1]:
                peak_amplitude = env[i]
                peak_index = i
                break

        # Calculate the attack time (time from onset to peak amplitude)
        attack_time = peak_index / sample_rate  # in seconds
        attack_times.append(attack_time)

    mean_attack_time = np.mean(attack_times)
    return mean_attack_time

# Full audio and per frequencies [20, 150, 400, 3200, 7000, 22000]
audio = es.MonoLoader(filename=filename)()
bandpass1 = es.BandPass(bandwidth=130, cutoffFrequency=85)
audio1 = bandpass1(audio)
bandpass2 = es.BandPass(bandwidth=250, cutoffFrequency=275)
audio2 = bandpass2(audio)
bandpass3 = es.BandPass(bandwidth=2800, cutoffFrequency=1800)
audio3 = bandpass3(audio)
bandpass4 = es.BandPass(bandwidth=3800, cutoffFrequency=5100)
audio4 = bandpass4(audio)
bandpass5 = es.BandPass(bandwidth=15000, cutoffFrequency=14500)
audio5 = bandpass5(audio)
audio_files = [audio, audio1, audio2, audio3, audio4, audio5]

# Pitch estimation
PitchMelodia = es.PitchMelodia()
pitch, confidence = PitchMelodia(audio)
pitch_final = pitch[confidence > 0.8]
mean_freq = np.mean(pitch_final)

# Melody range
min_pitch = 0 if not pitch_final else min(pitch_final)
max_pitch = 0 if not pitch_final else max(pitch_final)
melody_range = max_pitch - min_pitch

# Pulse Clarity
beat = es.BeatTrackerDegara() 
beat_position = beat(audio)
beatloudness = es.BeatsLoudness(beats=beat_position)
loudness, loudnessBandRatio= beatloudness(audio)
mean_clarity_per_band = np.mean(loudnessBandRatio,axis=0)
mean_clarity = np.mean(loudnessBandRatio)

# Meter (Do not use)
beatogram = es.Beatogram()
meter = es.Meter()
meter = meter(beatogram(loudness, loudnessBandRatio))

# Attack time
mean_attack_times = []

for audio in audio_files:
    mean_attack_time = calculate_attack_time(audio, sample_rate)
    mean_attack_times.append(mean_attack_time)
    
# Spectral Flux
spectral_flux_values = []
for audio in audio_files:
    flux = calculate_spectral_flux(audio, sample_rate)
    spectral_flux_values.append(flux)
