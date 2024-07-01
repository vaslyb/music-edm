import essentia.standard as es
import numpy as np
import os
import json

FRAME_SIZE = 4096
HOP_SIZE = 2048

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        return super(NumpyEncoder, self).default(obj)

def calculate_spectral_flatness(audio, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    
    # Frame generation and windowing
    frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
    windowed_frames = [es.Windowing(type='hann')(frame) for frame in frames]
    
    # Spectrum computation
    spectrums = [es.Spectrum()(windowed_frame) for windowed_frame in windowed_frames]
    
    # Spectral flatness computation
    flatness_values = []
    for spectrum in spectrums:
        # Convert spectrum to power spectrum
        power_spectrum = np.square(spectrum)
        # Calculate geometric mean
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        # Calculate arithmetic mean
        arithmetic_mean = np.mean(power_spectrum)
        # Calculate spectral flatness
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        flatness_values.append(flatness)
    
    return np.mean(flatness_values), np.var(flatness_values)

def calculate_spectral_flux(audio):
    window_function = es.Windowing(type='hann')
    spectrum = es.Spectrum(size=FRAME_SIZE)
    spectral_flux = es.Flux()
    frame_generator = es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True)
    
    flux_values = []
    for frame in frame_generator:
        windowed_frame = window_function(frame)
        current_spectrum = spectrum(windowed_frame)
        flux = spectral_flux(current_spectrum)
        flux_values.append(flux)
    return np.mean(flux_values), np.var(flux_values)
    
def calculate_attack_time(audio, sample_rate):
    attack_times = []
    onset_detector = es.OnsetDetectionGlobal()
    onsets = es.Onsets()([onset_detector(audio)],[1.0])
    envelope = es.Envelope()
    for index, onset in enumerate(onsets):
        start_sample = int(onset * sample_rate)
        end_sample = start_sample + int(0.1 * sample_rate)  # 100ms window
        segment = audio[start_sample:end_sample]
        env = envelope(segment)
        peak_amplitude = 0
        peak_index = 0
        for i in range(1, len(env) - 1):
            if env[i] > env[i-1] and env[i] > env[i+1]:
                peak_amplitude = env[i]
                peak_index = i
                break
        attack_time = peak_index / sample_rate  # in seconds
        attack_times.append(attack_time)
    mean_attack_time = np.mean(attack_times)
    var_attack_time = np.var(attack_times)
    return mean_attack_time, var_attack_time

def calculate_attack_slope(audio, sample_rate):
    attack_slopes = []
    onset_detector = es.OnsetDetectionGlobal()
    onsets = es.Onsets()([onset_detector(audio)],[1.0])
    envelope = es.Envelope()
    for index, onset in enumerate(onsets):
        start_sample = int(onset * sample_rate)
        end_sample = start_sample + int(0.1 * sample_rate)
        segment = audio[start_sample:end_sample]
        env = envelope(segment)
        peak_amplitude = 0
        peak_index = 0
        start_amplitude = env[0]
        for i in range(1, len(env) - 1):
            if env[i] > env[i-1] and env[i] > env[i+1]:
                peak_amplitude = env[i]
                peak_index = i
                break
        amplitude_diff = peak_amplitude - start_amplitude
        attack_slope = amplitude_diff / (peak_index / sample_rate)
        attack_slopes.append(attack_slope)
    mean_attack_slope = np.mean(attack_slopes)
    var_attack_slope = np.var(attack_slopes)
    return mean_attack_slope, var_attack_slope
        

def process_audio_file(filepath, sample_rate=44100):
    # Filter audio into 5 bands
    audio = es.MonoLoader(filename=filepath)()
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
    
    # Pitch
    PredominantPitchMelodia = es.PredominantPitchMelodia()
    pitch, confidence = PredominantPitchMelodia(audio)
    pitch_final = pitch[confidence > 0.8]
    mean_freq = np.mean(pitch_final)
    var_freq = np.var(pitch_final)
    
    # Melody range
    min_pitch = 0 if len(pitch_final) == 0 else min(pitch_final)
    max_pitch = 0 if len(pitch_final) == 0 else max(pitch_final)
    melody_range = max_pitch - min_pitch
    
    # Pulse clarity
    beat = es.BeatTrackerDegara() 
    beat_position = beat(audio)
    beatloudness = es.BeatsLoudness(beats=beat_position)
    loudness, loudnessBandRatio = beatloudness(audio)
    entropy = es.Entropy()
    entropia_clarity_per_band = []
    for i in range(loudnessBandRatio.shape[1]):
        entropia_clarity_per_band.append(entropy(loudnessBandRatio[:, i]))
    entropia_clarity = entropy(entropia_clarity_per_band)
    mean_clarity_per_band = np.mean(loudnessBandRatio, axis=0)
    var_clarity_per_band = np.var(loudnessBandRatio, axis=0)
    mean_clarity = np.mean(loudnessBandRatio)
    var_clarity = np.var(loudnessBandRatio)
    
    # Meter
    beatogram = es.Beatogram()
    meter = es.Meter()
    meter = meter(beatogram(loudness, loudnessBandRatio))
    
    # Attack time
    mean_attack_times = []
    var_attack_times = []
    for audio in audio_files:
        mean_attack_time, var_attack_time = calculate_attack_time(audio, sample_rate)
        mean_attack_times.append(mean_attack_time)
        var_attack_times.append(var_attack_time)
    
    # Attack slope
    mean_attack_slopes = []
    var_attack_slopes = []
    for audio in audio_files:
        mean_attack_slope, var_attack_slope = calculate_attack_slope(audio, sample_rate)
        mean_attack_slopes.append(mean_attack_slope)
        var_attack_slopes.append(var_attack_slope)
    
    # Spectral flux
    mean_spectral_flux_values = []
    var_spectral_flux_values = []
    for audio in audio_files:
        mean_flux, var_flux = calculate_spectral_flux(audio)
        mean_spectral_flux_values.append(mean_flux)
        var_spectral_flux_values.append(var_flux)
        
    # Spectral flatness
    mean_flatness_values = []
    var_flatness_values = []
    for audio in audio_files:
        mean_flatness,var_flatness = calculate_spectral_flatness(audio)
        mean_flatness_values.append(mean_flatness)
        var_flatness_values.append(var_flatness)

    features = {
        'meter': meter,
        'mean_freq': mean_freq,
        'var_freq': var_freq,
        'melody_range': melody_range,
        'mean_clarity': mean_clarity,
        'var_clarity': var_clarity,
        'mean_clarity_per_band': mean_clarity_per_band.tolist(),
        'var_clarity_per_band': var_clarity_per_band.tolist(),
        'entropia_clarity': entropia_clarity,
        'entropia_clarity_per_band': entropia_clarity_per_band,
        'mean_attack_times': mean_attack_times[0],
        'var_attack_times': var_attack_times[0],
        'mean_attack_times_per_band': mean_attack_times[1:],
        'var_attack_times_per_band': var_attack_times[1:],
        'mean_attack_slopes': mean_attack_slopes[0],
        'var_attack_slopes': var_attack_slopes[0],
        'mean_attack_slopes_per_band': mean_attack_slopes[1:],
        'var_attack_slopes_per_band': var_attack_slopes[1:],
        'mean_spectral_flux': mean_spectral_flux_values[0],
        'var_spectral_flux': var_spectral_flux_values[0],
        'mean_spectral_flux_per_band': mean_spectral_flux_values[1:],
        'var_spectral_flux_per_band': var_spectral_flux_values[1:],
        'mean_flatness': mean_flatness_values[0],
        'var_flatness': var_flatness_values[0],
        'mean_flatness_per_band': mean_flatness_values[1:],
        'var_flatness_per_band': var_flatness_values[1:]
    }
    
    return features

def process_all_files(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                filepath = os.path.join(root, file)
                print(f'Processing {filepath}')
                features = process_audio_file(filepath)
                relative_path = os.path.relpath(filepath, input_dir)
                output_filepath = os.path.join(output_dir, relative_path.replace('.mp3', '.json'))
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                with open(output_filepath, 'w') as f:
                    json.dump(features, f, cls=NumpyEncoder, indent=4)

if __name__ == '__main__':
    input_directory = './results/audio/'
    output_directory = './results/features_manual/'
    process_all_files(input_directory, output_directory)
