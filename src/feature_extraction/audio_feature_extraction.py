import subprocess
import os
import essentia.standard as es
import numpy as np
import json
import pandas as pd

FRAME_SIZE = 4096
HOP_SIZE = 2048

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

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

def calculate_bpm_degara(audio):
    beat = es.BeatTrackerDegara()
    beat_positions = beat(audio)
    return 60 / np.mean(np.diff(beat_positions))

def calculate_bpm_multi(audio):
    beat = es.BeatTrackerMultiFeature()
    beat_positions,_ = beat(audio)
    return 60 / np.mean(np.diff(beat_positions))

def calculate_bpm_rythm(audio):
    rhythm = es.RhythmExtractor()
    bpm, _, _, _ = rhythm(audio)

def calculate_bpm_rhytm_2013(audio):
    rhythm2013 = es.RhythmExtractor2013()
    bpm, _,_,_ ,_= rhythm2013(audio)
    
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
    
    # BPM
    bpm_degara = calculate_bpm_degara(audio)
    bpm_multi = calculate_bpm_multi(audio)
    bpm_rhythm = calculate_bpm_rythm(audio)
    bpm_rhythm_2013 = calculate_bpm_rhytm_2013(audio)
    
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
        'bpm_degara': bpm_degara,
        'bpm_multi': bpm_multi,
        'bpm_rhythm': bpm_rhythm,
        'bpm_rhythm_2013': bpm_rhythm_2013,
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


if __name__ == '__main__':

    # Path to the directory containing input files
    input_dir = "./results/audio/"

    # Path to the directory where you want to save output files
    output_dir = "./results/features/"
    os.makedirs(output_dir, exist_ok=True)

    # Path to the executable file
    exe_file = "./streaming_extractor_music"

    # Walk through the input directory recursively
    for root, dirs, files in os.walk(input_dir):
        # Iterate over each file in the current directory
        for file in files:
            # Construct the full path of the input file
            input_path = os.path.join(root, file)

            # Construct the relative path of the input file
            relative_path = os.path.relpath(input_path, input_dir)

            # Construct the output directory structure based on the input directory structure
            output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)

            # Construct the output file path with the same name as the input file
            output_path = os.path.join(output_subdir, file.split(".")[0] + ".json")

            # Run the executable file with the input and output parameters
            subprocess.run([exe_file, input_path, output_path])
            
            # Load the JSON file produced by the subprocess
            # with open(output_path, 'r') as f:
            #     json_data = json.load(f)
            json_data = pd.read_json(output_path)

            # Get the results from process_all_files
            extracted_data = {
                "loudness": json_data['lowlevel'][0],
                "dissonance_mean": json_data['lowlevel'][6]['mean'],
                "dynamic_complexity": json_data['lowlevel'][7],
                "pitch_salience_mean": json_data['lowlevel'][18]['mean'],
                "spectral_centroid_mean": json_data['lowlevel'][22]['mean'],
                "spectral_complexity_mean": json_data['lowlevel'][23]['mean'],
                "spectral_decrease_mean": json_data['lowlevel'][24]['mean'],
                "spectral_energy_mean": json_data['lowlevel'][25]['mean'],
                "spectral_energyband_high_mean": json_data['lowlevel'][26]['mean'],
                "spectral_energyband_low_mean": json_data['lowlevel'][27]['mean'],
                "spectral_energyband_middle_mean": json_data['lowlevel'][28]['mean'],
                "spectral_energyband_middle_high_mean": json_data['lowlevel'][29]['mean'],
                "spectral_energyband_middle_low_mean": json_data['lowlevel'][30]['mean'],
                "spectral_entropy_mean": json_data['lowlevel'][31]['mean'],
                "spectral_kurtosis_mean": json_data['lowlevel'][33]['mean'],
                "spectral_rms_mean": json_data['lowlevel'][34]['mean'],
                "spectral_rolloff_mean": json_data['lowlevel'][35]['mean'],
                "spectral_skewness_mean": json_data['lowlevel'][36]['mean'],
                "spectral_spread_mean": json_data['lowlevel'][37]['mean'],
                "zerocr_mean": json_data['lowlevel'][39]['mean'],
                "mfcc1_mean": json_data['lowlevel'][44]['mean'][0],
                "mfcc2_mean": json_data['lowlevel'][44]['mean'][1],
                "mfcc3_mean": json_data['lowlevel'][44]['mean'][2],
                "mfcc4_mean": json_data['lowlevel'][44]['mean'][3],
                "mfcc5_mean": json_data['lowlevel'][44]['mean'][4],
                "mfcc6_mean": json_data['lowlevel'][44]['mean'][5],
                "mfcc7_mean": json_data['lowlevel'][44]['mean'][6],
                "mfcc8_mean": json_data['lowlevel'][44]['mean'][7],
                "mfcc9_mean": json_data['lowlevel'][44]['mean'][8],
                "mfcc10_mean": json_data['lowlevel'][44]['mean'][9],
                "mfcc11_mean": json_data['lowlevel'][44]['mean'][10],
                "mfcc12_mean": json_data['lowlevel'][44]['mean'][11],
                "mfcc13_mean": json_data['lowlevel'][44]['mean'][12],
                "chroma1_mean": json_data['tonal'][72]['mean'][0],
                "chroma2_mean": json_data['tonal'][72]['mean'][1],
                "chroma3_mean": json_data['tonal'][72]['mean'][2],
                "chroma4_mean": json_data['tonal'][72]['mean'][3],
                "chroma5_mean": json_data['tonal'][72]['mean'][4],
                "chroma6_mean": json_data['tonal'][72]['mean'][5],
                "chroma7_mean": json_data['tonal'][72]['mean'][6],
                "chroma8_mean": json_data['tonal'][72]['mean'][7],
                "chroma9_mean": json_data['tonal'][72]['mean'][8],
                "chroma10_mean": json_data['tonal'][72]['mean'][9],
                "chroma11_mean": json_data['tonal'][72]['mean'][10],
                "chroma12_mean": json_data['tonal'][72]['mean'][11],
                "bpm": json_data['rhythm'][52],
                "danceability": json_data['rhythm'][59],
                "onset_rate": json_data['rhythm'][60],
                "chords_changes_rate": json_data['tonal'][63],
                "chords_number_rate": json_data['tonal'][64],
                "chord": json_data['tonal'][75],
                "chord_strength": json_data['tonal'][66]['mean'],
                "chord_scale": json_data['tonal'][76],
                "key": json_data['tonal'][77],
                "key_scale": json_data['tonal'][78],
                "key_strength": json_data['tonal'][67]
            }

            features_manual = process_audio_file(input_path)

            # Extract the desired parts from process_all_files results
            manual_extracted_data = {
                "meter": features_manual['meter'],
                "pulse_clarity_mean": features_manual['mean_clarity'],
                "pulse_clarity_low_mean": features_manual['mean_clarity_per_band'][0],
                "pulse_clarity_middle_low_mean": features_manual['mean_clarity_per_band'][1],
                "pulse_clarity_middle_mean": features_manual['mean_clarity_per_band'][2],
                "pulse_clarity_middle_high_mean": features_manual['mean_clarity_per_band'][3],
                "pulse_clarity_high_mean": features_manual['mean_clarity_per_band'][4],
                "attack_slope_mean": features_manual['mean_attack_slopes'],
                "attack_slope_low_mean": features_manual['mean_attack_slopes_per_band'][0],
                "attack_slope_middle_low_mean": features_manual['mean_attack_slopes_per_band'][1],
                "attack_slope_middle_mean": features_manual['mean_attack_slopes_per_band'][2],
                "attack_slope_middle_high_mean": features_manual['mean_attack_slopes_per_band'][3],
                "attack_slope_high_mean": features_manual['mean_attack_slopes_per_band'][4],
                "spectral_flatness_mean": features_manual['mean_flatness'],
                "spectral_flatness_low_mean": features_manual['mean_flatness_per_band'][0],
                "spectral_flatness_middle_low_mean": features_manual['mean_flatness_per_band'][1],
                "spectral_flatness_middle_mean": features_manual['mean_flatness_per_band'][2],
                "spectral_flatness_middle_high_mean": features_manual['mean_flatness_per_band'][3],
                "spectral_flatness_high_mean": features_manual['mean_flatness_per_band'][4],
                "entropia_clarity": features_manual['entropia_clarity'],
                "entropia_clarity_low": features_manual['entropia_clarity_per_band'][0],
                "entropia_clarity_middle_low": features_manual['entropia_clarity_per_band'][1],
                "entropia_clarity_middle": features_manual['entropia_clarity_per_band'][2],
                "entropia_clarity_middle_high": features_manual['entropia_clarity_per_band'][3],
                "entropia_clarity_high": features_manual['entropia_clarity_per_band'][4],
                "attack_time": features_manual['mean_attack_times'],
                "attack_time_low_mean": features_manual['mean_attack_times_per_band'][0],
                "attack_time_middle_low_mean": features_manual['mean_attack_times_per_band'][1],
                "attack_time_middle_mean": features_manual['mean_attack_times_per_band'][2],
                "attack_time_middle_high_mean": features_manual['mean_attack_times_per_band'][3],
                "attack_time_high_mean": features_manual['mean_attack_times_per_band'][4],
                "spectral_flux_mean": features_manual['mean_spectral_flux'],
                "spectral_flux_low_mean": features_manual['mean_spectral_flux_per_band'][0],
                "spectral_flux_middle_low_mean": features_manual['mean_spectral_flux_per_band'][1],
                "spectral_flux_middle_mean": features_manual['mean_spectral_flux_per_band'][2],
                "spectral_flux_middle_high_mean": features_manual['mean_spectral_flux_per_band'][3],
                "spectral_flux_high_mean": features_manual['mean_spectral_flux_per_band'][4]
            }

            # Combine the data
            combined_data = {**extracted_data, **manual_extracted_data}

            # Save the combined data back to the JSON file using NumpyEncoder
            with open(output_path, 'w') as f:
                json.dump(combined_data, f, indent=4, cls=NumpyEncoder)

            print(f"Updated JSON file saved to {output_path}")

    print("All files processed.")
