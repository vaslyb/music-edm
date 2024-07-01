import pandas as pd
import os
import json
import numpy as np
# Ensure the output directory exists
output_dir = './results/features/'
os.makedirs(output_dir, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

for root, dirs, files in os.walk('./results/features_auto/'):
    if root == './results/features_auto/':
        continue
    root_final = root.replace('features_auto','features_simple')
    root_manual = root.replace('features_auto','features_manual')
    os.makedirs(root_final, exist_ok=True)
    for file in files:
        if file.endswith('.json'):
            filepath = os.path.join(root, file)
            df = pd.read_json(filepath)
            
            # Extract features from the original file
            features = {
                "loudness": df['lowlevel'][0],
                "dissonance_mean": df['lowlevel'][6]['mean'],
                # "dissonance_var": df['lowlevel'][6]['var'],
                "dynamic_complexity": df['lowlevel'][7],
                "pitch_salience_mean": df['lowlevel'][18]['mean'],
                # "pitch_salience_var": df['lowlevel'][18]['var'],
                "spectral_centroid_mean": df['lowlevel'][22]['mean'],
                # "spectral_centroid_var": df['lowlevel'][22]['var'],
                "spectral_complexity_mean": df['lowlevel'][23]['mean'],
                # "spectral_complexity_var": df['lowlevel'][23]['var'],
                "spectral_decrease_mean": df['lowlevel'][24]['mean'],
                # "spectral_decrease_var": df['lowlevel'][24]['var'],
                "spectral_energy_mean": df['lowlevel'][25]['mean'],
                # "spectral_energy_var": df['lowlevel'][25]['var'],
                "spectral_energyband_high_mean": df['lowlevel'][26]['mean'],
                # "spectral_energyband_high_var": df['lowlevel'][26]['var'],
                "spectral_energyband_low_mean": df['lowlevel'][27]['mean'],
                # "spectral_energyband_low_var": df['lowlevel'][27]['var'],
                "spectral_energyband_middle_mean": df['lowlevel'][28]['mean'],
                # "spectral_energyband_middle_var": df['lowlevel'][28]['var'],
                "spectral_energyband_middle_high_mean": df['lowlevel'][29]['mean'],
                # "spectral_energyband_middle_high_var": df['lowlevel'][29]['var'],
                "spectral_energyband_middle_low_mean": df['lowlevel'][30]['mean'],
                # "spectral_energyband_middle_low_var": df['lowlevel'][30]['var'],
                "spectral_entropy_mean": df['lowlevel'][31]['mean'],
                # "spectral_entropy_var": df['lowlevel'][31]['var'],
                # "spectral_flux_mean": df['lowlevel'][32]['mean'],
                # "spectral_flux_var": df['lowlevel'][32]['var'],
                "spectral_kurtosis_mean": df['lowlevel'][33]['mean'],
                # "spectral_kurtosis_var": df['lowlevel'][33]['var'],
                "spectral_rms_mean": df['lowlevel'][34]['mean'],
                # "spectral_rms_var": df['lowlevel'][34]['var'],
                "spectral_rolloff_mean": df['lowlevel'][35]['mean'],
                # "spectral_rolloff_var": df['lowlevel'][35]['var'],
                "spectral_skewness_mean": df['lowlevel'][36]['mean'],
                # "spectral_skewness_var": df['lowlevel'][36]['var'],
                "spectral_spread_mean": df['lowlevel'][37]['mean'],
                # "spectral_spread_var": df['lowlevel'][37]['var'],
                "zerocr_mean": df['lowlevel'][39]['mean'],
                # "zerocr_var": df['lowlevel'][39]['var'],
                "mfcc1_mean": df['lowlevel'][44]['mean'][0],
                "mfcc2_mean": df['lowlevel'][44]['mean'][1],
                "mfcc3_mean": df['lowlevel'][44]['mean'][2],
                "mfcc4_mean": df['lowlevel'][44]['mean'][3],
                "mfcc5_mean": df['lowlevel'][44]['mean'][4],
                "mfcc6_mean": df['lowlevel'][44]['mean'][5],
                "mfcc7_mean": df['lowlevel'][44]['mean'][6],
                "mfcc8_mean": df['lowlevel'][44]['mean'][7],
                "mfcc9_mean": df['lowlevel'][44]['mean'][8],
                "mfcc10_mean": df['lowlevel'][44]['mean'][9],
                "mfcc11_mean": df['lowlevel'][44]['mean'][10],
                "mfcc12_mean": df['lowlevel'][44]['mean'][11],
                "mfcc13_mean": df['lowlevel'][44]['mean'][12],
                "chroma1_mean": df['tonal'][72]['mean'][0],
                # "chroma1_var": df['tonal'][72]['var'][0],
                "chroma2_mean": df['tonal'][72]['mean'][1],
                # "chroma2_var": df['tonal'][72]['var'][1],
                "chroma3_mean": df['tonal'][72]['mean'][2],
                # "chroma3_var": df['tonal'][72]['var'][2],
                "chroma4_mean": df['tonal'][72]['mean'][3],
                # "chroma4_var": df['tonal'][72]['var'][3],
                "chroma5_mean": df['tonal'][72]['mean'][4],
                # "chroma5_var": df['tonal'][72]['var'][4],
                "chroma6_mean": df['tonal'][72]['mean'][5],
                # "chroma6_var": df['tonal'][72]['var'][5],
                "chroma7_mean": df['tonal'][72]['mean'][6],
                # "chroma7_var": df['tonal'][72]['var'][6],
                "chroma8_mean": df['tonal'][72]['mean'][7],
                # "chroma8_var": df['tonal'][72]['var'][7],
                "chroma9_mean": df['tonal'][72]['mean'][8],
                # "chroma9_var": df['tonal'][72]['var'][8],
                "chroma10_mean": df['tonal'][72]['mean'][9],
                # "chroma10_var": df['tonal'][72]['var'][9],
                "chroma11_mean": df['tonal'][72]['mean'][10],
                # "chroma11_var": df['tonal'][72]['var'][10],
                "chroma12_mean": df['tonal'][72]['mean'][11],
                # "chroma12_var": df['tonal'][72]['var'][11],            
                # "pulseclarity_mean": df['rhythm'][51]['mean'],
                # "pulseclarity_var": df['rhythm'][51]['var'],
                "bpm": df['rhythm'][52],
                "danceability": df['rhythm'][59],
                "onset_rate": df['rhythm'][60],
                "chords_changes_rate": df['tonal'][63],
                "chords_number_rate": df['tonal'][64],
                "chord": df['tonal'][75],
                "chord_strength": df['tonal'][66]['mean'],
                "chord_scale": df['tonal'][76],
                "key": df['tonal'][77],
                "key_scale": df['tonal'][78],
                "key_strength": df['tonal'][67]
            }
            
            # Find the corresponding file in the manual directory
            manual_filepath = os.path.join(root_manual, file)
            if os.path.exists(manual_filepath):
                df_manual = pd.read_json(manual_filepath)
                
                # Extract additional features from the manual file
                additional_features = {
                    "meter": df_manual['meter'][0],
                    "pulse_clarity_mean": df_manual['mean_clarity'][0],
                    # "pulse_clarity_var": df_manual['var_clarity'][0],
                    "pulse_clarity_low_mean": df_manual['mean_clarity_per_band'][0],
                    # "pulse_clarity_low_var": df_manual['var_clarity_per_band'][0],
                    "pulse_clarity_middle_low_mean": df_manual['mean_clarity_per_band'][1],
                    # "pulse_clarity_middle_low_var": df_manual['var_clarity_per_band'][1],
                    "pulse_clarity_middle_mean": df_manual['mean_clarity_per_band'][2],
                    # "pulse_clarity_middle_var": df_manual['var_clarity_per_band'][2],
                    "pulse_clarity_middle_high_mean": df_manual['mean_clarity_per_band'][3],
                    # "pulse_clarity_middle_high_var": df_manual['var_clarity_per_band'][3],
                    "pulse_clarity_high_mean": df_manual['mean_clarity_per_band'][4],
                    # "pulse_clarity_high_var": df_manual['var_clarity_per_band'][4],
                    "attack_slope_mean": df_manual['mean_attack_slopes'][0],
                    "attack_slope_low_mean": df_manual['mean_attack_slopes_per_band'][0],
                    "attack_slope_middle_low_mean": df_manual['mean_attack_slopes_per_band'][1],
                    "attack_slope_middle_mean": df_manual['mean_attack_slopes_per_band'][2],
                    "attack_slope_middle_high_mean": df_manual['mean_attack_slopes_per_band'][3],
                    "attack_slope_high_mean": df_manual['mean_attack_slopes_per_band'][4],
                    "spectral_flatness_mean": df_manual['mean_flatness'][0],
                    # "spectral_flatness_var": df_manual['var_flatness'][0],
                    "spectral_flatness_low_mean": df_manual['mean_flatness_per_band'][0],
                    # "spectral_flatness_low_var": df_manual['var_flatness_per_band'][0],
                    "spectral_flatness_middle_low_mean": df_manual['mean_flatness_per_band'][1],
                    # "spectral_flatness_middle_low_var": df_manual['var_flatness_per_band'][1],
                    "spectral_flatness_middle_mean": df_manual['mean_flatness_per_band'][2],
                    # "spectral_flatness_middle_var": df_manual['var_flatness_per_band'][2],
                    "spectral_flatness_middle_high_mean": df_manual['mean_flatness_per_band'][3],
                    # "spectral_flatness_middle_high_var": df_manual['var_flatness_per_band'][3],
                    "spectral_flatness_high_mean": df_manual['mean_flatness_per_band'][4],
                    # "spectral_flatness_high_var": df_manual['var_flatness_per_band'][4],
                    "entropia_clarity": df_manual['entropia_clarity'][0],
                    # "entropia_clarity_var": df_manual['entropia_clarity'][0],
                    "entropia_clarity_low": df_manual['entropia_clarity_per_band'][0],
                    # "entropia_clarity_low_var": df_manual['entropia_clarity_per_band'][0],
                    "entropia_clarity_middle_low": df_manual['entropia_clarity_per_band'][1],
                    # "entropia_clarity_middle_low_var": df_manual['entropia_clarity_per_band'][1],
                    "entropia_clarity_middle": df_manual['entropia_clarity_per_band'][2],
                    # "entropia_clarity_middle_var": df_manual['entropia_clarity_per_band'][2],
                    "entropia_clarity_middle_high": df_manual['entropia_clarity_per_band'][3],
                    # "entropia_clarity_middle_high_var": df_manual['entropia_clarity_per_band'][3],
                    "entropia_clarity_high": df_manual['entropia_clarity_per_band'][4],
                    # "entropia_clarity_high_var": df_manual['entropia_clarity_per_band'][4],
                    "attack_time": df_manual['mean_attack_times'][0],
                    #"var_attack_time": df_manual['var_attack_times'][0],
                    "attack_time_low_mean": df_manual['mean_attack_times_per_band'][0],
                    # "attack_time_low_var": df_manual['var_attack_times_per_band'][0],
                    "attack_time_middle_low_mean": df_manual['mean_attack_times_per_band'][1],
                    # "attack_time_middle_low_var": df_manual['var_attack_times_per_band'][1],
                    "attack_time_middle_mean": df_manual['mean_attack_times_per_band'][2],
                    # "attack_time_middle_var": df_manual['var_attack_times_per_band'][2],
                    "attack_time_middle_high_mean": df_manual['mean_attack_times_per_band'][3],
                    # "attack_time_middle_high_var": df_manual['var_attack_times_per_band'][3],
                    "attack_time_high_mean": df_manual['mean_attack_times_per_band'][4],
                    # "attack_time_high_var": df_manual['var_attack_times_per_band'][4],
                    "spectral_flux_mean": df_manual['mean_spectral_flux'][0],
                    # "spectral_flux_var": df_manual['var_spectral_flux'][0],
                    "spectral_flux_low_mean": df_manual['mean_spectral_flux_per_band'][0],
                    # "spectral_flux_low_var": df_manual['var_spectral_flux_per_band'][0],
                    "spectral_flux_middle_low_mean": df_manual['mean_spectral_flux_per_band'][1],
                    # "spectral_flux_middle_low_var": df_manual['var_spectral_flux_per_band'][1],
                    "spectral_flux_middle_mean": df_manual['mean_spectral_flux_per_band'][2],
                    # "spectral_flux_middle_var": df_manual['var_spectral_flux_per_band'][2],
                    "spectral_flux_middle_high_mean": df_manual['mean_spectral_flux_per_band'][3],
                    # "spectral_flux_middle_high_var": df_manual['var_spectral_flux_per_band'][3],
                    "spectral_flux_high_mean": df_manual['mean_spectral_flux_per_band'][4],
                    # "spectral_flux_high_var": df_manual['var_spectral_flux_per_band'][4]
                    
                }
                
                features.update(additional_features)
            output_filepath = os.path.join(root_final, file)
            with open(output_filepath, 'w') as outfile:
                json.dump(features, outfile, indent=4,cls = NpEncoder)
