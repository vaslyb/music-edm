import essentia.standard as es

filename = './results/audio/dark_minimal_techno/0A9fWwK7HfFjUwKIIUbAu8.mp3'

audio = es.MonoLoader(filename=filename)()

# Predominant Pitch
predominator = es.PredominantPitchMelodia()
predominantpitches = predominator(audio)

# Pulse Clarity
beat = es.BeatTrackerDegara() 
beat_position = beat(audio)
beatloudness = es.BeatsLoudness(beats=beat_position)

# Meter
beatogram = es.Beatogram()
meter = es.Meter()

# Log Attack Time 
LogAttackTime = es.LogAttackTime()

## Per frequency

lowpass = es.LowPass(cutoffFrequency=2000)

bandpass = es.BandPass(bandwidth=500, cutoffFrequency=2000)

highpass = es.HighPass(cutoffFrequency=2000)

# Pulse Clarity
beat_position = beat(audio)
beatloudness = es.BeatsLoudness(beats=beat_position)
result = beatloudness(beat_position)
print(result)