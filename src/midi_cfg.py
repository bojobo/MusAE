beat_resolution = 24
time_sig_numerator = 4
# time_sig_denominator = 4
bar_size = beat_resolution * time_sig_numerator
bar_count = 2
phrase_size = bar_size * bar_count
n_cropped_notes = 130  # +2 because we include a silent note and a held note position
n_tracks = 2  # Piano and others
n_midi_pitches = 128  # constant for MIDI
n_midi_programs = 128  # constant for MIDI
