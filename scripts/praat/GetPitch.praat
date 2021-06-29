form GetFile
	text filename C:\Users\Tristan\nlp_project\mar2moon\scripts\test_podcast_00.00.20.080_00.00.28.160.wav
endform

Read from file... 'filename$'

To Pitch: 0.0, 75, 600
min_p = Get minimum: 0.0, 0.0, "Hertz", "parabolic"
max_p = Get maximum: 0.0, 0.0, "Hertz", "parabolic"
writeInfoLine: min_p
appendInfoLine: max_p
