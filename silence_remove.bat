@echo on

cmd /c "cd %1 & ffmpeg -i vocals.wav -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-60dB silence_removed.wav & del vocals.wav"

