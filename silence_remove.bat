@echo on
cmd /c "cd %3 & ffmpeg -i %1 -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-60dB %2"

