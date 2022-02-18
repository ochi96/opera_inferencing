@echo off

cmd /c "cd %1 & dir & ffmpeg -i silence_removed.wav -f segment -segment_time %2 -c copy %3 & del silence_removed.wav & ren *.* ????%4.wav"



