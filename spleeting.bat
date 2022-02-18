@echo off

echo %5

if exist %5\ (
  echo y | del %5
  echo spleeted folder exists. Deleted...
) else (
  echo spleeted audio folder yet to be created
)

cmd /c "cd %1 & spleeter separate -p spleeter:2stems -o output_dir %2 & del %3 & cd output_dir & cd %4 & del accompaniment.wav"
