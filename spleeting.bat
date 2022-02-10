@echo on
cmd /c "cd %1 & spleeter separate -p spleeter:2stems -o output_dir %2 & del %3 & cd output_dir & cd %4 & del accompaniment.wav & ren vocals.wav %3"
