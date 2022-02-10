import os
from joblib import load, dump
import subprocess


ffmpeg_input = r"E:\Projects\Freelancing\Opera audio files\opera_inferencing/Data/serving_data/output_dir/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M.wav"
ffmpeg_output = r"E:\Projects\Freelancing\Opera audio files\opera_inferencing/Data/serving_data/output_dir/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M/lol.wav"

# print(ffmpeg_input)
ffmpeg_dir = r"E:\Projects\Freelancing\Opera audio files\opera_inferencing/Data/serving_data/output_dir/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M"

subprocess.call([r'silence_remove.bat', ffmpeg_input, ffmpeg_output, ffmpeg_dir])
# & del %1 & ren %2 %1 & cd .. & cd .. & cd .. & cd .
# , os.path.basename(ffmpeg_input), os.path.basename(ffmpeg_output)]
# audio_file = 'testing_audio/Kerstin Thorborg/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M.wav'


# audio_file_name = r"{}".format(os.path.basename(audio_file))

# print(audio_file_name)
# lol = "Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M.wav"
# print(lol.rsplit('.',1)[0])

# ffmpeg_input = self.spleeting_dir + '/output_dir/' + self.audio_dirname + self.audio_file_name
# final_audio = separated_components_dir + '/' + item
# ffmpeg_output = separated_components_dir + '/'  + item.rsplit('.')[0] + '_silence_removed.wav'