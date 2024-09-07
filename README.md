# Generate-Emotional-Music(just individual Emotional-Music generation)

1. For training the music emotion classifier, please first run “lyrics_datasets_v3/splitdata.py”, then run “LSTM_cls.py” and “Transformer_cls.py”. 

2. For training the lyric and melody generator, please run “CNN_GRU_generator.py”,“GRU_generator.py” ,“CNN_Transformer_generator.py” and “Transformer_generator.py”. 

3. To generate music segments, please run “GRU_EBS.py” or "GRU_DRM" or “Transformer_EBS.py”. 
 command like "python GRU_EBS.py --emotion positive" or "python GRU_EBS.py --emotion nagative" 
     ## Configuration information, for example, GRU_EBS
     for generator_file ,Use GRU or CNN_GRU pkl files, preferably using absolute paths
# System use
1. run GUI.py 
## before run GUI.py  Correct path information needs to be configured
2. Configure the path of emotion recognition, the first is the environment of the emotion recognition project, the middle is the code path, and the last is the cwd instruction 
process = subprocess.Popen(
                ["D:\\emotion-recognition-using-speech-master\\venv\\Scripts\\python.exe",
                 "D:\\emotion-recognition-using-speech-master\\emotion-recognition-using-speech-master\\test.py"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd="D:\\emotion-recognition-using-speech-master\\emotion-recognition-using-speech-master"
            )
preferably using absolute paths
3. Configure the path of the EBS file, with the environment to generate the music project in the front, the code path in the middle, and the cwd instruction at the end
ebs_command = [
                "python",
                "D:\\Generate-Emotional-Music-main\\Generate-Emotional-Music-main\\GRU_EBS.py",
                "--emotion",
                mapped_emotion
            ]
preferably using absolute paths
also can use other py,like GRU_DRM,CNN_GRU_EBS
4. The GUI needs to display sheet music with lyrics, so the path to MuseScore needs to be configured in yinyuepu.py to generate music score files
   environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')
5. Running the GUI for the first time can take longer because it involves the startup of the project
   
## (See the README in the project for how individual emotion recognition works. It's in another branch)
