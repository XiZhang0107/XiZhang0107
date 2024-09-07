from PIL import Image
import subprocess
import numpy as np
from music21 import stream, note, meter, clef, environment
import os
# 设置 MuseScore 的路径
environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')

def parse_music_data(line):
    items = line.strip()[1:-1].split('), array(')
    items[0] = items[0].replace('array([', '').strip()
    items[-1] = items[-1].replace('])', '').strip()
    music_data = []
    for item in items:
        values = item.replace('[', '').replace(']', '').split(',')
        values = [float(v.strip()) for v in values]
        music_data.append(np.array(values))
    return music_data

# 从文件中读取数据
with open('output.txt', 'r') as file:
    lines = file.readlines()

# 解析歌词数据
lyrics = eval(lines[0].strip())

# 解析音符数据
music_data = parse_music_data(lines[1])

# 创建乐谱流
melody_stream = stream.Part()

# 添加节拍、高音谱号，并不设置调性
melody_stream.append(meter.TimeSignature('4/4'))
melody_stream.append(clef.TrebleClef())

# 为每个音符添加歌词
lyric_index = 0
for music_note in music_data:
    pitch, duration, rest = music_note
    if rest == 2.0:
        r = note.Rest(quarterLength=duration)
        melody_stream.append(r)
        n = note.Note(pitch, quarterLength=duration)
        n.pitch.accidental = None
        if lyric_index < len(lyrics):
            n.lyric = lyrics[lyric_index][0]
            lyric_index += 1
    else:
        n = note.Note(pitch, quarterLength=duration)
        n.pitch.accidental = None
        if lyric_index < len(lyrics):
            n.lyric = lyrics[lyric_index][0]
            lyric_index += 1
    melody_stream.append(n)

# 保存为 MusicXML 文件
melody_stream.write('musicxml', fp='0output_file.xml')

input_file = '0output_file.xml'
output_file_name = 'output.png'
if os.path.exists(output_file_name):
    os.remove(output_file_name)
musescore_path = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

# 构建命令行
command = [musescore_path, input_file, '-o', output_file_name]

# 执行命令
subprocess.run(command, check=True)

# 打开生成的图像文件
image = Image.open("output-1.png")

# 获取图像的宽度和高度
width, height = image.size

# 裁剪图像的有音符的部分
upper_part = int(height * 0.12)  # 裁剪顶部的百分比
lower_part = int(height * 0.40)  # 保留到80%的位置

# 执行裁剪操作
cropped_image = image.crop((0, upper_part, width, lower_part))

# 保存裁剪后的图像
cropped_image.save("cropped_output.png")