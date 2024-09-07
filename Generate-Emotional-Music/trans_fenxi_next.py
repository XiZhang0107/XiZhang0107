import numpy as np

# 确保加载时设置 allow_pickle=True
generated_melodies = np.load('D:\Generate-Emotional-Music-main\Generate-Emotional-Music-main\cnnTransformer_generated_melodies.npy',
                             allow_pickle=True)

# 加载 music_vocabulary 字典
music_vocabulary_file = 'D:\\Generate-Emotional-Music-main\\Generate-Emotional-Music-main\\saved_model\\music_vocabulary_50.npy'
music_vocabulary = np.load(music_vocabulary_file, allow_pickle=True).item()

# 反转字典，将索引映射回音符信息
reverse_music_vocabulary = {v: k for k, v in music_vocabulary.items()}


# 解析音符序列
def parse_sequence(sequence, reverse_music_vocabulary):
    pitch = []
    dur = []
    rest = []

    for note_array in sequence:
        # 假设 note_array 是 logits 或 softmax 概率分布
        note = np.argmax(note_array)  # 获取最大概率对应的索引

        note_str = reverse_music_vocabulary[note]  # 获取对应的音符字符串
        note_parts = note_str.split('^')

        p = float(note_parts[0].split('_')[1])
        d = float(note_parts[1].split('_')[1])
        r = float(note_parts[2].split('_')[1])

        pitch.append(p)
        dur.append(d)
        rest.append(r)

    return pitch, dur, rest


# 解析所有生成的旋律
parsed_melodies = [parse_sequence(seq, reverse_music_vocabulary) for seq in generated_melodies]

# 将解析后的数据保存到txt文件
with open('cnnTrans_generated_melodies.txt', 'w') as f:
    for i, (pitch, dur, rest) in enumerate(parsed_melodies):
        f.write(f"Sequence {i + 1}:\n")
        f.write("Pitch: " + " ".join(map(str, pitch)) + "\n")
        f.write("Duration: " + " ".join(map(str, dur)) + "\n")
        f.write("Rest: " + " ".join(map(str, rest)) + "\n")
        f.write("\n")  # 分隔不同的序列

print("cnnTrans_generated_melodies.txt")