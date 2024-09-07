import pandas as pd

def reformat_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    pitch, duration, rest = [], [], []
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("Pitch:"):
            current_section = "Pitch"
            values = line.replace("Pitch:", "").strip().split()
            pitch.extend(map(float, values))
        elif line.startswith("Duration:"):
            current_section = "Duration"
            values = line.replace("Duration:", "").strip().split()
            duration.extend(map(float, values))
        elif line.startswith("Rest:"):
            current_section = "Rest"
            values = line.replace("Rest:", "").strip().split()
            rest.extend(map(float, values))

    data = pd.DataFrame({
        'Pitch': pitch,
        'Duration': duration,
        'Rest': rest
    })

    # 将数据保存为新的格式
    data.to_csv('zhengli_cnnTrans_generated_melodies.txt', sep=' ', index=False, header=False)

    return data

# 假设你的TXT文件路径为 'sequences.txt'
file_path = 'cnnTrans_generated_melodies.txt'
data = reformat_data(file_path)

# 显示结果
print(data)