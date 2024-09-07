import numpy as np
from scipy.stats import mode

# 初始化存储所有音高、时值和休止符的列表
all_pitches = []
all_durations = []
all_rests = []

# 读取文件并解析数据
with open('GRU_generated_melodies.txt', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 4):  # 每个序列占4行，分别是序列名、音高、时值、休止符
        if "Pitch:" in lines[i + 1] and "Duration:" in lines[i + 2] and "Rest:" in lines[i + 3]:
            pitch_line = lines[i + 1].strip().replace('Pitch: ', '')
            duration_line = lines[i + 2].strip().replace('Duration: ', '')
            rest_line = lines[i + 3].strip().replace('Rest: ', '')

            # 将数据转换为浮点数列表
            pitches = list(map(float, pitch_line.split()))
            durations = list(map(float, duration_line.split()))
            rests = list(map(float, rest_line.split()))

            # 将数据添加到全局列表中
            all_pitches.extend(pitches)
            all_durations.extend(durations)
            all_rests.extend(rests)

# 计算统计信息
min_pitch_value = min(all_pitches)
max_pitch_value = max(all_pitches)
mean_pitch_value = np.mean(all_pitches)
std_pitch = np.std(all_pitches)
unique_pitches = len(set(all_pitches))
mode_duration = mode(all_durations, axis=None).mode[0]
unique_durations = len(set(all_durations))
max_duration = max(all_durations)
min_duration = min(all_durations)
percentage_1_duration = (all_durations.count(1.0) / len(all_durations)) * 100
mode_rest = mode(all_rests, axis=None).mode[0]
unique_rests = len(set(all_rests))
max_rest = max(all_rests)
percentage_0_rest = (all_rests.count(0.0) / len(all_rests)) * 100

# 汇总结果
overall_results = {
    "Mean Pitch": mean_pitch_value,
    "Std Pitch": std_pitch,
    "Unique Pitches": unique_pitches,
    "Max Pitch": max_pitch_value,
    "Min Pitch": min_pitch_value,
    "Mode Duration": mode_duration,
    "Unique Durations": unique_durations,
    "Max Duration": max_duration,
    "Min Duration": min_duration,
    "Percentage 1.0 Duration": percentage_1_duration,
    "Mode Rest": mode_rest,
    "Unique Rests": unique_rests,
    "Max Rest": max_rest,
    "Percentage 0.0 Rest": percentage_0_rest,
}

# 打印最终结果
with open('tongji_GRU_generated_melodies.txt', 'w') as output_file:
    for key, value in overall_results.items():
        output_file.write(f"{key}: {value}\n")

# 打印最终结果
for key, value in overall_results.items():
    print(f"{key}: {value}")