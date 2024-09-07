import ast
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
# 逐行解析并构建数据列表，尝试去除意外的缩进和空格
data_list = []
with open('melody.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            # 进一步清理字符串
            cleaned_line = line.replace('list(', '[').replace(')', ']').strip()
            # 尝试解析每一行
            parsed_line = ast.literal_eval(cleaned_line)
            data_list.append(parsed_line)
        except Exception as e:
            print(f"Error during parsing line: {line.strip()}\nError: {str(e)}")

# 检查解析后的数据是否成功
if not data_list:
    raise ValueError("Data parsing failed.")

# 定义处理嵌套结构并提取音高、时值和休止符的函数
def process_note_group(note_group):
    for note in note_group:
        if isinstance(note, list) and len(note) == 3:  # 确保note包含三个元素
            pitch, duration, rest = note
            if isinstance(pitch, (int, float)) and isinstance(duration, (int, float)) and isinstance(rest, (int, float)):
                all_pitches.append(pitch)
                all_durations.append(duration)
                all_rests.append(rest)
        elif isinstance(note, list):  # 如果是嵌套列表，则递归处理
            process_note_group(note)

# 处理整个数据集
all_pitches = []
all_durations = []
all_rests = []

for data in data_list:
    for note_group in data:
        process_note_group(note_group)

# 计算合并后的整体统计特征
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
    "Mean value of pitch": mean_pitch_value,
    "Standard deviation of pitch": std_pitch,
    "Number of unique pitch value": unique_pitches,
    "Max Pitch value": max_pitch_value,
    "Min Pitch value": min_pitch_value,
    "Mode of duration": mode_duration,
    "Number of unique duration value": unique_durations,
    "Max Duration value": max_duration,
    "Min Duration value": min_duration,
    "Percentage of 1.0(%) ": percentage_1_duration,
    "Mode of rest": mode_rest,
    "Number of unique rest value": unique_rests,
    "Max rest value": max_rest,
    "Percentage of 0.0(%)": percentage_0_rest,
}
with open('ground_distribution.txt', 'w') as output_file:
    for key, value in overall_results.items():
        output_file.write(f"{key}: {value}\n")
# 打印最终结果
for key, value in overall_results.items():
    print(f"{key}: {value}")
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(all_pitches, bins=50, color='blue', alpha=0.7)
plt.title('Frequency-pitch Histogram')
plt.xlabel('Pitch')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.hist(all_durations, bins=50, color='green', alpha=0.7)
plt.title('Frequency-duration Histogram')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.hist(all_rests, bins=50, color='red', alpha=0.7)
plt.title('Frequency-rest Histogram')
plt.xlabel('Rest')
plt.ylabel('Frequency')
plt.grid(True)

# 保存合并后的图像
plt.savefig('ground_distribution.png')
plt.show()