import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
# 读取数据并解析为三列格式
data_list = []
with open('D:\Generate-Emotional-Music-main\Generate-Emotional-Music-main\zhengli_Trans_generated_melodies.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line:  # 确保行不为空
            try:
                # 将每一行的三个数值转换为浮点数列表
                pitch, duration, rest = map(float, line.split())
                data_list.append([pitch, duration, rest])
            except Exception as e:
                print(f"Error during parsing line: {line}\nError: {str(e)}")

# 检查解析后的数据是否成功
if not data_list:
    raise ValueError("Data parsing failed.")

# 转换为NumPy数组以便于后续处理
data_array = np.array(data_list)

# 分别提取 pitch, duration, rest 三列数据
all_pitches = data_array[:, 0]
all_durations = data_array[:, 1]
all_rests = data_array[:, 2]

# 计算统计特征
min_pitch_value = np.min(all_pitches)
max_pitch_value = np.max(all_pitches)
mean_pitch_value = np.mean(all_pitches)
std_pitch = np.std(all_pitches)
unique_pitches = len(np.unique(all_pitches))
mode_duration = mode(all_durations, axis=None).mode[0]
unique_durations = len(np.unique(all_durations))
max_duration = np.max(all_durations)
min_duration = np.min(all_durations)
percentage_1_duration = (np.sum(all_durations == 1.0) / len(all_durations)) * 100
mode_rest = mode(all_rests, axis=None).mode[0]
unique_rests = len(np.unique(all_rests))
max_rest = np.max(all_rests)
percentage_0_rest = (np.sum(all_rests == 0.0) / len(all_rests)) * 100

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

# 将结果写入文本文件
with open('Trans_results_distribution.txt', 'w') as output_file:
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
plt.savefig('Trans_distribution.png')
plt.show()