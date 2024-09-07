import numpy as np
from scipy.spatial.distance import jensenshannon


def load_data_from_txt(file_path):
    """ 从文件中加载数据并返回 pitch, dur, rest 列表 """
    pitch = []
    dur = []
    rest = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 检查非空行
                parts = line.split()
                if len(parts) == 3:  # 确保行中有三列数据
                    pitch.append(float(parts[0]))
                    dur.append(float(parts[1]))
                    rest.append(float(parts[2]))

    return np.array(pitch), np.array(dur), np.array(rest)


def calculate_jsd(p_real, p_generated):
    """ 计算 Jensen-Shannon 散度 """
    return jensenshannon(p_real, p_generated) ** 2


def get_distribution(data, bins):
    """ 将数据转化为概率分布，使用直方图归一化处理 """
    hist, _ = np.histogram(data, bins=bins, density=True)
    return hist / hist.sum()


# 加载真实数据和生成数据
real_pitch, real_dur, real_rest = load_data_from_txt('D:\Generate-Emotional-Music-main\Generate-Emotional-Music-main\saved_model\music_data.txt')
gen_pitch, gen_dur, gen_rest = load_data_from_txt('zhengli_cnnTrans_generated_melodies.txt')

# 定义 bins 范围
bins_pitch = np.linspace(0, 100, 50)  # 针对 pitch
bins_dur = np.linspace(0, 2, 50)  # 针对 duration
bins_rest = np.linspace(0, 2, 50)  # 针对 rest

# 计算每个维度的概率分布
pitch_real_dist = get_distribution(real_pitch, bins_pitch)
pitch_gen_dist = get_distribution(gen_pitch, bins_pitch)

dur_real_dist = get_distribution(real_dur, bins_dur)
dur_gen_dist = get_distribution(gen_dur, bins_dur)

rest_real_dist = get_distribution(real_rest, bins_rest)
rest_gen_dist = get_distribution(gen_rest, bins_rest)

# 计算 Jensen-Shannon 散度
jsd_pitch = calculate_jsd(pitch_real_dist, pitch_gen_dist)
jsd_dur = calculate_jsd(dur_real_dist, dur_gen_dist)
jsd_rest = calculate_jsd(rest_real_dist, rest_gen_dist)

# 输出Jensen-Shannon散度
output_filename = 'cnnTrans_jsd_results.txt'
with open(output_filename, 'w') as output_file:
    output_file.write(f"JSD for Pitch: {jsd_pitch}\n")
    output_file.write(f"JSD for Duration: {jsd_dur}\n")
    output_file.write(f"JSD for Rest: {jsd_rest}\n")

print(f"JSD results saved to {output_filename}")