import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from gensim.models import Word2Vec  # 用于加载 syllModel 和 wordModel
from CNN_Transformer_generator import Transformer_seq2seq, ParallelCNNTransformer  # 引用Transformer模型定义
from CNN_Transformer_generator import TxtDatasetProcessing, split_dataset  # 引用必要的类和函数

# 命令行参数解析
parser = argparse.ArgumentParser(description='lyrics_melody_generator.py')
parser.add_argument('--data', type=str, default='lyrics_datasets_v3/dataset_50_v3.npy', help="Dnd data.")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--seqlen', type=int, default=50, help="seqlen")
parser.add_argument('--lyc2vec', type=int, default=128, help="num epochs")
opt = parser.parse_args()

# 重新加载 syllModel, wordModel 和 music_vocabulary
syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_128.bin'
word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_128.bin'
syllModel = Word2Vec.load(syll_model_path)
wordModel = Word2Vec.load(word_model_path)

music_vocabulary_file = 'saved_model/music_vocabulary_' + str(opt.seqlen) + '.npy'
music_vocabulary = np.load(music_vocabulary_file, allow_pickle=True).item()

# 重新加载数据集并进行划分
def load_data():
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    dataset = np.load(opt.data)

    # 划分数据集
    train_data, test_data = split_dataset(dataset, test_size=0.1)

    # 创建 DataLoader
    train_dataset = TxtDatasetProcessing(train_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    test_dataset = TxtDatasetProcessing(test_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=0)

    return train_loader, test_loader

# 主程序入口
if __name__ == "__main__":
    # 加载数据集
    train_loader, test_loader = load_data()

    # 初始化模型参数
    input_dim = opt.lyc2vec * 2
    nhead = 16
    num_encoder_layers = 12
    num_decoder_layers = 12
    dim_feedforward = 512

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer_model = ParallelCNNTransformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
    model = Transformer_seq2seq(transformer_model=transformer_model, dim=input_dim, lyric_vocabulary=syllModel.wv.key_to_index, music_vocabulary=music_vocabulary)
    model = model.to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load('D:\Generate-Emotional-Music-main\Generate-Emotional-Music-main\cnnbing_Transformer_generator_seqlen_50_embed_128_epoch_59.pkl'))
    model.to(device)
    model.eval()  # 设置模型为评估模式

    # 生成旋律
    generated_melodies = []
    ground_truth = []

    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for batch in test_loader:
            lyric_input, _, music_input, _ = batch
            lyric_input = lyric_input.to(device)
            music_input = music_input.to(device)

            # 生成旋律
            _, de_pred = model(lyric_input, music_input)
            generated_melodies.append(de_pred.cpu().numpy())  # 转回CPU并转换为numpy数组
            ground_truth.append(music_input.cpu().numpy())

    # 转换为numpy数组
    generated_melodies = np.concatenate(generated_melodies)
    ground_truth = np.concatenate(ground_truth)

    # 评估模型 (如果需要)
    # mse = mean_squared_error(ground_truth, generated_melodies)
    # print(f'Generated Melodies MSE: {mse}')

    # 保存生成的旋律
    np.save('cnnTransformer_generated_melodies.npy', generated_melodies)