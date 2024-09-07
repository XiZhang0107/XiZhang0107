from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import pandas as pd
import pickle
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.autograd import Variable
from collections import Counter
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pdb
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocabulary, kernel_size=3, dropout=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary = vocabulary
        self.n_vocab = len(vocabulary)

        # GRU Layer
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, dropout=dropout)

        # CNN Layer
        self.cnn = nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_size,
                             kernel_size=kernel_size, padding=kernel_size // 2)  # Same padding

        # Fully Connected Layer to output vocabulary probabilities
        self.fc = nn.Linear(self.hidden_size * 2, self.n_vocab)  # *2 because of concatenation of GRU and CNN outputs
#此处输入特征层数变为了两层
    def forward(self, x, prev_state):
        # GRU forward
        states, hidden = self.gru(x, prev_state)  # x: (batch_size, seq_len, input_size)

        # CNN forward
        # Permute x to fit Conv1d input requirements: (batch_size, channels, seq_len)
        cnn_input = x.permute(0, 2, 1)
        cnn_output = F.relu(self.cnn(cnn_input))
        # Permute back to match GRU output dimensions: (batch_size, seq_len, output_channels)
        cnn_output = cnn_output.permute(0, 2, 1)

        # Combine GRU and CNN outputs
        combined_output = torch.cat([states, cnn_output], dim=2)  # Concatenate along the feature dimension

        # Fully connected layer
        logits = self.fc(combined_output)

        return logits, hidden, states
#states(seq_len, batch_size, hidden_size)
    def init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
    #这里面 self.hidden_size 乘以2的原因主要是 将解码器的当前隐藏状态与编码器的输出结合了，
        # 解码器的隐藏状态指的是hidden形状为[batch_size, hidden_size]，
        # 编码器的输出是encoder_outputs
        #通常形状为[seq_len, batch_size, hidden_size]（如果是双向的，可能是[seq_len, batch_size, hidden_size * 2]）。
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)#取得是encoder_outputs的seq_len
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)#赋值+位置变换，变为【batch_size, seq_len, hidden_size】
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]，位置变换
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        #dim2的意思 hidden 的形状假设为 [batch_size, seq_len, hidden_size]（假设已经进行了必要的重复和调整以匹配 encoder_outputs 的形状）。
#encoder_outputs 的形状为 [batch_size, seq_len, hidden_size]。
#这个拼接操作会将这两个张量在最后一个维度（特征维）上合并，结果是形状为 [batch_size, seq_len, 2*hidden_size] 的张量。
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, vocabulary):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary = vocabulary
        self.n_vocab = len(vocabulary)

        self.embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_dim)
        self.attention = Attention(self.hidden_size)

        self.gru = nn.GRU(input_size=self.embedding_dim + self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_size * 2, self.n_vocab)
        # self.hidden_size多的一个说是attention的

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input).unsqueeze(0)  # (1,B,N)
        # 形状为(1, batch_size, embedding_dim)
        # embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # (batch_size, 1, seq_len)
        # 将encoder_outpus从(seqlen, batch, hidden) 转为 (batch, seqlen, hidden)，然后与注意力权重相乘
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        # （1, batch_size, embedding_dim）与（1，batch，hidden）
        output, hidden = self.gru(rnn_input, last_hidden)
        # rnn_input: 这是GRU层的输入，
        # 它是当前词的嵌入向量和注意力机制生成的上下文向量的组合。
        # 其形状为(1, batch_size, embedding_dim + hidden_size)
        # output: 这是GRU层在当前时间步的输出，其形状通常是(1, batch_size,hidden_size)。
        # 这个输出包含了当前时间步解码器的状态，通常会进一步用于生成最终的词汇预测。
        # hidden: 这是GRU层更新后的隐藏状态，形状同last_hidden，即(num_layers, batch_size,hidden_size)。
        # 这个新的隐藏状态将被用于下一个时间步的计算。
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.fc(torch.cat([output, context], 1))
        return output, hidden, attn_weights

# encoder_outputs 对应于 states 变量，形状应该为seqlen, batch_size, hidden_size






class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, lyric_input, music_input, en_state_h):
        # en_state_h, en_state_c = self.encoder.init_state(self.seqlen)
        # en_pred: seqlen, batch size, vocab size
        # en_state_h: num layers, batch size, hidden size
        # states: seqlen, batch size, hidden size
        en_pred, en_state_h, en_states = self.encoder(lyric_input, en_state_h)
        hidden = en_state_h

        de_pred = Variable(torch.zeros(music_input.size(0), music_input.size(1), self.decoder.n_vocab)).cuda()
        for t in range(music_input.size(0)):
            inputw = Variable(music_input[t, :])
            output, hidden, attn_weights = self.decoder(inputw, hidden, en_states)
            de_pred[t] = output
        # de_pred = self.decoder(music_input, en_state_h)
        return en_pred, de_pred, en_state_h


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class TxtDatasetProcessing(Dataset):
    def __init__(self, dataset, syllModel, wordModel, seqlen, lyc2vec, music_vocabulary):
        self.dataset = dataset

        lyrics = list(dataset[1])
        musics = list(dataset[0])
        labels = list(dataset[2])

        index = []
        for i in range(len(labels)):
            if labels[i] == 'negative':
                index.append(i)

        negative_musics = []
        negative_lyrics = []
        negative_labels = []
        for i in index:
            negative_musics.append(musics[i])
            negative_lyrics.append(lyrics[i])
            negative_labels.append(labels[i])

        self.lyrics = lyrics + negative_lyrics
        self.musics = musics + negative_musics
        self.labels = labels + negative_labels

        self.syllModel = syllModel
        self.wordModel = wordModel
        self.seqlen = seqlen
        self.lyc2vec = lyc2vec
        self.music_vocabulary = music_vocabulary

    def __getitem__(self, index):
        lyric = self.lyrics[index]
        music = self.musics[index]

        lyric_input = torch.zeros((self.seqlen - 1, self.lyc2vec * 2), dtype=torch.float64)
        lyric_label = torch.LongTensor(np.zeros(self.seqlen - 1, dtype=np.int64))

        music_input = torch.LongTensor(np.zeros(self.seqlen - 1, dtype=np.int64))
        music_label = torch.LongTensor(np.zeros(self.seqlen - 1, dtype=np.int64))
        txt_len = 0
        for i in range(len(lyric)):
            word = ''
            for syll in lyric[i]:
                word += syll
            if word in self.wordModel.wv.index_to_key:  # ggggggggg
                word2Vec = self.wordModel.wv[word]  # ggggggggggg
            else:
                continue
            for j in range(len(lyric[i])):
                syll = lyric[i][j]
                note = 'p_' + str(music[i][j][0]) + '^' + 'd_' + str(music[i][j][1]) + '^' + 'r_' + str(music[i][j][2])
                note2idx = self.music_vocabulary[note]
                if syll in self.syllModel.wv.index_to_key:  # gggggg
                    syll2Vec = self.syllModel.wv[syll]
                    syll2idx = self.syllModel.wv.key_to_index[syll]
                else:
                    continue
                syllWordVec = np.concatenate((word2Vec, syll2Vec))
                if txt_len < self.seqlen - 1:
                    lyric_input[txt_len] = torch.from_numpy(syllWordVec)
                    music_input[txt_len] = note2idx
                if txt_len < self.seqlen and txt_len > 0:
                    lyric_label[txt_len - 1] = syll2idx
                    music_label[txt_len - 1] = note2idx
                txt_len += 1

            if txt_len >= self.seqlen:
                break
            if txt_len >= self.seqlen:
                break
        return lyric_input.type(torch.float32), lyric_label.type(torch.int64), music_input.type(
            torch.int64), music_label.type(torch.int64)

    def __len__(self):
        return len(self.lyrics)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def split_dataset(dataset, test_size=0.1):
    combined = list(zip(dataset[0], dataset[1], dataset[2]))
    train_combined, test_combined = train_test_split(combined, test_size=test_size, random_state=42)
    train_data_0, train_data_1, train_data_2 = zip(*train_combined)
    test_data_0, test_data_1, test_data_2 = zip(*test_combined)
    train_data = [np.array(train_data_0, dtype=object), np.array(train_data_1, dtype=object), np.array(train_data_2, dtype=object)]
    test_data = [np.array(test_data_0, dtype=object), np.array(test_data_1, dtype=object), np.array(test_data_2, dtype=object)]
    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lyrics_melody_generator.py')
    parser.add_argument('--data', type=str, default='lyrics_datasets_v3/dataset_50_v3.npy', help="Dnd data.")
    parser.add_argument('--batch_size', type=str, default=32, help="batch size")
    parser.add_argument('--seqlen', type=str, default=50, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=60, help="num pochs")
    parser.add_argument('--lyc2vec', type=str, default=128, help="num pochs")
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load(opt.data)

    syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_128.bin'
    word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_128.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)  # gggggggg

    lyric_vocabulary = syllModel.wv.key_to_index
    # music_vocabulary = build_vocabulary(dataset) #len=6064
    music_vocabulary_file = 'saved_model/music_vocabulary_' + str(opt.seqlen) + '.npy'
    music_vocabulary = np.load(music_vocabulary_file)
    music_vocabulary = music_vocabulary.item()

    # 数据集划分
    train_data, test_data = split_dataset(dataset, test_size=0.1)

    # 创建训练集和测试集的 DataLoader
    train_dataset = TxtDatasetProcessing(train_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    test_dataset = TxtDatasetProcessing(test_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=4)

    encoder = Encoder(input_size=opt.lyc2vec * 2, hidden_size=256, num_layers=4, vocabulary=lyric_vocabulary).to(device)
    decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    softmax = nn.Softmax(dim=0)

    for epoch in range(opt.num_epochs):
        model.train()
        en_state_h = encoder.init_state(opt.batch_size)
        en_state_h = Variable(en_state_h.to(device))
        optimizer = adjust_learning_rate(optimizer, epoch, opt.learning_rate)

        epoch_loss = 0.0
        num_batches = 0

        for iter, traindata in enumerate(train_loader):
            lyric_input, lyric_label, music_input, music_label = traindata
            lyric_input = Variable(lyric_input.transpose(0, 1).to(device))
            lyric_label = Variable(lyric_label.transpose(0, 1).to(device))
            music_input = Variable(music_input.transpose(0, 1).to(device))
            music_label = Variable(music_label.transpose(0, 1).to(device))

            optimizer.zero_grad()

            en_pred, de_pred, en_state_h = model(lyric_input, music_input, en_state_h)

            # Unlikelihood loss
            en_loss = 0
            en_pred = en_pred.transpose(0, 1)
            lyric_label = lyric_label.transpose(0, 1)
            for batch in range(opt.batch_size):
                en_pred_batch = en_pred[batch]
                lyric_label_batch = lyric_label[batch]
                for length in range(opt.seqlen - 1):
                    logits = en_pred_batch[length]
                    prob = softmax(logits)
                    with torch.no_grad():
                        label = lyric_label_batch[length]
                        negative_samples = list(set(lyric_label_batch[:length].tolist()))
                    likelihood_loss = -1 * torch.log(prob[label])
                    unlikelihood_loss = 0
                    if negative_samples:
                        unlikelihood_loss = (-1 * torch.log(1 - prob[negative_samples])).mean()
                    en_loss += (likelihood_loss + unlikelihood_loss)
            en_loss /= (opt.batch_size * (opt.seqlen - 1))

            de_dim = de_pred.shape[-1]
            de_pred = de_pred.view(-1, de_dim)
            music_label = music_label.reshape(-1)
            de_loss = criterion(de_pred, music_label)

            loss = en_loss + de_loss

            loss.backward()
            optimizer.step()

            en_state_h = en_state_h.detach()

            epoch_loss += loss.item()
            num_batches += 1

            if iter % 100 == 0:
                print({'epoch': epoch, 'batch': iter, 'loss': loss.item()})

        average_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')

        # 记录到txt文件
        with open("epoch_loss_log.txt", "a") as loss_log_file:
            loss_log_file.write(f'Epoch {epoch + 1}/{opt.num_epochs}, Average Loss: {average_loss}\n')

        filename = f'GRU_generator_seqlen_{opt.seqlen}_embed_{opt.lyc2vec}_epoch_{epoch}.pkl'
        torch.save(model.state_dict(), filename)
        print(f'File {filename} is saved.')