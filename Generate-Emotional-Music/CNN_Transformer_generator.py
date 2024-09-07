import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import Word2Vec
import argparse
from torch.nn import Transformer
import torch.optim as optim
import os


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class ParallelCNNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, kernel_size=3):
        super(ParallelCNNTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 卷积层：保持输入输出维度一致
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-Attention部分
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 卷积层：将输入从(batch_size, seq_len, d_model) 转换为 (batch_size, d_model, seq_len)
        src2 = src.transpose(1, 2)  # 转换为 (batch_size, d_model, seq_len)
        src2 = F.relu(self.conv(src2))  # 经过卷积层
        src2 = src2.transpose(1, 2)  # 转换回 (batch_size, seq_len, d_model)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # 前馈网络部分
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        return src


class ParallelCNNTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, kernel_size=3):
        super(ParallelCNNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ParallelCNNTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, kernel_size)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output


class ParallelCNNTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1,
                 kernel_size=3, batch_first=False):
        super(ParallelCNNTransformer, self).__init__()
        self.encoder = ParallelCNNTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                                                     kernel_size)
        self.decoder = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_decoder_layers,
                                   num_decoder_layers=num_decoder_layers, dropout=dropout, batch_first=batch_first)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        return memory, output


class Transformer_seq2seq(nn.Module):
    def __init__(self, transformer_model, dim, lyric_vocabulary, music_vocabulary):
        super().__init__()
        self.transformer_model = transformer_model
        self.dim = dim
        self.lyric_vocabulary = lyric_vocabulary
        self.music_vocabulary = music_vocabulary
        self.lyric_vocab = len(lyric_vocabulary)
        self.music_vocab = len(music_vocabulary)
        self.fc_lyric = nn.Linear(self.dim, self.lyric_vocab)
        self.fc_music = nn.Linear(self.dim, self.music_vocab)
        self.music_embedding = nn.Embedding(num_embeddings=self.music_vocab, embedding_dim=self.dim)

    def forward(self, src, tgt):
        tgt = self.music_embedding(tgt)  # torch.Size([64, 19, 256])
        en_hi, de_hi = self.transformer_model(src, tgt)  # torch.Size([64, 19, 256])
        en_output = self.fc_lyric(en_hi)  # torch.Size([64, 19, 20934])
        de_output = self.fc_music(de_hi)

        return en_output, de_output


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
        self.dataset = dataset
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

        word_input = torch.LongTensor(np.zeros(self.seqlen - 1, dtype=np.int64))
        syll_input = torch.LongTensor(np.zeros(self.seqlen - 1, dtype=np.int64))

        txt_len = 0
        for i in range(len(lyric)):
            word = ''
            for syll in lyric[i]:
                word += syll
            if word in self.wordModel.wv.index_to_key:
                word2Vec = self.wordModel.wv[word]
                word2idx = self.wordModel.wv.key_to_index[word]
            else:
                continue
            for j in range(len(lyric[i])):
                syll = lyric[i][j]
                note = 'p_' + str(music[i][j][0]) + '^' + 'd_' + str(music[i][j][1]) + '^' + 'r_' + str(music[i][j][2])
                note2idx = self.music_vocabulary[note]
                if syll in self.syllModel.wv.index_to_key:
                    syll2Vec = self.syllModel.wv[syll]
                    syll2idx = self.syllModel.wv.key_to_index[syll]
                else:
                    continue
                syllWordVec = np.concatenate((word2Vec, syll2Vec))
                if txt_len < self.seqlen - 1:
                    lyric_input[txt_len] = torch.from_numpy(syllWordVec)
                    word_input[txt_len] = word2idx
                    syll_input[txt_len] = syll2idx
                    music_input[txt_len] = note2idx
                if txt_len < self.seqlen and txt_len > 0:
                    lyric_label[txt_len - 1] = syll2idx
                    music_label[txt_len - 1] = note2idx
                txt_len += 1

            if txt_len >= self.seqlen:
                break
        return lyric_input.type(torch.float32), lyric_label.type(torch.int64), music_input.type(
            torch.int64), music_label.type(torch.int64)

    def __len__(self):
        return len(self.lyrics)


def split_dataset(dataset, test_size=0.1):
    combined = list(zip(dataset[0], dataset[1], dataset[2]))
    train_combined, test_combined = train_test_split(combined, test_size=test_size, random_state=42)
    train_data_0, train_data_1, train_data_2 = zip(*train_combined)
    test_data_0, test_data_1, test_data_2 = zip(*test_combined)
    train_data = [np.array(train_data_0, dtype=object), np.array(train_data_1, dtype=object),
                  np.array(train_data_2, dtype=object)]
    test_data = [np.array(test_data_0, dtype=object), np.array(test_data_1, dtype=object),
                 np.array(test_data_2, dtype=object)]
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lyrics_melody_generator.py')
    parser.add_argument('--data', type=str, default='lyrics_datasets_v3/dataset_50_v3.npy', help="Dnd data.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--seqlen', type=int, default=50, help="seqlen")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=int, default=60, help="num epochs")
    parser.add_argument('--lyc2vec', type=int, default=128, help="lyc2vec dimension")
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = np.load(opt.data, allow_pickle=True)

    syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_128.bin'
    word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_128.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)

    lyric_vocabulary = syllModel.wv.key_to_index
    music_vocabulary_file = 'saved_model/music_vocabulary_' + str(opt.seqlen) + '.npy'
    music_vocabulary = np.load(music_vocabulary_file, allow_pickle=True).item()

    # Split the dataset into training and testing sets
    train_data, test_data = split_dataset(dataset, test_size=0.1)

    dtrain_set = TxtDatasetProcessing(train_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    dtest_set = TxtDatasetProcessing(test_data, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)

    train_loader = DataLoader(dtrain_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(dtest_set, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=4)

    transformer_model = ParallelCNNTransformer(d_model=opt.lyc2vec * 2, nhead=16, num_encoder_layers=12,
                                               num_decoder_layers=12, batch_first=True)
    model = Transformer_seq2seq(transformer_model=transformer_model, dim=opt.lyc2vec * 2,
                                lyric_vocabulary=lyric_vocabulary, music_vocabulary=music_vocabulary)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    softmax = nn.Softmax(dim=0)
    print(f'Model on GPU: {next(model.parameters()).is_cuda}')

    with open("cnnbing_transform_loss.txt", "w") as log_file:
        for epoch in range(opt.num_epochs):
            model.train()
            optimizer = adjust_learning_rate(optimizer, epoch, opt.learning_rate)
            total_train_loss = 0

            for iter, traindata in enumerate(train_loader):
                lyric_input, lyric_label, music_input, music_label = traindata
                lyric_input = lyric_input.to(device)
                lyric_label = lyric_label.to(device)
                music_input = music_input.to(device)
                music_label = music_label.to(device)

                optimizer.zero_grad()

                en_pred, de_pred = model(lyric_input, music_input)

                en_loss = 0
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

                total_train_loss += loss.item()

                if iter % 100 == 0:
                    print({'epoch': epoch, 'batch': iter, 'loss': loss.item()})

            avg_train_loss = total_train_loss / len(train_loader)

            # Evaluation on the test set
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for iter, testdata in enumerate(test_loader):
                    lyric_input, lyric_label, music_input, music_label = testdata
                    lyric_input = lyric_input.to(device)
                    lyric_label = lyric_label.to(device)
                    music_input = music_input.to(device)
                    music_label = music_label.to(device)

                    en_pred, de_pred = model(lyric_input, music_input)

                    en_loss = 0
                    for batch in range(opt.batch_size):
                        en_pred_batch = en_pred[batch]
                        lyric_label_batch = lyric_label[batch]
                        for length in range(opt.seqlen - 1):
                            logits = en_pred_batch[length]
                            prob = softmax(logits)
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

                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)

            # Log the train and test losses
            log_file.write(
                f'Epoch {epoch + 1}/{opt.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}\n')
            print(
                f'Epoch {epoch + 1}/{opt.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

            # Save the model after each epoch
            filename = f'cnnbing_Transformer_generator_seqlen_{opt.seqlen}_embed_{opt.lyc2vec}_epoch_{epoch}.pkl'
            torch.save(model.state_dict(), filename)
            print(f'File {filename} is saved.')