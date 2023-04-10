# CHECK WORD EMBEDDINGS FOR INCREMENTAL TRAIINNG>>>>>

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import han_config
from utils import load_torch_model, downsampling, clean_text_han, preprocess_han, mat_mul, ele_wise_mul, calc_acc

class SentimentDataset(Dataset):
    def __init__(self, vectors, labels):
        super().__init__()
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.vectors[index], self.labels[index]
    
class WordAttNet(nn.Module):
    def __init__(self, embeddings, hidden_size=50):
        super().__init__()
        pad_unk_word = np.zeros((1, embeddings.shape[1]))
        embeddings = torch.from_numpy(np.concatenate([pad_unk_word, embeddings], axis=0).astype(
            np.float32))  # try with float16 - mixedprecision and check metrics ##################

        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size).zero_())
        self.word_weight = nn.Parameter(torch.empty(
            2 * hidden_size, 2 * hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(
            torch.empty(2 * hidden_size, 1).normal_(0.0, 0.05))

        self.lookup = nn.Embedding.from_pretrained(embeddings)
        self.gru = nn.GRU(embeddings.shape[1], hidden_size, bidirectional=True)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output, hidden_state)

        output = mat_mul(f_output, self.word_weight, self.word_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = ele_wise_mul(f_output, output.permute(1, 0))

        return output, h_output
    
class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2):
        super().__init__()

        self.sent_bias = nn.Parameter(
            torch.empty(1, 2 * sent_hidden_size).zero_())
        self.sent_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 2 * sent_hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 1).normal_(0.0, 0.05))

        self.gru = nn.GRU(2 * word_hidden_size,
                          sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = mat_mul(f_output, self.sent_weight, self.sent_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = ele_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output
    
class HAN(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes,
                 embeddings, max_sent_length, max_word_length):
        super().__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(
            embeddings, word_hidden_size)
        self.sent_att_net = SentAttNet(
            sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        batch_size = self.batch_size

        if last_batch_size:
            batch_size = last_batch_size

        self.word_hidden_state = torch.zeros(
            2, batch_size, self.word_hidden_size, device='cuda')
        self.sent_hidden_state = torch.zeros(
            2, batch_size, self.sent_hidden_size, device='cuda')

    def forward(self, input_v):

        output_list = []
        input_v = input_v.permute(1, 0, 2)
        for i in input_v:
            output, self.word_hidden_state = self.word_att_net(
                i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(
            output, self.sent_hidden_state)

        return output

class Model:
    def __init__(self, docs, y, wb_dict, embeddings, init, test_size=0.2, batch_size=128):

        # y, x = downsampling(y, x)

        max_word_length = 15
        max_sent_length = 10

        docs = clean_text_han(docs)

        embeddings = np.asarray(embeddings)

        # if init:
        #     max_word_length, max_sent_length = get_max_lengths(
        #         doc)  # change to train only

        docs = preprocess_han(docs, wb_dict,
                        max_word_length, max_sent_length)

        x_train, x_test, y_train, y_test = train_test_split(
            docs, y, test_size=0.2, random_state=42)

        training_set = SentimentDataset(x_train, y_train)

        self.training_generator = DataLoader(
            training_set, batch_size=han_config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_set = SentimentDataset(x_test, y_test)
        self.test_generator = DataLoader(
            test_set, batch_size=han_config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

        if init:
            self.model = HAN(han_config.WORD_HIDDEN_SIZE, han_config.SENT_HIDDEN_SIZE, han_config.BATCH_SIZE, han_config.N_CLASS,
                        embeddings, max_sent_length, max_word_length).cuda()
        else:
            self.model = load_torch_model('ssl-clf.pth').cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters(
        )), lr=han_config.LR)  # , momentum=han_config.MOMENTUM)
        
        self.best_model = None

    def fit(self, epoch):
        best_epoch = 0
        best_loss = 1e5

        for epoch in range(han_config.EPOCHS):
            # start_time = time.time()
            self.model.train()
            train_loss = 0
            train_acc = 0

            for vec, label in self.training_generator:
                vec = vec.cuda()
                label = label.cuda()
                self.optimizer.zero_grad()
                self.model._init_hidden_state()
                pred = self.model(vec)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss
                train_acc += calc_acc(label, pred)
            train_loss /= len(self.training_generator)
            train_acc /= len(self.training_generator)

            self.model.eval()
            val_loss = 0
            val_acc = 0
            for vec, label in self.test_generator:
                num_sample = len(label)
                vec = vec.cuda()
                label = label.cuda()
                with torch.no_grad():
                    self.model._init_hidden_state(num_sample)
                    pred = self.model(vec)
                val_loss += self.criterion(pred, label)
                val_acc += calc_acc(label, pred)

            val_loss /= len(self.test_generator)
            val_loss = val_loss.item()
            val_acc /= len(self.test_generator)

            print(f"epoch: {epoch+1}, training loss: {train_loss:.4f}, training acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                print(f"Best loss {val_loss}")
                self.best_model = self.model

            if epoch - best_epoch > han_config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Stop training at epoch {epoch+1}. The lowest loss achieved is {best_loss}")
                print(best_epoch)
                break

    def fit_and_save(self, filename, epoch=500):
        self.fit(epoch=epoch)
        self.best_model.eval()
        torch.save(self.best_model, filename)