# =========================
# 1. Імпорт бібліотек
# =========================
import math
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from datasets import load_dataset

import spacy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

nlp_es = spacy.load("es_core_news_sm")
nlp_pl = spacy.load("pl_core_news_sm")

def tokenize_es(text):
    return [tok.text for tok in nlp_es(text)]

def tokenize_pl(text):
    return [tok.text for tok in nlp_pl(text)]


# =========================
# 2. Завантаження датасету
# =========================
# Helsinki-NLP/europarl, сабсет es-pl
dataset = load_dataset("Helsinki-NLP/europarl", "es-pl")

# У цього датасету зазвичай є тільки train-спліт
train_data = dataset["train"]

print("Total samples:", len(train_data))

# Щоб не вбити MX550/CPU, візьмемо невеликий сабсет, наприклад 20k
MAX_SAMPLES = 20000
train_data = train_data.select(range(min(MAX_SAMPLES, len(train_data))))

# Розіб'ємо на train/valid вручну
VAL_RATIO = 0.1
val_size = int(len(train_data) * VAL_RATIO)
train_size = len(train_data) - val_size

train_dataset_hf = train_data.select(range(train_size))
val_dataset_hf = train_data.select(range(train_size, train_size + val_size))

print("Train size:", len(train_dataset_hf))
print("Val size:", len(val_dataset_hf))


# =========================
# 3. Токенізація та словники
# =========================

# Проста токенізація: split по пробілу
def tokenize_src(text):
    return tokenize_es(text)

def tokenize_trg(text):
    return tokenize_pl(text)

# Спеціальні токени
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def build_vocab(texts, tokenizer, max_size=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        tokens = tokenizer(t)
        counter.update(tokens)

    vocab_tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    vocab_tokens = vocab_tokens[:max_size - len(SPECIAL_TOKENS)]

    itos = SPECIAL_TOKENS + vocab_tokens
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    return itos, stoi


# Зберемо всі іспанські та польські речення
es_texts = [ex["translation"]["es"] for ex in train_dataset_hf]
pl_texts = [ex["translation"]["pl"] for ex in train_dataset_hf]

# Побудова словників
src_itos, src_stoi = build_vocab(es_texts, tokenizer=tokenize_es, max_size=20000, min_freq=2)
trg_itos, trg_stoi = build_vocab(pl_texts, tokenizer=tokenize_pl, max_size=20000, min_freq=2)

SRC_PAD_IDX = src_stoi[PAD_TOKEN]
SRC_SOS_IDX = src_stoi[SOS_TOKEN]
SRC_EOS_IDX = src_stoi[EOS_TOKEN]
SRC_UNK_IDX = src_stoi[UNK_TOKEN]

TRG_PAD_IDX = trg_stoi[PAD_TOKEN]
TRG_SOS_IDX = trg_stoi[SOS_TOKEN]
TRG_EOS_IDX = trg_stoi[EOS_TOKEN]
TRG_UNK_IDX = trg_stoi[UNK_TOKEN]

print("SRC vocab size:", len(src_itos))
print("TRG vocab size:", len(trg_itos))


# =========================
# 4. Dataset & DataLoader
# =========================

MAX_LEN = 50  # обрізаємо довгі речення

def numericalize(tokens, stoi, sos=True, eos=True, max_len=MAX_LEN):
    ids = []
    if sos:
        ids.append(stoi[SOS_TOKEN])
    for tok in tokens:
        ids.append(stoi.get(tok, stoi[UNK_TOKEN]))
        if len(ids) >= max_len - (1 if eos else 0):
            break
    if eos:
        ids.append(stoi[EOS_TOKEN])
    return ids

class TranslationDataset(Dataset):
    def __init__(self, hf_split, src_stoi, trg_stoi):
        self.data = hf_split
        self.src_stoi = src_stoi
        self.trg_stoi = trg_stoi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        src_text = ex["translation"]["es"]
        trg_text = ex["translation"]["pl"]

        src_tokens = tokenize_es(src_text)
        trg_tokens = tokenize_pl(trg_text)

        src_ids = numericalize(src_tokens, self.src_stoi, sos=True, eos=True)
        trg_ids = numericalize(trg_tokens, self.trg_stoi, sos=True, eos=True)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def collate_fn(batch):
    src_seqs, trg_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    trg_lens = [len(t) for t in trg_seqs]

    max_src_len = max(src_lens)
    max_trg_len = max(trg_lens)

    padded_src = torch.full((len(batch), max_src_len), SRC_PAD_IDX, dtype=torch.long)
    padded_trg = torch.full((len(batch), max_trg_len), TRG_PAD_IDX, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, trg_seqs)):
        padded_src[i, :len(s)] = s
        padded_trg[i, :len(t)] = t

    return padded_src, padded_trg

train_dataset = TranslationDataset(train_dataset_hf, src_stoi, trg_stoi)
val_dataset = TranslationDataset(val_dataset_hf, src_stoi, trg_stoi)

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# =========================
# 5. Модель: Encoder, Attention, Decoder, Seq2Seq
# =========================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=SRC_PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch, src_len, hid_dim]
        # hidden: [n_layers, batch, hid_dim]
        # cell:   [n_layers, batch, hid_dim]
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch, dec_hid_dim] (останній шар)
        # encoder_outputs: [batch, src_len, enc_hid_dim]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # repeat hidden src_len разів
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attn_weights = torch.softmax(attention, dim=1)  # [batch, src_len]
        return attn_weights


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=TRG_PAD_IDX)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        # input: [batch] (поточний токен)
        # hidden: [n_layers, batch, dec_hid_dim]
        # cell:   [n_layers, batch, dec_hid_dim]
        # encoder_outputs: [batch, src_len, enc_hid_dim]

        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        # беремо останній шар hidden для уваги
        dec_hidden_last = hidden[-1]  # [batch, dec_hid_dim]
        attn_weights = self.attention(dec_hidden_last, encoder_outputs, mask=mask)  # [batch, src_len]

        # контекстний вектор
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, enc_hid_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim + enc_hid_dim]

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs: [batch, 1, dec_hid_dim]

        # об'єднуємо для проєкції в словник
        output = outputs.squeeze(1)  # [batch, dec_hid_dim]
        context = context.squeeze(1)  # [batch, enc_hid_dim]
        embedded = embedded.squeeze(1)  # [batch, emb_dim]

        pred_input = torch.cat((output, context, embedded), dim=1)  # [batch, dec_hid_dim + enc_hid_dim + emb_dim]
        prediction = self.fc_out(pred_input)  # [batch, output_dim]

        return prediction, hidden, cell, attn_weights.squeeze(1)  # attn_weights: [batch, src_len]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def create_src_mask(self, src):
        # src: [batch, src_len]
        mask = (src != self.src_pad_idx).to(src.device)  # 1 там, де не PAD
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        attn_matrices = []

        encoder_outputs, hidden, cell = self.encoder(src)
        mask = self.create_src_mask(src)

        # перший токен декодера - SOS
        input = trg[:, 0]  # [batch]

        for t in range(1, trg_len):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs, mask=mask)
            outputs[:, t, :] = output
            attn_matrices.append(attn_weights.detach().cpu())  # [batch, src_len]

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # [batch]

            input = trg[:, t] if teacher_force else top1

        # attn_matrices: список довжиною trg_len-1, кожен [batch, src_len]
        # для аналізу уваги можна брати перший елемент batch
        return outputs, attn_matrices

# =========================
# 6. Ініціалізація моделі, лос, оптимізатор
# =========================

INPUT_DIM = len(src_itos)
OUTPUT_DIM = len(trg_itos)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, n_layers=N_LAYERS, dropout=ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, n_layers=N_LAYERS, dropout=DEC_DROPOUT)

model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================
# 7. Функції навчання / валідації
# =========================

def train_epoch(model, loader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, trg in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()

        output, _ = model(src, trg, teacher_forcing_ratio=0.5)
        # output: [batch, trg_len, output_dim]

        # зрушуємо на 1, бо перший токен - SOS
        output_dim = output.shape[-1]
        output = output[:, 1:, :].contiguous().view(-1, output_dim)  # [batch*(trg_len-1), output_dim]
        trg = trg[:, 1:].contiguous().view(-1)  # [batch*(trg_len-1)]

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            output, _ = model(src, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)

# =========================
# 8. Цикл навчання
# =========================

N_EPOCHS = 10  # для MX550/CPU можна почати з 5-10

train_losses = []
val_losses = []

for epoch in range(1, N_EPOCHS + 1):
    start_time = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = eval_epoch(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    end_time = time.time()
    epoch_mins = int((end_time - start_time) // 60)
    epoch_secs = int((end_time - start_time) % 60)

    print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(f"\t Val. Loss: {val_loss:.3f}")

# =========================
# 9. Графік функції втрат
# =========================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Seq2Seq es→pl: Loss over epochs")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 10. Функція перекладу + збереження уваги
# =========================

def translate_sentence(model, sentence, src_stoi, src_itos, trg_itos, max_len=MAX_LEN):
    model.eval()
    tokens = tokenize_es(sentence)

    src_ids = numericalize(tokens, src_stoi, sos=True, eos=True, max_len=max_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, src_len]

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        mask = model.create_src_mask(src_tensor)

        # починаємо з SOS
        input = torch.tensor([TRG_SOS_IDX], dtype=torch.long).to(DEVICE)

        outputs = []
        attn_mats = []

        for t in range(max_len):
            output, hidden, cell, attn_weights = model.decoder(input, hidden, cell, encoder_outputs, mask=mask)
            attn_mats.append(attn_weights.detach().cpu())  # [1, src_len]

            pred_token = output.argmax(1).item()
            if pred_token == TRG_EOS_IDX:
                break
            outputs.append(pred_token)
            input = torch.tensor([pred_token], dtype=torch.long).to(DEVICE)

    trg_tokens = [trg_itos[idx] for idx in outputs]
    return trg_tokens, attn_mats, tokens

# =========================
# 11. Візуалізація механізму уваги
# =========================

def plot_attention(src_tokens, trg_tokens, attn_mats):
    # attn_mats: список довжиною len(trg_tokens), кожен [1, src_len]
    import numpy as np

    attn = torch.stack(attn_mats, dim=0).squeeze(1).numpy()  # [trg_len, src_len]

    fig, ax = plt.subplots(figsize=(len(src_tokens) * 0.5, len(trg_tokens) * 0.5))
    im = ax.imshow(attn, cmap="viridis")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))

    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(trg_tokens)

    ax.set_xlabel("Source (es)")
    ax.set_ylabel("Target (pl)")
    ax.set_title("Attention weights")

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

# =========================
# 12. Приклади перекладу + увага
# =========================

# Візьмемо кілька речень з валідаційного датасету
for i in range(3):
    ex = val_dataset_hf[i]
    src_text = ex["translation"]["es"]
    trg_ref = ex["translation"]["pl"]

    pred_tokens, attn_mats, src_tokens = translate_sentence(model, src_text, src_stoi, src_itos, trg_itos)
    pred_text = " ".join(pred_tokens)

    print("=" * 80)
    print("SRC (es):", src_text)
    print("REF (pl):", trg_ref)
    print("PRED(pl):", pred_text)

    # Візуалізація уваги
    plot_attention(src_tokens, pred_tokens, attn_mats)

