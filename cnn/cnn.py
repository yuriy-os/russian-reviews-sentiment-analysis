import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from gensim.models import Word2Vec
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator
from sklearn.metrics import classification_report

SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
EMBEDDING_DIM = 300
N_FILTERS = 20
FILTER_SIZES = [1, 2, 3, 4, 5]
DROPOUT = 0.2
N_EPOCHS = 4

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator):

        optimizer.zero_grad()

        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator):

            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                    padding=(fs - 1, 0),
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


if __name__ == "__main__":
    embeddings = Vectors(
        name="./rureviews/rureviews.w2v.300d.txt",
        cache="./rureviews",
        unk_init=torch.Tensor.normal_,
    )

    TEXT = Field(batch_first=True)
    LABEL = LabelField()

    tabular_dataset = TabularDataset(
        path="./dataset.csv",
        format="csv",
        skip_header=True,
        fields=[("text", TEXT), ("label", LABEL)],
    )

    train_data, test_data = tabular_dataset.split(random_state=random.seed(SEED))
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors=embeddings)

    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        device=device,
    )

    input_dim = len(TEXT.vocab)
    output_dim = len(LABEL.vocab)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    pretrained_embeddings = TEXT.vocab.vectors

    model = CNN(
        input_dim, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, output_dim, DROPOUT, pad_idx
    )
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[pad_idx] = torch.zeros(EMBEDDING_DIM)

    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        print("Training ...")
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        print("Evaluating ...")
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "rureviews_sentiment_model.pt")

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    y_pred = []
    y_test = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_iterator):
            y_test.extend(batch.label.tolist())
            _, y_pred_label = torch.max(model(batch.text), dim=1)
            y_pred.extend(y_pred_label.tolist())
    print(classification_report(y_test, y_pred, digits=4))
