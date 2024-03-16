import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import requests
# !wget https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py

url = "https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py"
response = requests.get(url)

import torch
# print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id2label = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG' , 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC' ,8:'I-MISC', -100:'<UNK>'}

from datasets import load_dataset


from tqdm.auto import tqdm
import torch

print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter

import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
from collections import defaultdict

from conlleval import evaluate
import itertools

dataset = load_dataset("conll2003")

def preprocess_example(example):
    # Rename "ner_tags" to "labels"
    example["labels"] = example["ner_tags"]
    
    # Remove the "pos_tags" and "chunk_tags" columns
    example.pop("pos_tags")
    example.pop("chunk_tags")
    example.pop("ner_tags")
    
    # Convert the text to lowercase
#     example["tokens"] = [token.lower() for token in example["tokens"]]
    
    return example

# Apply the preprocessing function to each split in the dataset
dataset = dataset.map(preprocess_example)

# Access the preprocessed splits
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Print the first example in the training dataset to check the changes
# print(train_dataset[0])

word_dict = defaultdict(int)
for line in train_dataset:
    for word in line['tokens']:
        word_dict[word] += 1

word_dict['<UNK>'] = 0
word_dict['<PAD>'] = 1
word2id = {}
id2word = {}
for idx, word in enumerate(word_dict.keys()):
    word2id[word] = idx
    id2word[idx] = word

# Define hyperparameters
embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
lstm_dropout = 0.33
linear_output_dim = 128
vocab_size = len(word2id)
num_labels = 9

learning_rate = 0.01
batch_size = 32
num_epochs = 60

vocab = []
embeddings = []
glove_embeddings = {}
embedding_matrix = []
glove_file = "/content/glove.6B.100d"
with open(glove_file, 'rt') as fi:
    full_content = fi.read().strip().split('\n')
    for i in tqdm(range(len(full_content))):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
        glove_embeddings[i_word] = i_embeddings  # Corrected key

embedding_matrix = torch.tensor(embeddings, dtype=torch.float32)

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

vocab_npa = np.insert(vocab_npa, 0, '<PAD>')
vocab_npa = np.insert(vocab_npa, 1, '<UNK>')

pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.

# insert embeddings for pad and unk tokens at the top of embs_npa.
embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
print(embs_npa.shape)

word2id = {}
id2word = {}

# Add padding and unknown tokens first
word2id['<PAD>'] = 0.0
word2id['<UNK>'] = 1.0
id2word[0] = '<PAD>'
id2word[1] = '<UNK>'

# Then, add the words from your GloVe embeddings
for idx, word in enumerate(vocab_npa[2:]):
    word2id[word] = float(idx) + 2  # Start index from 2 to account for the two special tokens
    id2word[float(idx) + 2] = word  # Start index from 2 to account for the two special tokens



def get_additional_features(token):
    additional_features = []
    if word!="<PAD>" :
        is_uppercase = 1.0 if token.isupper() else 0.0
        is_lowercase = 1.0 if token.islower() else 0.0
        is_alphanumeric = 1.0 if token.isalnum() else 0.0
        is_title = 1.0 if token.istitle() else 0.0
    
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])
    token_features = np.array([is_uppercase, is_lowercase, is_alphanumeric, is_title])
#     additional_features.append(token_features)

    return token_features

    
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x["tokens"]), reverse=True)
    max_len = len(batch[0]["tokens"])
    tokens = []
    labels = []
    og_tokens = []
    word_tokens_list =[]
#     get_add_feature = []

    for example in batch:

        token_ids = [word2id.get(token.lower(), word2id['<UNK>']) for token in example["tokens"]]
        
        tokens.append(torch.tensor(token_ids, dtype=torch.long))
        labels.append(torch.tensor(example["labels"], dtype=torch.long))

    padding_value = float(word2id['<PAD>'])
    

    token_tensor = pad_sequence(tokens, batch_first=True, padding_value=padding_value)
    labels_tensor = pad_sequence(labels, batch_first=True, padding_value=-100)

    additional_features = []

    for example in batch:
        word_tokens = example["tokens"]
        padded_word_tokens = word_tokens + ["<PAD>"] * (token_tensor.shape[1] - len(word_tokens))

        add_feat = [get_additional_features(token) for token in padded_word_tokens]
        additional_features.append(torch.tensor(add_feat, dtype=torch.float32))

    # Convert the list of tensors to a single tensor
    additional_features_tensor = torch.stack(additional_features)

    return {"tokens": token_tensor, "labels": labels_tensor, 'og_tokens': additional_features_tensor}


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

embedding_dim = len(glove_embeddings['example'])  # Get the embedding dimension from the GloVe vectors

class GloVeBiLSTMModel(nn.Module):
    def __init__(self, lstm_hidden_dim, num_lstm_layers, dropout_prob, linear_output_dim, num_labels):
        super(GloVeBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), padding_idx=int(word2id['<PAD>']), freeze=True)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim+4,#+4,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_labels)

    def forward(self, x, y):
#         x = x.long()
        embedded = self.embedding(x)
        add_features = y
#         print(embedded.shape, add_features.shape)
        embed = torch.cat([embedded, add_features], dim=2)
        lstm_out, _ = self.bilstm(embed)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.elu(self.linear(lstm_out))
        logits = self.classifier(linear_out)
        return logits

model = GloVeBiLSTMModel(lstm_hidden_dim, num_lstm_layers, lstm_dropout, linear_output_dim, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Print the model architecture
print(model)

def convert_predictions_to_tags(predictions, id2label):
    # Convert the prediction tensor to a list of tag sequences
    tag_sequences = []
    for prediction in predictions:
        tag_sequence = [id2label[tag_id.item()] for tag_id in prediction]
        tag_sequences.append(tag_sequence)
    return tag_sequences

def convert_labels_to_tags(labels, id2label):
    # Convert the label tensor to a list of tag sequences
    tag_sequences = []
    for label_sequence in labels:
        tag_sequence = [id2label[label_id.item()] for label_id in label_sequence if label_id != -100]
        tag_sequences.append(tag_sequence)
    return tag_sequences

checkpoint = torch.load('/content/best_train_model.pth', map_location=torch.device('cpu'))

# Instantiate your model class
model = GloVeBiLSTMModel(lstm_hidden_dim, num_lstm_layers, lstm_dropout, linear_output_dim, num_labels)

# Move the model to the same device as the checkpoint
model.load_state_dict(checkpoint)
model.to(device)

# Set the model in evaluation mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():  # Ensure no gradient calculation during evaluation
    for batch in tqdm(test_loader):  # Iterate over the validation data loader
        inputs = batch["tokens"].to(device)  # Access the inputs from the batch dictionary
        labels = batch["labels"].to(device)  # Access the labels from the batch dictionary
        og_tokens = batch["og_tokens"].to(device)
        
        outputs = model(inputs, og_tokens)
        
        max_tag_ids = torch.argmax(outputs, dim=-1)
        preds = convert_predictions_to_tags(max_tag_ids, id2label)
        golds = convert_labels_to_tags(labels, id2label)

        # Remove padding from both predictions and actual labels
        for pred, gold in zip(preds, golds):
            pred = [p for p, label in zip(pred, gold) if label != '<UNK>']
            gold = [label for label in gold if label != '<UNK>']
            all_preds.extend(pred)
            all_labels.extend(gold)

# Compute precision, recall, and F1-score using conlleval
precision, recall, f1 = evaluate(all_labels, all_preds)

# Print the evaluation metrics
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-Score: {f1:.4f}")

