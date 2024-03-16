import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
print(torch.cuda.is_available())
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
print(train_dataset[0])

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

batch_size = 32
learning_rate = 0.01
batch_size = 32
num_epochs = 60

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, num_lstm_layers, dropout_prob, linear_output_dim, num_labels):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id['<PAD>'])  # Use padding_idx
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.elu(self.linear(lstm_out))
        logits = self.classifier(linear_out)
        return logits

# Initialize the model
model = BiLSTMModel(vocab_size, embedding_dim, lstm_hidden_dim, num_lstm_layers, lstm_dropout, linear_output_dim, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# Print the model architecture
print(model)

def collate_fn(batch):
    # Sort the batch by sequence length in decreasing order
    batch = sorted(batch, key=lambda x: len(x["tokens"]), reverse=True)
    
    # Get the length of the longest sequence in the batch
    max_len = len(batch[0]["tokens"])
    
    # Initialize lists for tokens and labels
    tokens = []
    labels = []
    
    for example in batch:
        # Convert tokens to numerical values using the vocabulary
        token_ids = [word2id.get(token, word2id['<UNK>']) for token in example["tokens"]]
        tokens.append(torch.tensor(token_ids, dtype=torch.long))
        
        # You can remove the custom padding for labels
        labels.append(torch.tensor(example["labels"], dtype=torch.long))
    
    # Use pad_sequence to pad the tokens to the same length
    token_tensor = pad_sequence(tokens, batch_first=True, padding_value=word2id['<PAD>'])
    labels_tensor = pad_sequence(labels, batch_first=True, padding_value=-100)
    # No need to pad labels since they should already be of the same length
    
    return {"tokens": token_tensor, "labels": labels_tensor}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

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

# Evaluation loop for the validation dataset
if torch.cuda.is_available():
    checkpoint = torch.load('/content/best_train_model.pth')
else:
    checkpoint = torch.load('/content/best_train_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

model.eval()  # Set the model in evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # Ensure no gradient calculation during evaluation
    for batch in tqdm(test_loader):  # Iterate over the validation data loader
        inputs = batch["tokens"].to(device) # Access the inputs from the batch dictionary
        labels = batch["labels"].to(device) # Access the labels from the batch dictionary

#         print(labels)
        outputs = model(inputs)
        max_tag_ids = torch.argmax(outputs, dim=-1)
        value_to_remove = -100

        preds = convert_predictions_to_tags(max_tag_ids, id2label)
        golds = convert_labels_to_tags(labels, id2label)

        # Remove padding from both predictions and actual labels
        for pred, gold in zip(preds, golds):
#             print(pred, gold)
            pred = [p for p, label in zip(pred, gold) if label != '<UNK>']
            gold = [label for label in gold if label != '<UNK>']
            all_preds.extend(pred)
            all_labels.extend(gold)
#         print(all_preds)

# Compute precision, recall, and F1-score using conlleval
precision, recall, f1 = evaluate(all_labels, all_preds)

# Print the evaluation metrics
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-Score: {f1:.4f}")
