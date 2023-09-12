import pandas as pd
import torch
import numpy as np
import random
import time
import argparse
import logging
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import metrics

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers_interpret import SequenceClassificationExplainer

# Set up logging
logging.basicConfig(filename='sentiment_analysis.log', level=logging.INFO)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    pd.np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Preprocess data
def preprocess_data(df, num_samples):
    df_pos = df['pos'].to_list()
    pos_label = [1] * num_samples
    df_neg = df['neg'].to_list()
    neg_label = [0] * num_samples
    df_cls = pd.DataFrame(list(zip(df_pos + df_neg, pos_label + neg_label)), columns=['Text', 'Label'])
    df_cls = df_cls.sample(frac=1)
    return df_cls

def encode_data(df):
        encoded_data = tokenizer.batch_encode_plus(
            df.Text.values,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(df.Label.values)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset

# Evaluate function
def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals

# Training loop
def train_model(model):
    best_val_loss = float('inf')
    early_stop_cnt = 0
    epochs = 10
    batch_size = 3

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_dev = DataLoader(dataset_dev,
                                sampler=SequentialSampler(dataset_dev),
                                batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        start_time = time.time()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        loss_train_avg = loss_train_total / len(dataloader_train)
        val_loss, predictions, true_vals = evaluate(dataloader_dev)

        if val_loss < best_val_loss:
            early_stop_cnt = 0
        elif val_loss >= best_val_loss:
            early_stop_cnt += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'finetuned_BERT_best.model')

        val_f1 = f1_score(predictions.argmax(axis=1), true_vals, average='weighted')
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Logging
        logging.info(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(f'\tTrain Loss: {loss_train_avg:.3f}')
        logging.info(f'\t Val. Loss: {val_loss:.3f}')
        logging.info(f'\t F1 Score (Weighted): {val_f1:.3f}')

        if early_stop_cnt == 5:
            logging.info('Early Stopping...')
            break

# Function to predict the label of a text
def predict_label(inp):
    inputs = tokenizer(inp, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax()

def main(args):     
    # Load data
    train_df = pd.read_csv(args.train_data)
    dev_df = pd.read_csv(args.dev_data)
    test_df = pd.read_csv(args.test_data)

    train_df_cls = preprocess_data(train_df, 400)
    dev_df_cls = preprocess_data(dev_df, 100)
    test_df_cls = preprocess_data(test_df, 500)
    
    # Initialize BERT tokenizer and encode data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    dataset_train = encode_data(train_df_cls)
    dataset_dev = encode_data(dev_df_cls)
    dataset_test = encode_data(test_df_cls)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Start training
    train_model(model)
    
    # Load the best model
    model.load_state_dict(torch.load('finetuned_BERT_best.model'))

    #predict_label(inp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    
    parser.add_argument('--train_data', type=str, help='Path to the training data')
    parser.add_argument('--dev_data', type=str, help='Path to the development data')
    parser.add_argument('--test_data', type=str, help='Path to the test data')
    # Add more arguments as needed

    args = parser.parse_args()
    
    main(args)