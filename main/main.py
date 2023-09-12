import shutil
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import argparse
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartConfig
)
from automatic_eval import AutomaticEvaluator

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    pd.np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def preprocess_data(src_data, trg_data):
    df = pd.DataFrame({src_col: src_data[src_col], trg_col: trg_data[trg_col]})
    df = df.sample(frac=1)
    return df

def tokenize_datasets(df, src_col, trg_col, tokenizer, max_length):
    src_encodings = tokenizer(
        df[src_col].values.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    trg_encodings = tokenizer(
        df[trg_col].values.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    dataset = CreateDataset(src_encodings, trg_encodings)
    return dataset

def train_model(model, train_dataset, dev_dataset, tokenizer, args):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.evaluate()

def generate_predictions(model, test_data, src_col, tokenizer, output_file):
    predictions = []
    for idx in range(len(test_data[src_col])):
        src = test_data[src_col].values[idx]
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=args.max_length)
        prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(prediction)

    with open(output_file, 'wb') as f:
        pickle.dump(remove_prefix(predictions, 'NEG: '), f)

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

def write_evaluation_results(acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl, filepath):
    with open(filepath, 'w') as file:
        file.write('Accuracy: ' + str(acc) + '\n')
        file.write('Bleu with Source: ' + str(bleu_withsrc) + '\n')
        file.write('Bleu with Target: ' + str(bleu_withtrg) + '\n')
        file.write('Similarity: ' + str(sim) + '\n')
        file.write('GPT2 PPL: ' + str(gpt2_ppl) + '\n')

def remove_prefix(strings, prefix):
    return [string.replace(prefix, '') for string in strings]

def remove_newline(df):
    df = df.apply(lambda x: x.str.replace('\n', ''))
    return df

def main(args):
    shutil.rmtree('facebook', ignore_errors=True)
    set_seed(args.seed_value)

    train_df = remove_newline(pd.read_csv(args.train_file))
    train_df = train_df.sample(frac = 1, random_state = args.seed_value)
    
    dev_df = remove_newline(pd.read_csv(args.dev_file))
    dev_df = dev_df.sample(frac = 1, random_state = args.seed_value)
    
    test_df = remove_newline(pd.read_pickle(args.test_file))

    neg_prompt = 'NEG: '
    pos_prompt = 'POS: '
    if args.prompt_enabled:
        
        train_df["pos"] = train_df["pos"].apply(lambda x: pos_prompt+x)
        train_df["neg"] = train_df["neg"].apply(lambda x: neg_prompt+x)
        
        dev_df["pos"] = dev_df["pos"].apply(lambda x: pos_prompt+x)
        dev_df["neg"] = dev_df["neg"].apply(lambda x: neg_prompt+x)
        
        test_df["pos"] = test_df["pos"].apply(lambda x: pos_prompt+x)
        test_df["neg"] = test_df["neg"].apply(lambda x: neg_prompt+x)

    with open('../output/pos_to_neg/src.pkl', 'wb') as f:
        pickle.dump(remove_prefix(test_df[args.src_col].values.tolist(), pos_prompt), f)

    with open('../output/pos_to_neg/trg.pkl', 'wb') as f:
        pickle.dump(remove_prefix(test_df[args.trg_col].values.tolist(), neg_prompt), f)

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    train_dataset = tokenize_datasets(train_df, args.src_col, args.trg_col, tokenizer, args.max_length)
    dev_dataset = tokenize_datasets(dev_df, args.src_col, args.trg_col, tokenizer, args.max_length)

    # Load the BART model
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    config = BartConfig.from_pretrained(args.model_name)
    config.dropout = args.dropout #0.15
    config.attention_dropout = args.attention_dropout #0.05
    config.activation_dropout = args.activation_dropout #0.05
    
    config.label_smoothing_factor = args.label_smoothing_factor #0.05
    
    model.config = config
    
    # Initialize training arguments
    training_args = Seq2SeqTrainingArguments(
        args.model_name,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=1,
        save_strategy='epoch',
        load_best_model_at_end=True,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=True
    )

    # Train the model
    train_model(model, train_dataset, dev_dataset, tokenizer, training_args)

    # Generate predictions on test data
    # test_data = pd.read_csv(args.test_file)
    generate_predictions(model, test_df, args.src_col, tokenizer, args.output_file)
    print('Completed.')

    evaluator = AutomaticEvaluator(seed_value=args.seed_value)
    evaluator.set_seed()
    evaluator.load_models()
    acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl = evaluator.evaluate_all(
        srcs='../output/pos_to_neg/src.pkl',
        refs='../output/pos_to_neg/trg.pkl',
        preds='../output/pos_to_neg/pred.pkl',
        target_label='NEGATIVE'
    )
    
    # Print or use the evaluation results as needed
    # print('Accuracy', acc)
    # print('Bleu with Source', bleu_withsrc)
    # print('Bleu with Target', bleu_withtrg)
    # print('Similarity', sim)
    # print('GPT2 PPL', gpt2_ppl)
    write_evaluation_results(acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl, args.eval_scores_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_value', type=int, default=53)
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--activation_dropout', type=float, default=0.0)
    parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
    
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--eval_scores_file', type=str, required=True)
    parser.add_argument('--prompt_enabled', action='store_true', help='Enable the prompt flag if you want to use')
    parser.add_argument('--src_col', type=str, default='pos')
    parser.add_argument('--trg_col', type=str, default='neg')
    parser.add_argument('--dev_size', type=float, default=0.1)
    parser.add_argument('--train_pkl_path', type=str, required=True)

    args = parser.parse_args()
    main(args)
