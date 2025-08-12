import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import random


# Seed setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess_code(code, language):
    try:
        if language == 'py':
            code = re.sub(r'#.*?\n', '\n', code)
            code = re.sub(r'\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"', '', code, flags=re.DOTALL)
        elif language == 'java':
            code = re.sub(r'//.*?\n|/\*.*?\*/', '\n', code, flags=re.DOTALL)
        elif language == 'cs':
            code = re.sub(r'///?.*?\n', '\n', code)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # code = re.sub(r'[{}();:,\[\]]', ' ', code)
        code = ' '.join(code.split()).strip()

        return code if code else ""
    except Exception as e:
        print(f"Error preprocessing code: {e}")
        return ""

def preprocess_dataset(df):
    df_processed = df.copy()
    df_processed['clone1'] = df_processed.apply(lambda row: preprocess_code(row['clone1'], row['language']), axis=1)
    df_processed['clone2'] = df_processed.apply(lambda row: preprocess_code(row['clone2'], row['language']), axis=1)
    return df_processed

class CloneDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.clone1 = df['clone1'].tolist()
        self.clone2 = df['clone2'].tolist()
        self.labels = df['is_semantic_clone'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        c1 = self.tokenizer(self.clone1[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        c2 = self.tokenizer(self.clone2[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids1': c1['input_ids'].squeeze(0),
            'attention_mask1': c1['attention_mask'].squeeze(0),
            'input_ids2': c2['input_ids'].squeeze(0),
            'attention_mask2': c2['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def evaluate_model_cosine(model, test_loader, device, df_test, model_name, threshold=0.7):
    model.eval()
    preds, labels, probs = [], [], []

    is_codet5 = 't5' in model_name.lower()

    with torch.no_grad():
        for batch in test_loader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            batch_labels = batch['labels'].to(device)

            if is_codet5:
                # output1 = model.encoder(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state[:, 0, :]
                # output2 = model.encoder(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state[:, 0, :]
                output1 = model.encoder(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state.mean(1)
                output2 = model.encoder(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state.mean(1)
            else:
                # output1 = model(input_ids1, attention_mask=attention_mask1).last_hidden_state[:, 0, :]
                # output2 = model(input_ids2, attention_mask=attention_mask2).last_hidden_state[:, 0, :]
                output1 = model(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state.mean(1)
                output2 = model(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state.mean(1)

            sim_scores = torch.nn.functional.cosine_similarity(output1, output2, dim=1)
            prob_scores = sim_scores.cpu().numpy()
            predictions = (sim_scores > threshold).long().cpu().numpy()

            preds.extend(predictions)
            labels.extend(batch_labels.cpu().numpy())
            probs.extend(prob_scores)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)
    TN, FP, FN, TP = cm.ravel()

    df_output = df_test.copy().reset_index(drop=True)
    df_output['prediction'] = preds
    df_output['probability'] = probs
    df_output['correct'] = df_output['prediction'] == df_output['is_semantic_clone']

    output_filename = f"{model_name}_results_cosine_2_U.xlsx"
    df_output.to_excel(output_filename, index=False)
    print(f"Saved detailed results to {output_filename}")

    return precision, recall, f1, TN, FP, FN, TP, preds, labels, cm

def analyze_performance_by_language(df_test, preds, labels):
    df_test = df_test.copy().reset_index(drop=True)
    df_test['prediction'] = preds
    df_test['correct'] = df_test['prediction'] == df_test['is_semantic_clone']

    metrics = {
        'Overall': {
            'Precision': precision_recall_fscore_support(labels, preds, average='binary')[0],
            'Recall': precision_recall_fscore_support(labels, preds, average='binary')[1],
            'F1': precision_recall_fscore_support(labels, preds, average='binary')[2]
        }
    }

    for lang in ['py', 'java', 'cs']:
        lang_df = df_test[df_test['language'] == lang]
        if not lang_df.empty:
            lang_preds = lang_df['prediction']
            lang_labels = lang_df['is_semantic_clone']
            precision, recall, f1, _ = precision_recall_fscore_support(lang_labels, lang_preds, average='binary')
            metrics[lang] = {
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }

    return metrics

# df = pd.read_csv("/home/rushabh.botadra/ModelEvaluation/clone_dataset_final_with_cs.csv")
df = pd.read_csv("clone_dataset.csv")
df_preprocessed = preprocess_dataset(df)

models = [
    { 
        'name': 'CodeBERT', 
        'path': 'microsoft/codebert-base'
    },
    { 
        'name': 'GraphCodeBERT', 
        'path': 'microsoft/graphcodebert-base' 
    },
    { 
        'name': 'UnixCoder', 
        'path': 'microsoft/unixcoder-base' 
    },
    { 
        'name': 'CodeT5', 
        'path': 'Salesforce/codet5-base' 
    },
]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

for model_info in models:
    print(f"\nEvaluating {model_info['name']} Model")
    tokenizer = AutoTokenizer.from_pretrained(model_info['path'])

    if model_info['name'] == 'CodeT5':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_info['path'])
    else:
        model = AutoModel.from_pretrained(model_info['path'])

    model.to(device)

    test_dataset = CloneDataset(df_preprocessed, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    precision, recall, f1, TN, FP, FN, TP, preds, labels, cm = evaluate_model_cosine(
        model, test_loader, device, df_preprocessed, model_info['name'], threshold=0.7
    )

    print(f"\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    metrics = analyze_performance_by_language(df_preprocessed, preds, labels)

    print("\nPerformance Analysis by Language:")
    for lang, m in metrics.items():
        print(f"{lang}:")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1']:.4f}")