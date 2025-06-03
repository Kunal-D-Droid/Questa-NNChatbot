import pandas as pd
import json
import os

def load_corpora_data():
    """Load data from hotel-corpora CSV files"""
    data = []
    
    # Load dialogs
    dialogs_df = pd.read_csv('data/raw/hotel-corpora/dialogs.csv')
    for _, row in dialogs_df.iterrows():
        if pd.notna(row['question']) and pd.notna(row['answer']):
            data.append({
                "input": row['question'],
                "output": row['answer']
            })
    
    # Load paraphrases
    paraphrases_df = pd.read_csv('data/raw/hotel-corpora/paraphrases.csv')
    for _, row in paraphrases_df.iterrows():
        if pd.notna(row['original']) and pd.notna(row['paraphrase']):
            data.append({
                "input": row['paraphrase'],
                "output": row['original']
            })
    
    return data

def main():
    # Load original training data
    with open('data/processed/training_data.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load corpora data
    corpora_data = load_corpora_data()
    
    # Combine data
    combined_data = {
        "conversations": original_data.get("conversations", []) + corpora_data
    }
    
    # Save augmented data
    with open('data/processed/training_data_augmented.json', 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Original conversations: {len(original_data.get('conversations', []))}")
    print(f"Added corpora conversations: {len(corpora_data)}")
    print(f"Total conversations: {len(combined_data['conversations'])}")

if __name__ == "__main__":
    main() 