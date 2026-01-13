import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ast

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = './test2.csv'
SAVE_DIR = './analysis_output'
os.makedirs(SAVE_DIR, exist_ok=True)

def parse_history(x):
    try:
        if isinstance(x, list):
            return x
        return ast.literal_eval(x)
    except:
        return []

def main():
    print(f"Analyzing {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} users.")
    
    df['parsed_history'] = df['history_item_id'].apply(parse_history)
    lengths = df['parsed_history'].apply(len)
    
    # 1. Sequence Length Distribution (focused on 1-50)
    plt.figure(figsize=(12, 6))
    bins = list(range(1, 52))
    counts, _, patches = plt.hist(lengths[lengths <= 50], bins=bins, color='#4C72B0', edgecolor='white', alpha=0.85)
    
    # Highlight 1-10 range
    for i, patch in enumerate(patches):
        if i < 10:
            patch.set_facecolor('#C44E52')
    
    plt.axvline(x=10.5, color='red', linestyle='--', linewidth=2, label='Threshold=10')
    plt.title('Sequence Length Distribution (Length <= 50)', fontsize=14)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage annotation
    short_seq_ratio = (lengths <= 10).sum() / len(lengths) * 100
    plt.annotate(f'Short Sequences (1-10): {short_seq_ratio:.1f}%', 
                 xy=(5, counts[0:10].max()), fontsize=11, color='#C44E52', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'length_dist_focused.png'), dpi=150)
    plt.close()
    
    # 2. Item Popularity Top 30
    all_items = [item for sublist in df['parsed_history'] for item in sublist]
    item_counts = pd.Series(all_items).value_counts()
    
    plt.figure(figsize=(14, 6))
    top30 = item_counts.head(30)
    bars = plt.bar(range(len(top30)), top30.values, color='#55A868', edgecolor='white')
    plt.title('Top 30 Most Popular Items', fontsize=14)
    plt.xlabel('Item Rank', fontsize=12)
    plt.ylabel('Interaction Count', fontsize=12)
    plt.xticks(range(len(top30)), [f'#{i+1}' for i in range(len(top30))], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'top_items_30.png'), dpi=150)
    plt.close()
    
    # 3. Long-tail distribution (log-log)
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(item_counts)+1), item_counts.values, 'o', markersize=2, color='#8172B2', alpha=0.6)
    plt.title('Item Popularity Long-Tail Distribution (Log-Log)', fontsize=14)
    plt.xlabel('Item Rank (log scale)', fontsize=12)
    plt.ylabel('Interaction Count (log scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'longtail_loglog.png'), dpi=150)
    plt.close()
    
    print(f"Images saved to {SAVE_DIR}")
    print(f"Short sequences (<=10): {short_seq_ratio:.2f}%")

if __name__ == "__main__":
    main()
