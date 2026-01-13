import pandas as pd
import ast
import argparse

def parse_list_string(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

def run_ensemble(file1, file2, weight1, weight2, output_file):
    print(f"Loading {file1}...")
    df1 = pd.read_csv(file1)
    print(f"Loading {file2}...")
    df2 = pd.read_csv(file2)
    
    # Ensure user_ids match
    # Assuming both files have the same users in the same order or we merge on user_id
    # Let's merge
    merged = pd.merge(df1, df2, on='user_id', suffixes=('_1', '_2'))
    
    results = []
    
    print("Ensembling...")
    for idx, row in merged.iterrows():
        user_id = row['user_id']
        items1 = parse_list_string(row['item_id_1'])
        items2 = parse_list_string(row['item_id_2'])
        
        # Reciprocal Rank Fusion
        # score = sum(weight * (1 / (k + rank)))
        # k is a constant, usually 60
        k = 60
        scores = {}
        
        # Process list 1
        for rank, item in enumerate(items1):
            scores[item] = scores.get(item, 0) + weight1 * (1.0 / (k + rank + 1))
            
        # Process list 2
        for rank, item in enumerate(items2):
            scores[item] = scores.get(item, 0) + weight2 * (1.0 / (k + rank + 1))
            
        # Sort items by score descending
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 10
        top_10 = [item for item, score in sorted_items[:10]]
        
        # Format as string
        results.append(f'{user_id},"{str(top_10)}"')
        
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("user_id,item_id\n")
        for line in results:
            f.write(line + "\n")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, required=True, help="Path to first submission file")
    parser.add_argument("--file2", type=str, required=True, help="Path to second submission file")
    parser.add_argument("--w1", type=float, default=1.0, help="Weight for file 1")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight for file 2")
    parser.add_argument("--output", type=str, default="ensemble_submission.csv", help="Output file path")
    
    args = parser.parse_args()
    
    run_ensemble(args.file1, args.file2, args.w1, args.w2, args.output)
