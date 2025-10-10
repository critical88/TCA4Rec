import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/data/linfake/llm/llm_code')
from models.backbone.SASRec import SASRec


def load_test_results(checkpoint_path):
    """
    Load test results from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing test results
        
    Returns:
        Dictionary containing test results
    """
    # Find the test results file in the checkpoint directory
    test_files = [f for f in os.listdir(checkpoint_path) if f.endswith('test_results.json')]
    
    if not test_files:
        raise FileNotFoundError(f"No test results file found in {checkpoint_path}")
    
    test_file_path = os.path.join(checkpoint_path, test_files[0])
    
    with open(test_file_path, 'r') as f:
        test_results = json.load(f)
    
    return test_results


def extract_predicted_items(test_results):
    """
    Extract predicted item IDs from test results.
    
    Args:
        test_results: Dictionary containing test results
        
    Returns:
        Dictionary mapping user_id to (target_id, predicted_items)
    """
    predictions = {}
    
    for data_idx, data in test_results['raw_predictions'].items():
        user_id = data['user_id']
        target_id = data['target_id']
        
        # Extract generated titles which need to be mapped to item IDs
        generated_titles = data['generated_title']
        
        predictions[int(data_idx)] = {
            'target_id': target_id,
            'generated_titles': generated_titles
        }
    
    return predictions


def map_titles_to_ids(predictions, datapath):
    """
    Map generated titles to item IDs using the item.csv file from the processed dataset.
    
    Args:
        predictions: Dictionary mapping user_id to (target_id, predicted_items)
        datapath: Path to the processed dataset
        
    Returns:
        Dictionary mapping user_id to (target_id, predicted_item_ids)
    """
    # Load item mapping from CSV
    item_csv_path = f'{datapath}/item.csv'
    item_df = pd.read_csv(item_csv_path)
    
    # Create a title to ID mapping
    title_to_id = {}
    for row in item_df.to_dict('records'):
        item_id = row['iid']
        title = row['title']
        # Use the first part of the title for matching since generated titles might be truncated
        title_to_id[title] = int(item_id)
    
    # Map generated titles to item IDs
    for data_idx, data in predictions.items():
        predicted_item_ids = []
        
        for title in data['generated_titles']:
            # Use the first part of the title for matching
            
            # Find the closest match if exact match not found
            if title not in title_to_id:
                title = title + "!"
            if title not in title_to_id:
                predicted_item_ids.append(0)
                continue
            predicted_item_ids.append(title_to_id[title])
        
        data['predicted_item_ids'] = predicted_item_ids
    
    return predictions


def load_sasrec_model(model_name, dataset, data_path):
    """
    Load a SASRec model from a checkpoint.
    
    Args:
        model_name: Name of the SASRec model
        data_path: Path to the processed dataset
        
    Returns:
        Loaded SASRec model
    """
    # Load dataset statistics to get user_num and item_num
    statistics_path = os.path.join(data_path, 'data_stat.json')
    
    # Parse statistics file
    if os.path.exists(statistics_path):
        with open(statistics_path, 'r') as f:
            data_stat = json.load(f)
            user_num = data_stat['num_user']
            item_num = data_stat['num_item']
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(user_num, item_num, device, maxlen=10, hidden_units=64)
    
    model_path = os.path.join("cf_model", model_name, f"{dataset}.pt")
    ""
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def get_sequences(data_path):
    """
    Get interaction sequences from the dataset based on hist_id.
    
    Args:
        data_path: Path to the processed dataset
        
    Returns:
        Dictionary mapping hist_id to interaction sequence
    """
    # Load test data
    dataset_path = data_path
    test_file = os.path.join(dataset_path, 'test_5000.csv')
    
    sequences = {}
    
    # Parse test_5000.csv file which contains interaction sequences
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        for row_idx, row in enumerate(test_df.to_dict('records')):
            hist_id = eval(row['hist_id'])
            # Get the sequence from the history column
            
            sequences[row_idx] = {
                'hist_id': hist_id,
                'user_id': row['uid'],
                'target_id': row['iid']
            }
    
    return sequences


def calculate_rankings(model, device, predictions, sequences):
    """
    Calculate the ranking of target items for each history sequence.
    
    Args:
        model: SASRec model
        device: PyTorch device
        predictions: Dictionary mapping data_idx to (target_id, predicted_item_ids)
        sequences: Dictionary mapping row_idx to sequence information
        
    Returns:
        Dictionary containing ranking information
    """
    rankings = {}
    
    for data_idx, data in tqdm(predictions.items(), desc="Calculating rankings"):
        target_id = data['target_id']
        
        # Match with the corresponding row in test_5000.csv by data_idx
        if data_idx in sequences:
            row_data = sequences[data_idx]
            user_id = row_data['user_id']
            # Use the user_id to get the user's history from the model
            # For SASRec, we don't need the actual sequence here as we'll use the user_id
        else:
            continue
        
        # Get the target item ID and user ID from the sequence data
        target_id = row_data['target_id']
        user_id = row_data['user_id']
        hist_id = row_data['hist_id']
        
        # Get user history from train.txt or other source
        # For this implementation, we'll use a simple approach with a fixed sequence length
        # In a real implementation, you would load the actual user history
        
        # Create a dummy sequence for demonstration
        # In a real implementation, you would load the actual user history from train.txt
        seq = hist_id  # Use hist_id as a placeholder
        
        # Prepare sequence for model
        seq_tensor = torch.LongTensor(seq).to(device)
        if len(seq) < model.maxlen:
            # Pad sequence if needed
            pad = torch.zeros(model.maxlen - len(seq), dtype=torch.long).to(device)
            seq_tensor = torch.cat([pad, seq_tensor])
        else:
            # Take last maxlen items
            seq_tensor = seq_tensor[-model.maxlen:]
        
        seq_tensor = seq_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get user embedding
        with torch.no_grad():
            user_emb = model.get_user_embs(None, seq_tensor)
        
        # Get all item embeddings
        all_items = torch.arange(model.item_num + 1).to(device)
        with torch.no_grad():
            all_item_embs = model.item_emb(all_items)
        
        # Calculate scores
        scores = torch.matmul(user_emb, all_item_embs.transpose(0, 1))
        scores = scores.squeeze()
        
        # Get rankings
        _, indices = torch.sort(scores, descending=True)
        indices = indices.cpu().numpy()
        
        # Find rank of target item
        target_rank = np.where(indices == target_id)[0][0] + 1 if target_id in indices else -1
        
        # Find ranks of predicted items
        predicted_ranks = []
        for pred_id in data['predicted_item_ids'][:1]:
            # Skip items with pred_id equal to 0
            if pred_id == 0:
                continue
            if pred_id in indices:
                rank = np.where(indices == pred_id)[0][0] + 1
                predicted_ranks.append(int(rank))
        
        # Calculate average predicted rank if there are any valid predictions
        avg_predicted_rank = np.mean(predicted_ranks) if predicted_ranks else -1
        
        rankings[data_idx] = {
            'target_id': target_id,
            'target_rank': target_rank,
            'avg_predicted_rank': avg_predicted_rank
        }
    
    return rankings


def analyze_rankings(rankings):
    """
    Analyze the rankings of predicted items.
    
    Args:
        rankings: Dictionary containing ranking information
        
    Returns:
        Dictionary containing analysis results
    """
    all_target_ranks = []
    all_avg_predicted_ranks = []
    
    # Count items with rank 1
    target_rank_1_count = 0
    predicted_rank_1_count = 0
    
    for data_idx, data in rankings.items():
        if data['target_rank'] > 0:
            all_target_ranks.append(data['target_rank'])
            if data['target_rank'] == 1:
                target_rank_1_count += 1
        
        if data['avg_predicted_rank'] > 0:
            all_avg_predicted_ranks.append(data['avg_predicted_rank'])
            if data['avg_predicted_rank'] == 1:
                predicted_rank_1_count += 1
    
    results = {
        'target_items': {
            'count': len(all_target_ranks),
            'mean_rank': np.mean(all_target_ranks) if all_target_ranks else None,
            'median_rank': np.median(all_target_ranks) if all_target_ranks else None,
            'min_rank': np.min(all_target_ranks) if all_target_ranks else None,
            'max_rank': np.max(all_target_ranks) if all_target_ranks else None,
            'rank_1_count': target_rank_1_count,
            'rank_1_percentage': (target_rank_1_count / len(all_target_ranks) * 100) if all_target_ranks else None
        },
        'predicted_items': {
            'count': len(all_avg_predicted_ranks),
            'mean_rank': np.mean(all_avg_predicted_ranks) if all_avg_predicted_ranks else None,
            'median_rank': np.median(all_avg_predicted_ranks) if all_avg_predicted_ranks else None,
            'min_rank': np.min(all_avg_predicted_ranks) if all_avg_predicted_ranks else None,
            'max_rank': np.max(all_avg_predicted_ranks) if all_avg_predicted_ranks else None,
            'rank_1_count': predicted_rank_1_count,
            'rank_1_percentage': (predicted_rank_1_count / len(all_avg_predicted_ranks) * 100) if all_avg_predicted_ranks else None
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze test results and calculate rankings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint directory containing test results')
    parser.add_argument('--model', type=str, default="sasrec", help='Path to the model checkpoint (e.g., SASRec)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ml-1m, Toys_and_Games)')
    parser.add_argument('--output', type=str, default=None, help='Path to save the analysis results')
    
    args = parser.parse_args()
    
    data_path= f"/data2/linfake/data/processed_10_k_core=5/{args.dataset}"
    # Load test results
    print(f"Loading test results from {args.checkpoint}")
    test_results = load_test_results(args.checkpoint)
    
    # Extract predicted items
    print("Extracting predicted items")
    predictions = extract_predicted_items(test_results)
    
    # Map titles to item IDs
    print(f"Mapping titles to item IDs using item.csv from {args.dataset}")
    predictions = map_titles_to_ids(predictions, data_path)
    
    # Load SASRec model
    print(f"Loading model from {args.model}")
    model, device = load_sasrec_model(args.model, args.dataset, data_path)
    
    # Get sequences
    print(f"Loading interaction sequences from {args.dataset}")
    sequences = get_sequences(data_path)
    
    # Calculate rankings
    print("Calculating rankings")
    rankings = calculate_rankings(model, device, predictions, sequences)
    
    # Analyze rankings
    print("Analyzing rankings")
    results = analyze_rankings(rankings)
    
    # Print results
    print("\nAnalysis Results:")
    
    print("\nPredicted Items:")
    print(f"  Count: {results['predicted_items']['count']}")
    print(f"  Rank 1 Count: {results['predicted_items']['rank_1_count']}")
    print(f"  Rank 1 Percentage: {results['predicted_items']['rank_1_percentage']:.2f}%")
    
    # Save results if output path is provided
    if args.output:
        output_data = {
            'results': results,
            'rankings': rankings
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
