"""
Utility script to manage predictions and export data
"""

import pickle
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def export_model_summary(model_path):
    """Export model summary to JSON"""
    model_path = Path(model_path)
    meta_path = model_path.parent / f"{model_path.stem}_meta.json"
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        print(f"\nModel: {model_path.name}")
        print("=" * 50)
        print(f"State/City: {meta.get('state', 'Unknown')}")
        print(f"Features: {len(meta.get('features', []))}")
        print("\nMetrics:")
        metrics = meta.get('metrics', {})
        print(f"  MAE: ₹{metrics.get('mae', 0):,.2f}")
        print(f"  RMSE: ₹{metrics.get('rmse', 0):,.2f}")
        print(f"  R²: {metrics.get('r2', 0):.4f}")
        
        return meta
    else:
        print(f"No metadata found for {model_path.name}")
        return None

def compare_models(model_dir="."):
    """Compare all available models"""
    model_dir = Path(model_dir)
    models = list(model_dir.glob("model_*.pkl"))
    
    if not models:
        print("No models found!")
        return
    
    comparison = []
    
    for model_path in models:
        meta_path = model_dir / f"{model_path.stem}_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            metrics = meta.get('metrics', {})
            comparison.append({
                'Model': model_path.name,
                'State': meta.get('state', 'Unknown'),
                'Features': len(meta.get('features', [])),
                'MAE': metrics.get('mae', 0),
                'RMSE': metrics.get('rmse', 0),
                'R²': metrics.get('r2', 0)
            })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('R²', ascending=False)
    
    print("\nModel Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

def export_predictions_to_csv(history, filename=None):
    """Export prediction history to CSV"""
    if not history:
        print("No predictions to export!")
        return
    
    df = pd.DataFrame(history)
    
    if filename is None:
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Exported {len(df)} predictions to {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="House Price Predictor Utilities")
    parser.add_argument('action', choices=['summary', 'compare', 'export'],
                       help="Action to perform")
    parser.add_argument('--model', help="Model path for summary")
    parser.add_argument('--history', help="History JSON file for export")
    
    args = parser.parse_args()
    
    if args.action == 'summary':
        if args.model:
            export_model_summary(args.model)
        else:
            print("Please provide --model path")
    
    elif args.action == 'compare':
        compare_models()
    
    elif args.action == 'export':
        if args.history:
            with open(args.history, 'r') as f:
                history = json.load(f)
            export_predictions_to_csv(history)
        else:
            print("Please provide --history JSON file")
