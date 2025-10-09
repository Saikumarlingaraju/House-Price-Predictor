"""Small utility to load a model_*.pkl and run a sample prediction.

Usage:
    python test_model.py model_Hyderabad.pkl
"""
import sys
import pickle
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_model.py <model_file.pkl>')
        raise SystemExit(1)
    model_path = sys.argv[1]
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)
    # Build a tiny sample input (Area=1000, Bedrooms=2, City blank)
    sample = pd.DataFrame([{'Area': 1000, 'No. of Bedrooms': 2, 'City': ''}])
    try:
        pred = model.predict(sample)
        print('Sample prediction:', float(pred[0]))
    except Exception as e:
        print('Prediction failed:', e)


if __name__ == '__main__':
    main()
