"""Filter `merged_files.csv` to only rows where City contains 'Hyderabad' (case-insensitive).

Usage:
    python filter_hyderabad.py           # writes merged_hyderabad.csv
    python filter_hyderabad.py --inplace # overwrite merged_files.csv (creates backup merged_files_backup.csv)
"""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='merged_files.csv', help='Path to merged CSV')
    parser.add_argument('--out', default='merged_hyderabad.csv', help='Output filtered CSV')
    parser.add_argument('--inplace', action='store_true', help='Overwrite the original CSV (makes a backup)')
    args = parser.parse_args()

    root = Path(__file__).parent
    csv_path = root / args.csv
    if not csv_path.exists():
        print(f'Dataset not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    total = len(df)
    if 'City' not in df.columns:
        print('No City column in CSV; cannot filter by Hyderabad')
        return

    # case-insensitive contains match
    hy = df[df['City'].astype(str).str.contains('Hyderabad', case=False, na=False)].copy()
    hy_count = len(hy)

    out_path = root / args.out
    hy.to_csv(out_path, index=False)
    print(f'Wrote {hy_count} Hyderabad rows to: {out_path} (from {total} total rows)')

    if args.inplace:
        backup = root / (args.csv.replace('.csv','') + '_backup.csv')
        csv_path.replace(backup)
        out_path.replace(csv_path)
        print(f'Backed up original to {backup} and replaced original with Hyderabad-only CSV')

if __name__ == '__main__':
    main()
