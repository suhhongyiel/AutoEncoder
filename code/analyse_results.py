import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_csv in glob.glob(str(results_dir / '*/*/metrics.csv')):
        parts = Path(metrics_csv).parts
        modality, label = parts[-3].split('_')
        df = pd.read_csv(metrics_csv)
        df['modality'] = modality
        df['label'] = label
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_dir / 'all_metrics.csv', index=False)
    print('[saved]', out_dir / 'all_metrics.csv')
    # Boxplots
    for metric in ['mse','psnr','ssim']:
        plt.figure(figsize=(8,5))
        all_df.boxplot(column=metric, by=['modality','label'])
        plt.title(f'{metric} by modality/label')
        plt.suptitle('')
        plt.ylabel(metric)
        plt.savefig(out_dir / f'{metric}_boxplot.png')
        plt.close()
    print('[saved]', out_dir / 'boxplots')

if __name__ == '__main__':
    main() 