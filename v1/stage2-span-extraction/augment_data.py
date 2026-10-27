"""
Stage 2 - auxiliary: training-set augmentation.

Duplicates posts that carry a span annotation (span != "NONE") a fixed
number of times to counter the positive/negative imbalance, then shuffles
and writes the augmented CSV consumed by config.train_file.
"""
import pandas as pd
import argparse

def augment_data(input_file, output_file, repeat_times=3):
    """Repeat span-annotated (positive) rows repeat_times, keep negatives as-is."""
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"原始数据量: {len(df)}")

    pos_df = df[df['span'] != "NONE"].copy()
    neg_df = df[df['span'] == "NONE"].copy()
    print(f"正样本数: {len(pos_df)}, 负样本数: {len(neg_df)}")

    augmented_pos = pd.concat([pos_df] * repeat_times, ignore_index=True)
    print(f"增强后正样本数: {len(augmented_pos)}")

    augmented_df = pd.concat([augmented_pos, neg_df], ignore_index=True)
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"增强后总数据量: {len(augmented_df)}")
    print(f"保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/秘制小训练集.csv")
    parser.add_argument("--output", default="data/秘制小训练集_augmented.csv")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    augment_data(args.input, args.output, args.repeat)
