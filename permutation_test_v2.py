import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

genotypes = ['WT', 'UAS-TeTxLC', 'UAS-PTX', 'UAS-Kir2.1', 'NaChBac', 'Pdf-RNAi']
sheet_names = ['ZT0', 'ZT6']
file_path = r"C:\Users\yanggeq\Downloads\permtest2_s7E.xlsx"
column_numbers = list(range(len(genotypes)))
perm_num = 10000


def load_excel_data(filepath, sheet_names, genotypes, cols):
    dfs = []
    for sheet in sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet, usecols=cols).dropna(how='all')
        df.columns = genotypes
        df.dropna(how='all', inplace=True)
        df = df.melt(var_name='Genotype', value_name='Normalized Mean Intensity').dropna()
        df['Timepoint'] = sheet
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).dropna()


def compute_diff_and_ratio(df, genotype):
    # Compute mean difference and ratio between ZT6 and ZT0 for a given genotype.
    df_g = df[df['Genotype'] == genotype].dropna()

    mean_zt0 = df_g[df_g['Timepoint'] == 'ZT0']['Normalized Mean Intensity'].mean()
    mean_zt6 = df_g[df_g['Timepoint'] == 'ZT6']['Normalized Mean Intensity'].mean()

    return mean_zt6 - mean_zt0, mean_zt6 / mean_zt0

def permutation_test(df, genotype, wt_diff, wt_ratio, num_permutations):
    #using np.random.permutation instead of rng.permuted
    observed_diff, observed_ratio = compute_diff_and_ratio(df, genotype)
    diff_stat = abs(observed_diff - wt_diff)
    ratio_stat = abs(observed_ratio - wt_ratio)

    perm_diffs = []
    perm_ratios = []

    for _ in range(num_permutations):
        shuffled = df.copy()
        shuffled['Timepoint'] = np.random.permutation(shuffled['Timepoint'])
        wt_d, wt_r = compute_diff_and_ratio(shuffled, 'WT')
        mut_d, mut_r = compute_diff_and_ratio(shuffled, genotype)
        perm_diffs.append(abs(wt_d - mut_d))
        perm_ratios.append(abs(wt_r - mut_r))

    # Compute p-values
    #alpha, beta = 0.5, 1  # Prior to avoid zero p-values
    #p_value_diff = (np.sum(np.array(perm_diffs) >= observed_diff) + alpha) / (num_permutations + beta)
    #p_value_ratio = (np.sum(np.array(perm_ratios) >= observed_ratio) + alpha) / (num_permutations + beta)

    p_diff = np.mean(np.array(perm_diffs) >= diff_stat)
    p_ratio = np.mean(np.array(perm_ratios) >= ratio_stat)
    return diff_stat, ratio_stat, p_diff, p_ratio, perm_diffs, perm_ratios




def plot_permutation_results(diff_null, ratio_null, observed_diff, observed_ratio, genotype):
    # Plot null distributions and observed test statistics for difference and ratio
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(diff_null, bins=50, kde=True, ax=axes[0])
    axes[0].axvline(observed_diff, color='red', linestyle='--', label=f'Observed Diff = {observed_diff:.2e}')
    axes[0].set(title=f'Diff Null Distribution\n{genotype} vs WT',
                xlabel='Difference of Means', ylabel='Frequency')
    axes[0].legend()

    sns.histplot(ratio_null, bins=50, kde=True, ax=axes[1])
    axes[1].axvline(observed_ratio, color='red', linestyle='--', label=f'Observed Ratio = {observed_ratio:.2e}')
    axes[1].set(title=f'Ratio Null Distribution\n{genotype} vs WT',
                xlabel='Ratio of Means', ylabel='Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def main():

    df = load_excel_data(file_path, sheet_names, genotypes, column_numbers)
    wt_diff, wt_ratio = compute_diff_and_ratio(df, 'WT')

    results = {}

    for genotype in df['Genotype'].unique():
        if genotype == 'WT':
            continue

        diff_stat, ratio_stat, p_diff, p_ratio, perm_diffs, perm_ratios = permutation_test(
            df, genotype, wt_diff, wt_ratio, perm_num
        )

        plot_permutation_results(perm_diffs, perm_ratios, diff_stat, ratio_stat, genotype)

        results[genotype] = {
            'P-value (Difference)': p_diff,
            'P-value (Ratio)': p_ratio
        }

    results_df = pd.DataFrame(results).T.applymap(lambda x: f"{x:.2e}")
    print("\nPermutation Test Results (Scientific Notation):")
    print(results_df)


if __name__ == "__main__":
    main()