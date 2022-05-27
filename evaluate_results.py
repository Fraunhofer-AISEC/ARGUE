import argparse
import math
import pandas as pd

from pathlib import Path
from typing import List, Tuple

from scipy.stats import wilcoxon

from libs.constants import BASE_PATH

# Determine the mean and standard deviation of all result files - output is a LaTeX table
# Unsupervised:
# python3 evaluate_results.py 110 210 310 410 510 610 710 --show_stddev 1
# Semi-supervised:
# python3 evaluate_results.py 110 210 310 410 510 610 710 --show_stddev 1 --p_pollution 0.0 --n_anomalies 100
# Performance vs. Number of normal samples:
# python3 evaluate_results.py 110 210 310 410 510 610 710 --show_stddev 1 --p_pollution 0.01 --first_experiment 1 --out_path ./results/n_anomalies.csv

# Configuration
this_parse = argparse.ArgumentParser(description="Merge the test result of all random seeds")
this_parse.add_argument(
    'random_seeds', nargs='+', help='Random seeds of the experiments'
)
this_parse.add_argument(
    '--p_pollution', nargs='+', help='Pollution factors to evaluate, if not set use 0.01 and 0.05'
)
this_parse.add_argument(
    '--n_anomalies', default=0, type=int, help='Number of known anomalies during training'
)
this_parse.add_argument(
    "--model_path", default=BASE_PATH / "models", type=Path, help="Path to the results (usually where the models are)"
)
this_parse.add_argument(
    "--metric_name", default=None, type=str, help="Name of the metric, usually 'AUC' or 'AP', if None show both"
)
this_parse.add_argument(
    "--show_stddev", default=False, type=bool, help="Show the standard deviation next to the mean"
)
this_parse.add_argument(
    "--first_experiment", default=False, type=bool, help="Setting for the first experiment (performance vs. train anomaly classes)"
)
this_parse.add_argument(
    "--transpose", default=True, type=bool, help="If True, the data sets are on the rows and baselines on the columns"
)
this_parse.add_argument(
    "--out_path", default=None, type=Path, help="Path to output csv - if None, use stddout instead"
)
this_args = this_parse.parse_args()

AD_NAME = "ARGUE"
P_POLLUTION = [0.01] if not this_args.p_pollution else this_args.p_pollution
BASELINE_METHODS = [AD_NAME, "A4RGUE", "AE", "MEx_CVAEC", "DeepSAD", "DevNet", "A3"] if this_args.n_anomalies > 0 \
    else [AD_NAME, "A4RGUE", "AE", "MEx_CVAEC", "DAGMM", "GANomaly", "fAnoGAN", "DeepSVDD", "REPEN", "A3"]

NAME_TO_ID = {
    # Known clusters
    "CovType_{SEED}_y_normal:1,2,3,4,y_anomalous:5,6,7,p_pollution:{POLL},n_train_anomalies:{NANO}": " CovType",
    "MNIST_{SEED}_y_normal:0,1,2,3,4,y_anomalous:5,6,7,8,9,p_pollution:{POLL},n_train_anomalies:{NANO}": " MNIST",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,6,7,8,9,10,11,12,13,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": " EMNIST",
    "FMNIST_{SEED}_y_normal:0,1,2,3,4,y_anomalous:5,6,7,8,9,p_pollution:{POLL},n_train_anomalies:{NANO}": " FMNIST",
    # Estimated clusters
    "Darknet_{SEED}_y_normal:0,6,17,y_anomalous:Tor,VPN,p_pollution:{POLL},n_train_anomalies:{NANO}": "Darknet",
    "URL_{SEED}_y_normal:0,1,y_anomalous:Defacement,malware,phishing,spam,p_pollution:{POLL},n_train_anomalies:{NANO}": "URL",
    "DoH_{SEED}_y_normal:2,3,50,51,y_anomalous:Malicious,p_pollution:{POLL},n_train_anomalies:{NANO}": "DoH",
    "KDD_{SEED}_y_normal:icmp,tcp,udp,y_anomalous:anomaly,p_pollution:{POLL},n_train_anomalies:{NANO}": "KDD",
    "Census_{SEED}_y_normal:Male,Female,y_anomalous:Anomalous,p_pollution:{POLL},n_train_anomalies:{NANO}": "Census",
    "IDS_{SEED}_y_normal:0,1,2,3,4,5,y_anomalous:Bot,BruteForce,Infiltration,WebAttacks,p_pollution:{POLL},n_train_anomalies:{NANO}": "IDS",

} if not this_args.first_experiment else {
    # Exp. 1: performance vs number of normal classes
    "EMNIST_{SEED}_y_normal:1,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A",
    "EMNIST_{SEED}_y_normal:1,2,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-B",
    "EMNIST_{SEED}_y_normal:1,2,3,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-C",
    "EMNIST_{SEED}_y_normal:1,2,3,4,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-D",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-E",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,6,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-F",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,6,7,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-G",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,6,7,8,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:{NANO}": "A-H",
}


def get_path(basepath: Path, p_pollution: float, random_seed: int, n_anomalies: int, file_name: str, file_suffix: str = ".metric.csv"):
    out_path = basepath
    # There are subfolders based on the pollution and the random seed
    out_path /= f"{p_pollution}_{random_seed}"
    # The filename contains the random seed (bad design decision btw)
    parsed_name = file_name.replace("{SEED}", str(random_seed))
    # And also the pollution (honestly, this makes things harder)
    parsed_name = parsed_name.replace("{POLL}", str(p_pollution).replace(".", ""))
    parsed_name = parsed_name.replace("{NANO}", str(n_anomalies))
    out_path /= parsed_name
    out_path = out_path.with_suffix(file_suffix)

    return out_path


if __name__ == '__main__':
    # In the end, we want a DF with all results indexed by the contamination and the experiment IDs
    df_tot = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [P_POLLUTION, list(NAME_TO_ID.values())],
            names=["Cont.", "Exp."]
        )
    )
    df_out = df_tot.copy()
    df_out.index = pd.MultiIndex(levels=2*[[]], codes=2*[[]], names=["Method", "Metric"])

    # Go through all metric files
    for cur_pollution in P_POLLUTION:
        for cur_name, cur_id in NAME_TO_ID.items():

            # We open all metric files given their random seed
            all_metrics = []
            for cur_seed in this_args.random_seeds:
                cur_path = get_path(
                    basepath=this_args.model_path,
                    p_pollution=cur_pollution,
                    n_anomalies=this_args.n_anomalies,
                    random_seed=cur_seed,
                    file_name=cur_name
                )
                # List all files that are missing
                try:
                    all_metrics.append(pd.read_csv(cur_path, index_col=0))
                except FileNotFoundError:
                    print(f"Cannot find {cur_path}. Please check the path.")
                    continue

            # Once opened, we merge them
            pd_concat = pd.concat(all_metrics)
            concat_by_method = pd_concat.groupby(pd_concat.index)
            # We want everything in one series which will become a row in the final DF
            this_mean = concat_by_method.mean().stack()
            this_std = concat_by_method.std().stack()
            # Also we should add a new level to the MultiIndex to mark the mean and stddev
            this_mean = pd.DataFrame(this_mean)
            this_mean.loc[:, "type"] = "mean"
            this_mean = this_mean.set_index("type", append=True)
            this_std = pd.DataFrame(this_std)
            this_std.loc[:, "type"] = "std"
            this_std = this_std.set_index("type", append=True)

            # Add to the overall DF
            merged_metric = pd.concat([this_mean, this_std])
            df_tot[(cur_pollution, NAME_TO_ID[cur_name])] = merged_metric[0]

    all_baselines = df_tot.index.unique(0)
    # If not given otherwise, the AUC is our main metric
    if this_args.metric_name is None:
        df_tot = df_tot.reindex(["AUC", "AP"], axis=0, level=1)
    else:
        df_tot = df_tot.loc[(all_baselines, [this_args.metric_name]), :]
    # Reorder baselines
    level_name_reordered = BASELINE_METHODS
    # Reorder
    df_tot = df_tot.reindex(level_name_reordered, axis=0, level=0)

    # Round
    df_not_rounded = df_tot.copy()
    df_tot = df_tot.round(decimals=2)
    # Save if desired
    if this_args.out_path is not None:
        df_csv = df_not_rounded.copy()
        # Combine the MultiIndex for easier indexing afterwards
        df_csv.index = df_csv.index.to_series().apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}")
        df_csv = df_csv.transpose()
        # Save
        df_csv.to_csv(this_args.out_path)

    # Decision: let's build the LaTeX code here instead of using pgfplotstable & Co
    for cur_idx, cur_df in df_tot.groupby(level=[0, 1]):
        # Merge to "mean \pm stddev"
        this_latex = cur_df.iloc[0, :].map("{:,.2f}".format)
        if this_args.show_stddev:
            this_latex += " \\scriptscriptstyle \\pm " + cur_df.iloc[1, :].map("{:,.2f}".format)
        # Add the math environment
        this_latex = "$" + this_latex + "$"

        # Highest score should be black, rest gray
        max_per_column = df_tot.loc[(slice(None), cur_idx[1], "mean"), :].max(axis=0)
        is_max = cur_df.loc[cur_idx + ("mean",), :] == max_per_column
        this_latex.loc[is_max] = "\\color{black}" + this_latex.loc[is_max]
        # this_latex.loc[is_max] = "\\textbf{" + this_latex.loc[is_max] + "}"

        df_out.loc[cur_idx, :] = this_latex

    # Add p-value
    for cur_idx, cur_df in df_not_rounded.groupby(axis=0, level=[0, 1]):
        # Don't compare ARGUE to itself
        if cur_idx[0] == AD_NAME:
            continue

        # First group by baseline & metric, then by contamination level
        for cur_idx_2, cur_df_2 in cur_df.groupby(axis=1, level=0):
            # Compare the distribution of ARGUE to the baseline
            dist_argue = df_not_rounded.loc[(AD_NAME, cur_idx[1], "mean"), (cur_idx_2, slice(None))]
            dist_baseline = cur_df_2.loc[cur_idx + ("mean", ), :]
            # Calculate the p-value
            _, p_val = wilcoxon(x=dist_argue, y=dist_baseline)

            # Prepare for LaTeX
            p_val = round(p_val, ndigits=2)
            p_val = f"${p_val:,.2f}$"
            # Add to the output DF
            df_out.loc[cur_idx, (cur_idx_2, "p-val")] = p_val
            pass
    # Mark ARGUE's p-value by "-"
    df_out.loc[(AD_NAME, slice(None)), (slice(None), "p-val")] = "-"

    # Add average row
    for cur_idx, cur_df in df_not_rounded.groupby(axis=1, level=0):
        df_avg = cur_df.mean(axis=1).round(decimals=2)
        # We're only interested in the mean of the mean
        df_avg = df_avg.loc[(slice(None), slice(None), "mean")]
        # Add the math environment
        this_latex = "$" + df_avg.map("{:,.2f}".format) + "$"
        # Show the maximum
        all_max = []
        for max_idx, max_df in df_avg.groupby(axis=0, level=1):
            all_max.append(
                max_df.loc[(slice(None), [max_idx])] == max_df.max()
            )
        is_max = pd.concat(all_max)
        this_latex.loc[is_max] = "\\color{black}" + this_latex.loc[is_max]
        df_out[(cur_idx, "mean")] = this_latex

    # Convert to TeX
    df_out = df_out.sort_index(axis=1, level=0)
    if this_args.transpose:
        df_out = df_out.transpose()
    latex = df_out.to_latex(
        multicolumn_format="c", column_format=">{\\color{gray}}c "*(df_out.index.nlevels + len(df_out.columns)), escape=False
    )
    # Get back the backslash and math environments
    latex = latex.replace("\\textbackslash ", "\\").replace("\\$", "$").replace("0.", ".")

    if this_args.out_path is None:
        print(latex)
