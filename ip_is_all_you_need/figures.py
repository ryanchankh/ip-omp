from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

from ip_is_all_you_need.plots import get_phase_transition_data

matplotlib.rcParams.update({"font.size": 22})


c = pl.col
sns.set()
sns.set_context("talk")


def filter_df(
    df: pl.DataFrame, algorithm: Literal["ip", "omp", None] = None
) -> pl.DataFrame:
    if algorithm:
        df = df.filter(c("algorithm") == algorithm)
    df = df.with_columns(
        c("iter")
        .max()
        .over(["experiment_number", "trial", "algorithm"])
        .alias("max_iter")
    )
    return df.filter(c("iter") == c("max_iter"))


def early_termination(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            c("iter")
            .max()
            .over(["experiment_number", "trial", "algorithm"])
            .alias("max_iter")
        )
        .with_columns((c("max_iter") < c("sparsity") - 1).alias("early_term"))
        .groupby(["experiment_number", "trial", "algorithm"])
        .agg(
            c("m").first(),
            c("n").first(),
            c("sparsity").first(),
            c("early_term").mean(),
        )
    )


def get_phase_transition_data(
    df: pl.DataFrame, algorithm: Literal["ip", "omp"]
) -> pl.DataFrame:
    df_pt = (
        # filter to only the last iteration
        filter_df(df, algorithm=algorithm)
        # define success as relative reconstruction error < eps
        .with_columns(
            (c("mse_x") / (c("norm_x") ** 2) < 1e-14).alias("success"),
        )
        # for each experiment
        .groupby("experiment_number")
        # record the settings, success rate, and iou statistics
        .agg(
            c("m").first(),
            c("n").first(),
            c("measurement_rate").first(),
            c("sparsity").first(),
            c("noise_std").first(),
            c("iou").mean(),
            c("iou").quantile(0.05).alias("iou_lo"),
            c("iou").quantile(0.95).alias("iou_hi"),
            c("success").mean().alias("success_rate"),
            (c("mse_x") / c("norm_x") ** 2).mean().alias("mse_x_mean"),
            (c("mse_x") / c("norm_x") ** 2).median().alias("mse_x_median"),
            (c("mse_x") / c("norm_x") ** 2).quantile(0.05).alias("mse_x_05p"),
            (c("mse_x") / c("norm_x") ** 2).quantile(0.95).alias("mse_x_95p"),
            (c("mse_x") / c("norm_x") ** 2).quantile(0.25).alias("mse_x_25p"),
            (c("mse_x") / c("norm_x") ** 2).quantile(0.75).alias("mse_x_75p"),
            (c("mse_x") / c("norm_x") ** 2).std().alias("mse_x_std"),
        ).with_columns(
            (c("m") / c("n")).alias("measurement_rate"),
            (c("sparsity") / c("m")).alias("sparsity_rate"),
        )
    )
    return df_pt


def plot_phase_transition(df: pl.DataFrame, algorithm: Literal["ip", "omp"]) -> None:
    n = df["n"][0]
    df_pt = get_phase_transition_data(df, algorithm)
    tbl = (
        df_pt.sort(by=["m", "sparsity"], descending=[True, False])
        .pivot(
            values="success_rate",
            index="m",
            columns="sparsity",
            aggregate_function="first",
        )
        .to_pandas()
    )
    tbl = tbl.set_index("m", drop=True)
    sns.heatmap(tbl)
    plt.xlabel("Sparsity $s$")
    plt.ylabel("Number of measurements $m$")

    plt.title(f"Phase Transition for {algorithm.upper()} (n={n})")


def plot_probability_curve(df: pl.DataFrame, save_file: Path | None = None) -> None:
    df_pt_omp = get_phase_transition_data(df, "omp")
    df_pt_ip = get_phase_transition_data(df, "ip")
    n = df_pt_omp["n"][0]

    labels = []
    lines = []
    plt.figure()
    for s in sorted(df_pt_omp["sparsity"].unique()):
        labels.append(f"$s$={s}")
        df_pt_at_s_omp = df_pt_omp.filter(c("sparsity") == s).sort("m")
        df_pt_at_s_ip = df_pt_ip.filter(c("sparsity") == s).sort("m")
        cur_lines = plt.plot(df_pt_at_s_omp["m"], df_pt_at_s_omp["success_rate"])
        lines.append(cur_lines[0])
        cur_lines = plt.plot(
            df_pt_at_s_ip["m"],
            df_pt_at_s_ip["success_rate"],
            "o",
            fillstyle="none",
            color=cur_lines[0].get_color(),
        )
        plt.xlabel("Number of measurements $m$")
        plt.ylabel("Probability of exact recovery")
        plt.title(f"Number of dictionary atoms $n$={n}")
        plt.grid("on")

    plt.legend(lines, labels, bbox_to_anchor=(1.32, 0.75))

    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.savefig(save_file, bbox_inches="tight")


def plot_probability_curves(
    df_small: pl.DataFrame, df_large: pl.DataFrame, save_file: Path | None = None
) -> None:
    _, axs = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for k, df in enumerate([df_small, df_large]):
        ax = axs[k]
        df_pt_omp = get_phase_transition_data(df, "omp")
        df_pt_ip = get_phase_transition_data(df, "ip")
        n = df_pt_omp["n"][0]

        labels = []
        lines = []
        for s in sorted(df_pt_omp["sparsity"].unique()):
            labels.append(f"$s$={s}")
            df_pt_at_s_omp = df_pt_omp.filter(c("sparsity") == s).sort("m")
            df_pt_at_s_ip = df_pt_ip.filter(c("sparsity") == s).sort("m")
            cur_lines = ax.plot(df_pt_at_s_omp["m"], df_pt_at_s_omp["success_rate"])
            lines.append(cur_lines[0])
            cur_lines = ax.plot(
                df_pt_at_s_ip["m"],
                df_pt_at_s_ip["success_rate"],
                "o",
                fillstyle="none",
                color=cur_lines[0].get_color(),
            )
            ax.set_xlabel("Number of measurements $m$")
            if k == 0:
                ax.set_ylabel("Probability of exact recovery")
            ax.set_title(f"Number of dictionary atoms $n$={n}")
            ax.grid("on")

        ax.legend(lines, labels, loc="lower right")

    plt.subplots_adjust(wspace=0.05)

    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.savefig(save_file, bbox_inches="tight")


def main(
    small_result_path: Path,
    large_result_path: Path,
    save_file: Path,
    max_m_small: int | None = None,
    max_m_large: int | None = None,
    together: bool = False,
) -> None:
    df_small = pl.read_parquet(small_result_path)
    df_large = pl.read_parquet(large_result_path)

    if max_m_small:
        df_small = df_small.filter(pl.col("m") <= max_m_small)

    if max_m_large:
        df_large = df_large.filter(pl.col("m") <= max_m_large)

    if together:
        plot_probability_curves(df_small, df_large, save_file=save_file)
    else:
        plot_probability_curve(
            df_small,
            save_file=save_file.parent / (f"{save_file.stem}_small" + save_file.suffix),
        )
        plot_probability_curve(
            df_large,
            save_file=save_file.parent / (f"{save_file.stem}_large" + save_file.suffix),
        )


if __name__ == "__main__":
    typer.run(main)
