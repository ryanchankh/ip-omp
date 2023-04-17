import json
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from rich.logging import RichHandler
import seaborn as sns
import typer

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()

TRIALS = 100
SETTINGS = {
    "dimensions": [(500, 800), (1600, 2400)],
    "sparsity": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    "noise_std": [0.0, 0.01, 0.1],
}

try:
    import cupy as np
except ModuleNotFoundError:
    logger.info("cupy installation not found. Falling back to numpy.")
    import numpy as np


def gen_dictionary(m, n):
    Phi = np.random.randn(m, n)
    return Phi / np.linalg.norm(Phi, axis=0)


def projection(Phi_t, perp=False):
    U, *_ = np.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.T

    if perp:
        return np.eye(P.shape[0]) - P

    return P


def generate_measurements_and_coeffs(Phi, p=0.01, noise_std=0.0):
    m, n = Phi.shape
    supp = np.random.rand(n) <= p
    x = np.zeros(n)
    x[supp] = np.random.randn(int(np.sum(supp)))
    return (Phi @ x + noise_std * np.random.randn(m)).reshape(-1, 1), x.reshape(-1, 1)


class Log:
    def __init__(self, debug=None):
        self.debug = set(debug or [])
        self.keys = []

    def log(self, key, value, context=None):
        if not hasattr(self, key):
            self.keys.append(key)
            setattr(self, key, [])

        if self.debug and context:
            print(f"[{context}] {key}: {value}")
        elif key in self.debug:
            print(f"{key}: {value}")

        getattr(self, key).append(deepcopy(value))

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys()}

    def __repr__(self):
        return f"Log({self.keys()})"


def ip_objective(Phi, y, indices=None):
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    Phi_projected_normalized = Phi_projected / np.linalg.norm(
        Phi_projected, axis=0
    ).reshape(1, -1)
    objective = np.abs(Phi_projected_normalized.T @ y)
    objective[indices] = -np.inf
    return objective


def omp_objective(Phi, y, indices=None):
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    return np.abs(Phi_projected.T @ y)


def omp_estimate_y(Phi, indices, y):
    Phi_t = Phi[:, indices]
    return projection(Phi_t, perp=False) @ y


def ip_estimate_y(Phi, indices, y):
    return omp_estimate_y(Phi, indices, y)


def omp_estimate_x(Phi, indices, y):
    Phi_t = Phi[:, indices]
    x_hat = np.zeros((Phi.shape[1], 1))
    x_hat[indices] = np.linalg.pinv(Phi_t) @ y
    return x_hat


def ip_estimate_x(Phi, indices, y):
    return omp_estimate_x(Phi, indices, y)


def ip(Phi, y, tol=1e-6, debug=False):
    log = Log(debug=debug)
    indices = []
    while True:
        objective = ip_objective(Phi, y, indices=indices)
        max_objective = objective.max()
        log.log("objective", max_objective)
        if np.abs(max_objective) < tol:
            break
        indices.append(np.argmax(objective).item())
        log.log("indices", indices)
        y_hat = ip_estimate_y(Phi, indices, y)
        log.log("y_hat", y_hat)
        x_hat = ip_estimate_x(Phi, indices, y)
        log.log("x_hat", x_hat)

    return log


def omp(Phi, y, tol=1e-6, debug=False):
    log = Log(debug=debug)
    indices = []
    while True:
        P = projection(Phi[:, indices], perp=True)
        residual = P @ y
        squared_error = residual.T @ residual
        if squared_error < tol:
            break
        objective = np.abs(Phi.T @ residual)
        log.log("objective", objective.max())
        indices.append(np.argmax(objective).item())
        log.log("indices", indices)
        y_hat = ip_estimate_y(Phi, indices, y)
        log.log("y_hat", y_hat)
        x_hat = omp_estimate_x(Phi, indices, y)
        log.log("x_hat", x_hat)

    return log


def recall(estimated, true):
    return len(set(estimated).intersection(set(true))) / len(true)


def precision(estimated, true):
    return len(set(estimated).intersection(set(true))) / len(estimated)


def mse(estimated, true):
    return np.mean((estimated - true) ** 2)


def run_experiment(
    experiment_number: int,
    m: int,
    n: int,
    s: float,
    noise_std: float,
    output_dir: Path,
    plot: bool = False,
):
    output_dir.mkdir(exist_ok=True)
    experiment_results_dir = output_dir / str(experiment_number)
    experiment_results_dir.mkdir(exist_ok=True)

    logger.info(
        f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}"
    )
    Phi = gen_dictionary(m, n)
    y, x = generate_measurements_and_coeffs(Phi, p=s, noise_std=noise_std)
    true_support = set(np.where(x.ravel() != 0)[0])
    y = y.reshape(-1, 1)
    logger.info("Running IP")
    log_ip = ip(Phi, y, debug=None)
    logger.info("Running OMP")
    log_omp = omp(Phi, y, debug=None)

    ip_recall = []
    omp_recall = []
    ip_precision = []
    omp_precision = []
    ip_mse_y = []
    omp_mse_y = []
    ip_mse_x = []
    omp_mse_x = []

    logger.info("Generating metrics for IP")
    for indices, x_hat, y_hat in zip(log_ip.indices, log_ip.x_hat, log_ip.y_hat):
        ip_recall.append(recall(indices, true_support))
        ip_precision.append(precision(indices, true_support))
        ip_mse_x.append(mse(x_hat, x))
        ip_mse_y.append(mse(y_hat, y))

    logger.info("Generating metrics for OMP")
    for indices, x_hat, y_hat in zip(log_omp.indices, log_omp.x_hat, log_omp.y_hat):
        omp_recall.append(recall(indices, true_support))
        omp_precision.append(precision(indices, true_support))
        omp_mse_x.append(mse(x_hat, x))
        omp_mse_y.append(mse(y_hat, y))

    if plot:
        sns.set()
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "Helvetica",
            }
        )

        plt.plot(omp_recall)
        plt.plot(ip_recall, "--")
        plt.legend(["OMP", "IP"])
        plt.xlabel("Iteration")
        plt.ylabel(
            r"$\frac{|\mathrm{supp}(\widehat{x}) \, \cap \, \mathrm{supp}(x)|}{|\mathrm{supp}(x)|}$",
            fontsize="x-large",
        )
        plt.title("Recall of estimated support")
        plt.savefig(experiment_results_dir / "recall.png", dpi=300)
        plt.close()

        plt.plot(omp_precision)
        plt.plot(ip_precision, "--")
        plt.legend(["OMP", "IP"])
        plt.xlabel("Iteration")
        plt.ylabel(
            r"$\frac{|\mathrm{supp}(\widehat{x}) \, \cap \, \mathrm{supp}(x)|}{|\mathrm{supp}(\widehat{x})|}$",
            fontsize="x-large",
        )
        plt.title("Precision of estimated support")
        plt.savefig(experiment_results_dir / "precision.png", dpi=300)
        plt.close()

        plt.plot(omp_mse_x)
        plt.plot(ip_mse_x, "--")
        plt.legend(["OMP", "IP"])
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|x - \widehat{x}\|_2^2$")
        plt.title("MSE of Sparse Code Estimate")
        plt.savefig(experiment_results_dir / "mse_x.png", dpi=300)
        plt.close()

        plt.plot(omp_mse_y)
        plt.plot(ip_mse_y, "--")
        plt.legend(["OMP", "IP"])
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|y - \widehat{y}\|_2^2$")
        plt.title("MSE of Measurement Estimate")
        plt.savefig(experiment_results_dir / "mse_y.png", dpi=300)
        plt.close()

    results = {
        "precision_ip": ip_precision,
        "precision_omp": omp_precision,
        "recall_ip": ip_recall,
        "recall_omp": omp_recall,
        "mse_x_ip": ip_mse_x,
        "mse_x_omp": omp_mse_x,
        "mse_y_ip": ip_mse_y,
        "mse_y_omp": omp_mse_y,
        "iters_ip": len(log_ip.indices),
        "iters_omp": len(log_omp.indices),
        "max_objective_ip": log_ip.objective,
        "max_objective_omp": log_omp.objective,
    }

    logger.info(f"Saving metrics to {experiment_results_dir / 'results.json'}")
    with open(experiment_results_dir / "results.json", "w") as f:
        json.dump(results, f)


def main(results_dir: Path, overwrite: bool = False, plot: bool = False):
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    for k, ((m, n), s, noise_std) in enumerate(
        product(SETTINGS["dimensions"], SETTINGS["sparsity"], SETTINGS["noise_std"])
    ):
        run_experiment(
            k,
            m,
            n,
            s,
            output_dir=results_dir,
            noise_std=noise_std,
            plot=plot,
        )


if __name__ == "__main__":
    typer.run(main)
