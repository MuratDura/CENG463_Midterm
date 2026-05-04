# Machine Learning Experiment Suite

This project contains independent Python scripts that run experiments for different machine learning tasks.

## Project Structure

- `q1.py`: Regression on California Housing with feature engineering, cross-validation, and hold-out evaluation
- `q2.py`: Extreme class-imbalance classification with cost-sensitive learning, resampling, calibration, and threshold analysis
- `q3.py`: Dimensionality reduction comparison (PCA, Kernel PCA, t-SNE, UMAP, Autoencoder)
- `q4.py`: Clustering methods (K-Means, GMM, DBSCAN, Agglomerative), model selection, stability, and ensemble clustering
- `q5.py`: CIFAR experiments with MLP/CNN/transfer learning, hyperparameter search, interpretability, and adversarial evaluation

Each script writes its own outputs and figures to the project root or the relevant `q*_figures` directory.

## How The System Works

This repository is not a web service; it is a script-based experiment workflow.

1. Install the required dependencies.
2. Run the target script directly (`q1.py` ... `q5.py`).
3. The script loads its dataset (OpenML / torchvision / Hugging Face / dummy data).
4. Preprocessing, training, and evaluation steps are executed.
5. Results are saved as CSV, PNG, and some text files.

Scripts are independent, so you can run them individually or sequentially.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python q1.py
python q2.py
python q3.py
python q4.py
python q5.py --fast
```

Example options for `q5.py`:

```bash
python q5.py --dataset cifar100
python q5.py --data-source huggingface
python q5.py --skip-optuna --fast
python q5.py --dummy-data --fast
```

## Outputs

- Main metric CSV files: `q*_metrics_summary.csv`, `model_*`, and `q2_*` files
- Figures: `eda_figures/`, `q2_figures/`, `q3_figures/`, `q4_figures/`, `q5_figures/`
- Extra text: `q5_discussion.txt` (updated when Q5 is run)

## Notes

- `FAST_MODE` / `--fast` is intended for shorter test runs.
- Full runs can require significantly more time and memory.
- If a GPU is available, `q5.py` training is much faster.
