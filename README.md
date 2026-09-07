# Latent Graph Neural Networks for Clinical Medication Prediction

This repository hosts the Proof of Concept (POC) code for predicting patient medication states using clinical progression curves. The project utilizes PyTorch and Graph Neural Networks (GNNs) to perform multi-label classification on clinical datasets, learning latent patient similarity graphs in the process.

### Overview
The core of this POC evaluates how well patient clinical data can predict the presence of specific medications (e.g., predicting up to 7 medication classes). It tests standard neural networks against a custom Graph Convolutional Network (GCN) approach that dynamically infers the adjacency matrix between batch samples based on feature similarity. 

### Key Features
*   **Multi-Dataset Support**: Configured to process clinical data from ADNI and PPMI cohorts.
*   **Latent Graph Learning**: Implements a `BatchLatentGraphGNN` using `DenseGCNConv` to dynamically construct and process patient similarity graphs.
*   **Robust Evaluation**: Outputs comprehensive classification metrics including Macro F1, Precision, Recall, AUROC, AUPRC, and Mean Average Precision (mAP).
*   **Custom Data Loaders**: Features collate functions designed to handle time-series clinical curves, right-censoring, and dynamic padding.

---

### Prerequisites and Dependencies

To run this POC, you will need a Python environment with the following primary libraries installed:
*   `torch` (with CUDA support recommended)
*   `torch_geometric` (version 2.6.0)
*   `pandas` and `numpy`
*   `scikit-learn` and `xgboost`
*   `torchtuples`
*   **AutoCurve**: A custom internal module (`autocurve.*`) handling survival evaluation, data parsing, and baseline clinical models.

### Data Preparation
The script expects access to structured clinical data directories. By default, the code looks for a medications CSV file formatted like ADNI's data:
> `clinical/BACKMEDS_29Jun2026.csv`

Ensure your environment variables or path configurations (`DATASET_DIR`) point to the correct secure workspace containing the parsed dataset curves. 

### Usage and Configuration
The main execution pipeline is configured via variables at the top of the script. Before running, ensure you update the environment paths based on your host machine:

1.  **Set the Dataset**: Change `dataset_name` to `"ADNI"` or `"PPMI"`.
2.  **Adjust Paths**: The code checks the hostname (e.g., `frida` or `vega`/deucalion) to automatically route `DATASET_DIR` and `ENV_DIR`. You will need to update these hardcoded paths to match your local or cluster environment.
3.  **Run the Training L
