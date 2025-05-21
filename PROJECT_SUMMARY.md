# Project Summary: Fraud Detection Workflow

This project is a robust, modular, and reproducible workflow for fraud detection using Python. It includes data processing, model training, evaluation, and configuration management. All parameters and paths are centralized in a config file for maintainability. The workflow supports both script and notebook-based development, and is designed for easy extension and deployment.

## File-by-File Summary

### config.py
Central configuration file for the entire project. Contains all data paths, model paths, feature engineering settings, categorical and model features, target column, model parameters, and UI settings (e.g., progress bar). Imported by all scripts and notebooks to ensure consistency and reproducibility.

### model.py
Main script for data processing, model training, and model saving. Loads training data, processes features, encodes categorical variables, trains a decision tree classifier, and saves the trained model (along with features and target column) to disk. Uses logging and tqdm progress bar for user feedback. All parameters and paths are imported from `config.py`.

### test_model.py
Script for loading the trained model and evaluating it on test data. Loads the model and test dataset, processes and encodes features, makes predictions, and outputs evaluation metrics (confusion matrix, precision, recall, F1-score). Uses logging, slow_print for user feedback, and imports all settings from `config.py` for consistency.

### nyakamwizi.ipynb
Jupyter notebook for interactive data exploration, feature engineering, model training, and evaluation. Mirrors the logic in `model.py` but allows for step-by-step experimentation and visualization. Uses `config.py` for all parameters and paths, ensuring consistency with the scripts.

### model.pkl
Serialized trained model file. Contains the trained decision tree model, the list of features used, and the target column. Loaded by `test_model.py` for making predictions.

### requirements.txt
Lists all Python dependencies required for the project. Includes packages like scikit-learn (pinned to version 1.2.2 for model compatibility), pandas, tqdm, and others as needed. Used to set up the Python environment for reproducibility.

### README.md
Project documentation. (Assumed) Instructions for setup, usage, and project overview.

### tests.txt
(Assumed) May contain test cases, notes, or manual test instructions for the project.

### __pycache__/
Python bytecode cache directory. Contains compiled `.pyc` files for faster module loading.

---

Your project is well-structured for maintainability, reproducibility, and ease of use. All scripts and notebooks are parameterized via `config.py`, and you have both command-line and notebook interfaces for model evaluation and prediction. The workflow supports robust experimentation, deployment, and demonstration.
