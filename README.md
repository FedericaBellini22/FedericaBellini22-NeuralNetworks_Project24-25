# NeuralNetworks_Project24-25

# TSMixer: An All-MLP Architecture for Time Series Forecasting

This repository contains a PyTorch-based reimplementation of the core method described in the paper:

> **"TSMixer: An All-MLP Architecture for Time Series Forecasting"**  
> Si-An Chen, Chun-Liang Li, Nathanael C. Yoder, Sercan Ö. Arık, Tomas Pfister (Google Research, 2023)  
> [Paper PDF](https://arxiv.org/pdf/2303.06053v5) | [Official code](https://github.com/google-research/google-research/tree/master/tsmixer)

---

## 🧠 Theoretical Background

Traditional deep learning models for time series forecasting often rely on recurrent or attention-based architectures (e.g., LSTMs, Transformers). However, recent work has shown that even simple **linear models** can outperform more complex models on long-term forecasting tasks, especially when cross-variate dependencies are weak or noisy.

**TSMixer** is based on the idea that:
- **Temporal patterns** (such as trends or seasonality) can be captured by **time-step-dependent models**.
- **Cross-variate dependencies** can be learned using feature-wise transformations.
- Using **multi-layer perceptrons (MLPs)** instead of attention or recurrence leads to a much simpler and computationally efficient architecture.

The TSMixer architecture alternates two types of MLP blocks:
- **Time-Mixing MLP**: Captures temporal dependencies by operating along the time axis (shared across all features).
- **Feature-Mixing MLP**: Captures cross-variable interactions by operating along the feature axis (shared across time steps).

Key components:
- Residual connections
- Layer normalization
- Dropout
- ReLU non-linearity

This approach generalizes the **MLP-Mixer** architecture (originally used in vision) to the time series domain, enabling effective forecasting with reduced model complexity.

---

## 📌 Project Structure

- `TSmixer.ipynb`: Main notebook with full code, including:
  - Data loading and preprocessing
  - Custom PyTorch Dataset class for ETTh1
  - TimeMixing and FeatureMixing blocks
  - Full TSMixer model
  - Training, evaluation, metrics
  - Model checkpointing and saving results

- `training_results.json`: Stores final evaluation metrics.

---

## 📊 Dataset

The model is trained and evaluated on the **ETTh1** dataset (Electricity Transformer Temperature) provided in the original paper.

- Source: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
- File used: `ETTh1.csv`
- Preprocessing:
  - Normalization (mean/std)
  - Sliding window for input/output pairs
- Splitting:
  - Train: 70%
  - Validation: 20%
  - Test: 10%

---

## 📈 Main Results

The following metrics were computed on the **validation set**, after training on the ETTh1 dataset:

| Metric         | Value     |
|----------------|-----------|
| MAE (Val set)  | 0.6793 |
| MSE (Val set)  | 0.8726 |
| Train Loss     | 0.2357 |
| Val Loss       | 1.0789 |

> Note: These results are based on a simplified reimplementation of TSMixer and may differ from those reported in the original paper.

### 🔍 Comparison with Original Paper

In the original TSMixer paper, the authors report a MAE of **0.334** and MSE of **0.183** on the ETTh1 dataset (forecasting horizon = 96).  
Our reimplementation yields higher error values, which is expected due to the following reasons:

- This project uses a **simplified version** of TSMixer (e.g., no Reversible Instance Normalization or temporal projection layers).
- I trained the model on limited computational resources and for a smaller number of epochs.
- No hyperparameter tuning was performed beyond default values.

Despite these differences, the model produces meaningful forecasts and demonstrates a correct implementation of the core TSMixer architecture.

---

## 💻 How to Run

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/your-username/TSMixer-ETTh1.git
cd TSMixer-ETTh1
pip install torch pandas numpy torchmetrics
```

### 2. Launch the notebook
```bash
jupyter notebook TSmixer.ipynb
```
---

## ⚠️ Notes
  - This is a simplified reimplementation for educational purposes.
  - The original paper explores additional extensions (e.g., auxiliary features, M5 dataset), which are not included here.
  - No code from the official repository was copied. Only the paper was used as reference.
  - Some components (e.g., temporal projection layers or advanced normalization) were intentionally simplified to reduce training time and hardware requirements.

---

## 📚 References
  - Chen, S.-A., Li, C.-L., Yoder, N. C., Arık, S. Ö., & Pfister, T. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting. Transactions on Machine Learning Research. arXiv:2303.06053
- Dataset: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
