# Medical Insurance Cost Prediction — ANN (Regression)

A from-scratch **Artificial Neural Network** built with pure **NumPy** to predict individual medical insurance charges based on demographic and lifestyle features.

---

## Dataset

| Property      | Value                              |
|---------------|------------------------------------|
| File          | `insurance.csv`                    |
| Rows          | 1,338                              |
| Features      | 6 input + 1 target (`charges`)     |
| Source        | [Kaggle – Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance) |

### Features

| Column     | Type        | Description                                 |
|------------|-------------|---------------------------------------------|
| `age`      | Numerical   | Age of the primary beneficiary              |
| `sex`      | Categorical | `male` / `female`                           |
| `bmi`      | Numerical   | Body Mass Index                             |
| `children` | Numerical   | Number of dependents covered                |
| `smoker`   | Categorical | `yes` / `no`                                |
| `region`   | Categorical | Residential region in the US (4 values)     |
| `charges`  | **Target**  | Individual medical costs billed (USD)       |

---

## Data Preprocessing

### 1. Missing Values
```
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
```
No missing values found — no imputation required.

### 2. Categorical Encoding
| Column   | Method          | Result                          |
|----------|-----------------|---------------------------------|
| `sex`    | Label Encoding  | `female=0`, `male=1`            |
| `smoker` | Label Encoding  | `no=0`, `yes=1`                 |
| `region` | One-Hot Encoding| 4 new binary columns created    |

After encoding the feature matrix has **9 columns**.

### 3. Feature Scaling
`StandardScaler` (zero mean, unit variance) applied to:
- All input features `X`
- Target variable `y` (charges) — scaled for stable gradient flow, inverse-transformed before evaluation

---

## Train-Test Split

```
Total samples : 1,338
Training set  : 1,070  (80%)
Test set      :   268  (20%)
Random state  : 42
```

---

## ANN Architecture

```
Input Layer    →  9 neurons  (one per feature)
Hidden Layer 1 →  64 neurons  [ReLU]
Hidden Layer 2 →  32 neurons  [ReLU]
Hidden Layer 3 →  16 neurons  [ReLU]
Output Layer   →  1 neuron   [Linear]  ← regression output
```

| Hyperparameter   | Value  |
|------------------|--------|
| Learning Rate    | 0.01   |
| Epochs           | 500    |
| Loss Function    | MSE    |
| Optimiser        | SGD (full batch) |
| Regularisation   | L2 (λ = 0.001)   |
| Weight Init      | He Initialisation |

The network is implemented **entirely in NumPy** — no TensorFlow, PyTorch, or Keras required. Forward pass uses manual matrix multiplication; backward pass uses full chain-rule backpropagation with L2 regularisation on all weight matrices.

---

## Training Progress

| Epoch | MSE Loss |
|-------|----------|
| 100   | 0.42209  |
| 200   | 0.26963  |
| 300   | 0.22643  |
| 400   | 0.20507  |
| 500   | 0.19174  |

Loss steadily decreased over 500 epochs, confirming that the network is learning.

---

## Evaluation Results (Test Set)

| Metric | Value      |
|--------|------------|
| **MAE**  (Mean Absolute Error)     | **$3,681.19** |
| **MSE**  (Mean Squared Error)      | 30,140,921.62 |
| **RMSE** (Root Mean Squared Error) | **$5,490.07** |
| **R²**   (Coefficient of Determination) | **0.8059** |

---

## Observations & Analysis

### What worked well
- **Smoker** is the most influential feature — smokers' charges are 3–4× higher, and the model captures this clearly.
- **R² = 0.8059** means the model explains ~80.6% of variance in medical costs — a strong result for a small dataset without feature engineering.
- He initialisation + L2 regularisation kept gradients stable across 500 epochs with no divergence.

### Limitations
- A small subset of high-cost outliers (charges > $50,000) pulls RMSE higher than MAE, indicating right-skewed prediction errors.
- Without dropout or early stopping, there is a mild overfitting risk after epoch ~350; adding these would further improve generalisation.
- The dataset is small (1,338 rows); a deeper network or more epochs may not help without more data.

### Potential Improvements
- Apply **log-transform** on `charges` to reduce right skew before scaling.
- Add **dropout layers** (rate 0.2–0.3) between hidden layers.
- Use **Adam optimiser** instead of vanilla SGD for faster convergence.
- Experiment with deeper networks or **batch normalisation**.

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn

# Run the ANN
python ann_insurance.py
```

---

## Repository Structure

```
insurance-ann/
├── insurance.csv          # Dataset
├── ann_insurance.py       # Full ANN implementation
└── README.md              # This file
```

---

## Dependencies

```
numpy
pandas
scikit-learn
```
