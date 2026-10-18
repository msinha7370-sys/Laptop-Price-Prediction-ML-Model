# Laptop Price Prediction (ML)

Predicting laptop prices from their specifications (brand, screen, CPU, RAM, storage, GPU, OS, weight) with a regression model, in a single Jupyter notebook.

## What's in this repo

| File | Purpose |
|---|---|
| `Laptop Price Prediction ML Model.ipynb` | End-to-end analysis: cleaning, EDA, feature engineering, modeling |
| `laptop.csv` | Dataset: ~1,300 laptops, 11 feature columns plus `Price` (target) |
| `requirements.txt` | Python dependencies |

## Dataset

Each row is one laptop listing.

| Column | Example |
|---|---|
| Company | Apple, HP, Dell |
| TypeName | Ultrabook, Notebook, Gaming |
| Inches | 13.3 |
| ScreenResolution | `IPS Panel Retina Display 2560x1600` |
| Cpu | `Intel Core i5 7200U 2.5GHz` |
| Ram | `8GB` |
| Memory | `128GB SSD` |
| Gpu | `Intel HD Graphics 620` |
| OpSys | macOS, Windows 10 |
| Weight | `1.37kg` |
| **Price** | target (apparently Indian rupees) |

The data is a publicly circulated laptop-price dataset; see the original source for its licence terms.

## Approach

1. **Clean**: drop index columns, null rows and `?` placeholders; convert `Inches`, `Weight` and `Ram` to numbers.
2. **Explore**: distributions of price, screen size, weight and RAM; counts and price boxplots by company, type and resolution.
3. **Encode**: one-hot encode `Company`, `TypeName` and `OpSys`.
4. **Engineer features**: from `ScreenResolution` (width, height, IPS, Retina, touchscreen, Full HD, 4K, Quad HD+), from `Cpu` (clock speed, brand) and from `Gpu` (brand).
5. **Model**: 80/20 train/test split (`random_state=1`) and a `LinearRegression` baseline.

## Baseline results

Linear regression on the held-out test set (255 rows):

| Metric | Value |
|---|---|
| R² (train / test) | 0.741 / 0.780 |
| MAE | ≈ 12,849 |
| RMSE | ≈ 17,989 |

## Known limitations / planned improvements

- `Cpu_Type` is derived from the first word of the CPU string, so it is always `"Other"`; the CPU family (i3/i5/i7, Ryzen, ...) is never used.
- `Memory` is label-encoded, which imposes a meaningless order; parsing SSD/HDD sizes into numeric columns would be better.
- Price is right-skewed; modelling `log(Price)` should help.
- Only linear regression is tried; tree-based models (random forest, gradient boosting) are the obvious next step.
- The notebook was written in Google Colab. It reads `/content/laptop.csv` and uses `OneHotEncoder(sparse=False)`, which newer scikit-learn versions renamed to `sparse_output=False`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ML_CapsON_pro_laptop.ipynb
```

Change the data path in the notebook from `/content/laptop.csv` to `laptop.csv` when running locally.
