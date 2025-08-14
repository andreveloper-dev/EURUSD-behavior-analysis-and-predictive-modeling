# EURUSD Behavior Analysis and Predictive Modeling

This project analyzes and models the behavior of the EUR/USD exchange rate using historical data, time series analysis, and machine learning/deep learning techniques. The goal is to forecast currency movements and support data-driven decision-making in the financial domain.

## Project Structure

```
.gitignore
README.md
Requirements.txt
dataset/
  datasets_to_load/
    Datos_historicos_dow_jones.csv
    Datos_historicos_EUR_USD.csv
    Datos_historicos_indice_dolar.csv
    Datos_historicos_indice_euro.csv
    Datos_historicos_nasdaq.csv
    Datos_historicos_oro.csv
    ...
  datasets_to_use/
    df_completoV2.xlsx
models/
  lstm_model.h5
notebooks/
  EURUSD-behavior-analysis-and-predictive-modeling.ipynb
python_files/
  clean_and_preprocessing.py
  exploratory_analysis_eurusd.py
  models_creation/
    LSTM.py
```

## Main Features

- **Data Cleaning & Preprocessing:** Automated scripts to clean, merge, and preprocess multiple financial datasets.
- **Exploratory Data Analysis (EDA):** In-depth analysis of EUR/USD price behavior, stationarity, decomposition, outlier detection, and correlation analysis.
- **Technical Indicators:** Calculation of RSI, SMA, EMA, MACD, and momentum for EUR/USD.
- **Predictive Modeling:** Implementation of LSTM, RNN, Random Forest, and XGBoost models for forecasting.
- **Model Evaluation:** Metrics such as MAE, RMSE, R², and visual comparison of predictions vs. real values.
- **Feature Selection:** Use of SelectKBest, PCA, and correlation analysis to select relevant features.

## How to Run

1. **Install dependencies:**
   ```
   pip install -r Requirements.txt
   ```

2. **Prepare datasets:**
   - Place raw CSV files in `dataset/datasets_to_load/`.
   - Run preprocessing scripts to generate `df_completoV2.xlsx` in `dataset/datasets_to_use/`.

3. **Run notebooks:**
   - Open [notebooks/EURUSD-behavior-analysis-and-predictive-modeling.ipynb](notebooks/EURUSD-behavior-analysis-and-predictive-modeling.ipynb) for the main workflow.

4. **Train models:**
   - Use scripts in [python_files/models_creation/](python_files/models_creation/) or run cells in the notebook.

## Notebooks

- [EURUSD-behavior-analysis-and-predictive-modeling.ipynb](notebooks/EURUSD-behavior-analysis-and-predictive-modeling.ipynb): Main notebook for data analysis, feature engineering, and modeling.

## Scripts

- [python_files/clean_and_preprocessing.py](python_files/clean_and_preprocessing.py): Data cleaning and preprocessing.
- [python_files/exploratory_analysis_eurusd.py](python_files/exploratory_analysis_eurusd.py): EDA and visualization.
- [python_files/models_creation/LSTM.py](python_files/models_creation/LSTM.py): LSTM model implementation.

## Models

- Trained models are saved in [models/](models/), e.g., `lstm_model.h5`.

## Requirements

See [Requirements.txt](Requirements.txt) for all dependencies.

## License

MIT License.

## Authors

- Andres Felipe Moreno Calle

## References

- Historical financial data from various sources.
- Machine learning and deep learning libraries: scikit-learn, TensorFlow, Keras, XGBoost.
