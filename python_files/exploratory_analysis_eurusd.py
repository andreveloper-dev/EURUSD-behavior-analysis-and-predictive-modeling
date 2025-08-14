import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


def target_variable_analysis(df):
    target_column = 'eur_usd'

    print("Target Variable Analysis")
    print(f"Statistics of {target_column}:")
    print(df[target_column].describe())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(df.index, df[target_column], linewidth=1, alpha=0.8)
    axes[0, 0].set_title(f'Historical Price of {target_column}')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df[target_column].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'Distribution of {target_column}')
    axes[0, 1].set_xlabel('Price')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    returns = df[target_column].pct_change().dropna()
    axes[1, 0].plot(returns.index, returns, linewidth=0.5, alpha=0.7)
    axes[1, 0].set_title('Daily Returns')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Return')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Daily Returns')
    axes[1, 1].set_xlabel('Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Return Statistics")
    print(f"Average daily return: {returns.mean():.6f}")
    print(f"Daily volatility: {returns.std():.6f}")
    print(f"Annualized volatility: {returns.std() * np.sqrt(252):.6f}")
    print(f"Skewness: {returns.skew():.4f}")
    print(f"Kurtosis: {returns.kurtosis():.4f}")


def stationarity_analysis(timeseries, title):
    print(f"Stationarity Analysis: {title}")
    adf_result = adfuller(timeseries.dropna())
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")

    if adf_result[1] <= 0.05:
        print("Series is stationary according to ADF (reject H0)")
    else:
        print("Series is NOT stationary according to ADF (fail to reject H0)")

    kpss_result = kpss(timeseries.dropna())
    print(f"KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"p-value: {kpss_result[1]:.6f}")
    print("Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"\t{key}: {value:.3f}")

    if kpss_result[1] >= 0.05:
        print("Series is stationary according to KPSS (fail to reject H0)")
    else:
        print("Series is NOT stationary according to KPSS (reject H0)")
    print("\n" + "=" * 50 + "\n")


def time_series_decomposition_analysis(df, target_column):
    print("Time Series Decomposition")
    decomposition = seasonal_decompose(df[target_column].dropna(), model='additive', period=252)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    axes[0].plot(decomposition.observed, linewidth=1)
    axes[0].set_title('Original Series')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(decomposition.trend, linewidth=1, color='orange')
    axes[1].set_title('Trend Component')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(decomposition.seasonal, linewidth=1, color='green')
    axes[2].set_title('Seasonal Component')
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(decomposition.resid, linewidth=1, color='red')
    axes[3].set_title('Residuals')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def correlation_analysis(df, target_column='eur_usd'):
    print("Correlation Analysis")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    correlation_matrix = df_numeric.corr()
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    plt.show()

    print(f"Numeric variables analyzed: {len(numeric_columns)}")
    print(f"Included variables: {list(numeric_columns)}")
    target_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)
    print(f"Top 10 Correlations With {target_column}")
    print(target_correlations.head(10))

    print("Multicollinearity Detection")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((correlation_matrix.columns[i],
                                        correlation_matrix.columns[j],
                                        correlation_matrix.iloc[i, j]))

    if high_corr_pairs:
        print("Variable pairs with correlation > 0.8:")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No variable pairs found with correlation > 0.8")


def load_dataset_and_run_analysis():
    dataset_path = "dataset/datasets_to_use/df_completoV2.xlsx"
    target_column = 'eur_usd'
    df = pd.read_excel(dataset_path)
    decomposition = seasonal_decompose(df[target_column].dropna(), model='additive', period=252)

    returns = df[target_column].pct_change().dropna()
    stationarity_analysis(df[target_column], target_column)
    stationarity_analysis(returns, 'EUR/USD Returns')
    first_difference = df[target_column].diff().dropna()
    stationarity_analysis(first_difference, 'First Difference of EUR/USD')
    residuals = decomposition.resid.dropna()
    print("Decomposition Residual Statistics")
    print(f"Mean: {residuals.mean():.6f}")
    print(f"Standard deviation: {residuals.std():.6f}")
    print(f"Skewness: {residuals.skew():.4f}")
    print(f"Kurtosis: {residuals.kurtosis():.4f}")


load_dataset_and_run_analysis()
