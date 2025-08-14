import os
import re
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

def load_dataset(datasets_folder, output_folder):
    csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
    filtered_dataframes = {}
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2025-05-31")
    date_formats_by_file = {
        "vix_history.csv": "%m/%d/%Y",
        "tasa_de_desempleo_usa.csv": "%Y-%m-%d"
    }

    def process_datasets():
        for file in csv_files:
            path = os.path.join(datasets_folder, file)
            try:
                df = pd.read_csv(path)
                date_column = next((col for col in df.columns if 'fecha' in col.lower() or 'date' in col.lower()), None)
                if date_column is None:
                    continue
                date_format = date_formats_by_file.get(file.lower())
                if date_format:
                    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
                else:
                    df[date_column] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=True)

                df = df.rename(columns={date_column: 'date'})
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                value_columns = ['Último', 'CLOSE', 'Actual', 'tasa_desempleo_usa']
                value_column = next((col for col in df.columns if col in value_columns), None)
                if value_column is None:
                    continue
                series_name = extract_series_name(file)
                df = df[['date', value_column]].copy()
                df = df.rename(columns={value_column: series_name})
                df = df.sort_values(by='date')
                filtered_dataframes[series_name] = df
            except Exception as e:
                print(f"Error processing {file}: {e}")

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='2010-01-01', end_date='2025-05-31')
    trading_days = schedule.index
    monthly_indicators = ['tasa_interes_bce', 'tasa_interes_fed', 'tasa_de_desempleo_usa']

    process_datasets()

    for indicator in monthly_indicators:
        if indicator in filtered_dataframes:
            original_df = filtered_dataframes[indicator]
            expanded_df = expand_monthly_values(original_df, indicator, trading_days)
            expanded_df[indicator] = expanded_df[indicator].ffill()
            filtered_dataframes[indicator] = expanded_df

    complete_df = merge_with_trading_days(filtered_dataframes, trading_days)

    object_columns = ['dow_jones', 'eur_usd', 'indice_dolar', 'indice_euro', 'nasdaq',
                      'oro', 'petroleo_brent', 'petroleo_crudo_wti',
                      'rendimiento_bonos_10_eeuu', 'rendimiento_bonos_2_eeuu',
                      's_p_500', 'tasa_interes_bce', 'tasa_interes_fed']

    for col in object_columns:
        complete_df[col] = complete_df[col].str.replace('.', '', regex=False)
        complete_df[col] = complete_df[col].str.replace(',', '.', regex=False)
        complete_df[col] = pd.to_numeric(complete_df[col], errors='coerce')

    complete_df.dropna(inplace=True)
    complete_df['RSI'] = calculate_rsi(complete_df, column='eur_usd', period=14)
    complete_df['SMA'] = calculate_sma(complete_df, column='eur_usd', period=14)
    complete_df['EMA'] = calculate_ema(complete_df, column='eur_usd', period=14)
    complete_df['momentum'] = calculate_momentum(complete_df, column='eur_usd', period=10)
    macd_df = calculate_macd(complete_df, column='eur_usd')
    complete_df = pd.concat([complete_df, macd_df], axis=1)
    complete_df.dropna(inplace=True)

    save_dataset(complete_df, output_folder)

def save_dataset(df, folder):
    output_path = os.path.join(folder, 'df_completoV2.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Saved to: {output_path}")

def extract_series_name(filename):
    name = filename.lower().replace("datos_historicos_", "").replace(".csv", "")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name

def expand_monthly_values(df, column_name, trading_days):
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    trading_df = pd.DataFrame({'date': trading_days})
    trading_df['year_month'] = trading_df['date'].dt.to_period('M')
    expanded_df = trading_df.merge(df[['year_month', column_name]], on='year_month', how='left')
    expanded_df = expanded_df[['date', column_name]]
    print(f"Expanded monthly: {column_name} ({df.shape[0]} → {expanded_df.shape[0]} rows)")
    return expanded_df

def merge_with_trading_days(dataframes_dict, trading_days):
    merged_df = pd.DataFrame({'date': pd.to_datetime(trading_days)})
    for name, df in dataframes_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if 'date' not in df.columns:
            continue
        df = dataframes_dict.get(name, pd.DataFrame()).copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        merged_df = merged_df.merge(df, on='date', how='left')
        print(f"Merged: {name} ({df.shape[0]} rows)")
    return merged_df

def calculate_rsi(data, column='close', period=14):
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, column='close', period=14):
    sma = data[column].rolling(window=period, min_periods=1).mean()
    return sma

def calculate_ema(data, column='close', period=14):
    ema = data[column].ewm(span=period, adjust=False).mean()
    return ema

def calculate_momentum(data, column='close', period=10):
    momentum = data[column] - data[column].shift(period)
    return momentum

def calculate_macd(data, column='close', fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[column].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    })

datasets_folder = "dataset/datasets_to_load"
output_folder = "dataset/datasets_to_use"
load_dataset(datasets_folder, output_folder)
