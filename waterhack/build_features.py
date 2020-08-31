import numpy as np
import pandas as pd


def data_load_dma(file_csv="data/processed/DMAVolumePressureWeather.csv"):

    # Load csv from location
    df = pd.read_csv(file_csv, sep=",", parse_dates=["Timestamp", "DateRaised"])
    # print(df.info())

    # Select few columns
    df = df[["DMA", "Timestamp", "PressureBar", "m3Volume"]]

    # We filter for one DMA only
    selected_DMA = "NEWSEVMA"
    df_filt = df[df.DMA == selected_DMA]
    df_filt["Date"] = pd.to_datetime(df_filt["Timestamp"]).dt.date
    df_filt["Time"] = pd.to_datetime(df_filt["Timestamp"]).dt.time
    df_filt["Hour"] = pd.to_datetime(df_filt["Timestamp"]).dt.hour

    # Daily volume
    df_Vd = df_filt.groupby("Date").mean()

    # Night volue
    df_Vn = df_filt[(df_filt["Hour"] >= 2) & (df_filt["Hour"] < 4)]
    df_Vn = df_Vn.groupby("Date").mean()

    # Merge both datasets
    df_solution = pd.merge(
        df_Vd, df_Vn, left_index=True, right_index=True, suffixes=("_Vd", "_Vn")
    )

    # Drop hour column
    df_solution = df_solution.drop(["Hour_Vd", "Hour_Vn"], axis=1)

    # Add holiday indicator
    df_solution.index = pd.to_datetime(df_solution.index)
    df_solution["is_holiday"] = np.where(df_solution.index.dayofweek >= 5, 1, 0)

    return df_solution
