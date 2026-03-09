import pandas as pd

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    # convert date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # create time features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['Date'].dt.dayofweek

    # drop original date
    df = df.drop(columns=['Date'])

    return df