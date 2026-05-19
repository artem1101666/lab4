import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def download_data():
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    df = pd.read_csv(url)
    df.to_csv("boston_raw.csv", index=False)
    print(f"Downloaded {len(df)} rows")
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)

    print("Columns:", df.columns.tolist())
    print("Shape before cleaning:", df.shape)

    # Удаляем дубликаты
    df = df.drop_duplicates()

    # Удаляем строки с пропусками
    df = df.dropna()

    # Удаляем выбросы по целевой переменной medv (цена дома)
    # Здравый смысл: цена не может быть <= 0
    df = df[df['medv'] > 0]

    # Удаляем выбросы по IQR для числовых признаков
    num_cols = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    for col in num_cols:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Признак: высокая криминогенность (бинарный)
    df['high_crime'] = (df['crim'] > df['crim'].median()).astype(int)

    # Признак: близость к центру (dis < медианы)
    df['near_center'] = (df['dis'] < df['dis'].median()).astype(int)

    # Признак: tax_per_room
    df['tax_per_room'] = df['tax'] / (df['rm'] + 1e-5)

    df = df.reset_index(drop=True)

    print("Shape after cleaning:", df.shape)
    print("Sample:\n", df.head(3))

    df.to_csv('df_clear.csv', index=False)
    print("Saved to df_clear.csv")
    return True

download_data()
clear_data("boston_raw.csv")