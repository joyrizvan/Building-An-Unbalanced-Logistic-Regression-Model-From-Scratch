import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(df: pd.DataFrame, ind: list, dep: list, test_size: float = 0.2):
    y = df[dep]
    x = df[ind]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=42
    )
    train_dataset = pd.concat([x_train, y_train], axis=1)
    test_dataset = pd.concat([x_test, y_test], axis=1)
    return train_dataset, test_dataset
