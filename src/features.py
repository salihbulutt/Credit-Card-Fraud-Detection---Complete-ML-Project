import numpy as np
import pandas as pd

def add_log_amount(df):
    df["log_amount"] = np.log1p(df["Amount"])
    return df
