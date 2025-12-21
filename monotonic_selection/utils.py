import pandas as pd

#論理積
def min(x: pd.Series, y: pd.Series) -> pd.Series:
    return pd.concat([x, y], axis=1).min(axis=1)

#代数積
def algebraic(x: pd.Series, y: pd.Series) -> pd.Series:
    return x*y

#Dobois-Pradeのt-norm
def dubois(x: pd.Series, y: pd.Series) -> pd.Series:
    z = 0.5
    return x*y/pd.concat([x, y, pd.Series(z, index=x.index)], axis=1).max(axis=1)

