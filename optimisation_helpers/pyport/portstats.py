import pandas as pd 
import numpy as np 

__all__ = ['stats_summary']

def er(df, w, scaling_fact=252):
    """
    Calculates expected return.
    """

    ln_rets = (pd.DataFrame(np.log(df / df.shift(1))) * w).sum(axis=1)

    return np.exp(ln_rets.mean() * scaling_fact) - 1

def vol(df, w, scaling_fact=252):
    """
    Calculates portfolio vol.
    """

    ln_rets = (pd.DataFrame(np.log(df / df.shift(1))) * w).sum(axis=1)

    return ln_rets.std() * (scaling_fact ** 0.5)

def mdd(df, w, scaling_fact=252):
    """
    Calculates max drawdown.
    """

    ln_rets = (pd.DataFrame(np.log(df / df.shift(1))) * w).sum(axis=1)
    ret = (np.exp(ln_ret) - 1).add(1).cumprod()
    dd = ret.div(ret.cummax()).sub(1)

    return dd.min()

def stats_summary(df, rf=0, scaling_fact=252):
    """
    Calculates basic portfolio statistics.
    
    Parameters
    ----------
    df: DataFrame of portfolio prices
    rf: risk free rate, scalar
    scaling_fact: annualisation factor, scalar
    
    """

    yf = 365 / (max(df.index) - min(df.index)).days
    ln_rets = pd.DataFrame(np.log(df / df.shift(1)))

    res = []
    for col in ln_rets.columns:
        ln_ret = ln_rets[col]

        """Return and Ann. Return."""
        r = np.exp(ln_ret.sum()) - 1
        ar = (1+r) ** yf - 1

        """Historical max drawdown."""
        ret = (np.exp(ln_ret) - 1).add(1).cumprod()
        dd = ret.div(ret.cummax()).sub(1)
        mdd = dd.min()

        """Portfolio volatility."""
        pvol = ln_ret.std() * (scaling_fact ** 0.5)

        """Sharpe ratio target."""
        sr = (ar - rf) / pvol

        res.append([r, ar, pvol, sr, mdd])
    
    results = pd.DataFrame(res, columns=['cumu_return', 'ann_return', 'volatility', 'sharpe_ratio', 'max_drawdown'], index=ln_rets.columns).T

    return results

