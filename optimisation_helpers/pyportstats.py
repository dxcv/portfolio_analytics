import pandas as pd 
import numpy as np 

def stats_summary(df, rf=0, scaling_fact=252):
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

