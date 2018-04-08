import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import time
import datetime as dt

def summary_stats(w, df, rf=0, scaling_fact=252):
    """Calculates summary statistics needed for optimisation target."""
    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]

    ln_ret = np.log(df / df.shift(1))

    """Expected return target (based on historical mean)."""
    er = np.exp(np.dot(w, ln_ret.mean()) * scaling_fact) - 1

    """Historical max drawdown."""
    ret = ((np.exp(ln_ret) - 1) * w).sum(axis=1).add(1).cumprod()
    dd = ret.div(ret.cummax()).sub(1)
    mdd = dd.min()

    """Marginal risk, risk contribution, and risk parity target."""
    mvar = np.dot(w, ln_ret.cov().T) * w * scaling_fact
    pvar = sum(mvar)
    r_con = mvar / pvar
    rp_target = sum((pvar/len(w) - mvar)**2) * 10000 #Scale up by 10000 to make the optimisation easier

    """Portfolio volatility target and diversification ratio target."""
    pvol_target = pvar ** 0.5
    dr_target = sum((w * ln_ret.std() * scaling_fact ** 0.5)) / pvol_target

    """Sharpe ratio target."""
    sr = (er - rf) / pvol_target

    results = {
        'exp_return': er,
        'volatility': pvol_target,
        'sharpe_ratio': sr,
        'max_drawdown': mdd,
        'risk_contribution': r_con,
        'risk_parity': rp_target,
        'diversification_ratio': dr_target
    }

    return results

"""Objective functions for portfolio optimisation."""

def max_er(w, df, rf=0, scaling_fact=252):
    return -summary_stats(w, df, rf=0, scaling_fact=252)['exp_return']

def min_vol(w, df, rf=0, scaling_fact=252):
    return summary_stats(w, df, rf=0, scaling_fact=252)['volatility']

def max_sr(w, df, rf=0, scaling_fact=252):
    return -summary_stats(w, df, rf=0, scaling_fact=252)['sharpe_ratio']

def risk_parity(w, df, rf=0, scaling_fact=252):
    return summary_stats(w, df, rf=0, scaling_fact=252)['risk_parity']

def max_dr(w, df, rf=0, scaling_fact=252):
    return -summary_stats(w, df, rf=0, scaling_fact=252)['diversification_ratio']

def port_optimisation(func, df, rf=0, scaling_fact=252, bounds=None, constraints=(), v=True):
    """
    Create wrapper for scipy optimisation.
    Not really needed, just makes working with scipy a little easier.
    SLSQP selected as method as we will be working with equality constraints.
    
    """
    start = time.time()
    w = [1/len(df.columns) for x in df.columns]
    
    r = sco.minimize(
        func, w, 
        (df, rf, scaling_fact),
        method='SLSQP', 
        bounds=bounds, constraints=constraints
    ) 
    
    w = r.x
    s = r.nit > 1 and r.success
    end = time.time()

    if v:
        print('{} Success=={} after {} iterations.'.format(r.message, r.success, r.nit))
        print('Total time: {} secs'.format(end - start))

    return {'weights': w, 'success': s}