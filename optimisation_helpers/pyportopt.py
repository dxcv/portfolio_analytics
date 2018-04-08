import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import time
import datetime as dt

def summary_stats(w, df, rf=0, scaling_fact=252):
    """
    Calculates summary statistics needed for optimisation target,
    Returns a dictionary.

    w: List of weights
    df: DataFrame of prices
    rf: risk free rate used within calculations, scalar
    scaling_fact: annualisation factor used within calculations, scalar
    """
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
    Generates optimised weights based on the func entered.
    Returns a dictionary.

    func: objective function for optimisation. max_er, min_vol, max_sr, risk_parity, max_dr from pyportopt module
    df: DataFrame of prices
    rf: risk free rate used within calculations, scalar
    scaling_fact: annualisation factor used within calculations, scalar
    bounds: list of tuples for weight boundaries
    constraints: list of dictionary of constraints for scipy.optimize.minimize()
    v: Boolean to print out status of optimisation.
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

def dual_target_optimisation(prim_func, init_func, df, relax_tol=0.1, steps=10, rf=0, scaling_fact=252, bounds=None, constraints=[]):
    """
    Dual optimisation function. Initially optimises on init_func then re-optimises on prim_func until relaxation tolerence is breached.
    Returns a DataFrame.

    prim_func: objective function for optimisation. max_er, min_vol, max_sr, risk_parity, max_dr from pyportopt module
    init_func: objective function for optimisation. max_er, min_vol, max_sr, risk_parity, max_dr from pyportopt module
    df: DataFrame of prices
    rf: risk free rate used within calculations, scalar
    relax_tol: relaxation tolerance as a difference from optimal, e.g. 0.1 represents 10% from init_func optimal
    steps: number of steps until relax_tol is hit
    scaling_fact: annualisation factor used within calculations, scalar
    bounds: list of tuples for weight boundaries
    constraints: list of dictionary of constraints for scipy.optimize.minimize()
    v: Boolean to print out status of optimisation.
    """

    start = time.time()
    w = [1/len(df.columns) for x in df.columns]
    res, res_w = [], []
    
    init_r = sco.minimize(
        init_func, w, 
        (df, rf, scaling_fact),
        method='SLSQP', 
        bounds=bounds, constraints=constraints
    ) 

    opt_trgt = init_r.fun
    r_tol = abs(opt_trgt * relax_tol)

    for idx, rt in enumerate(np.linspace(0, r_tol, steps)):
        trgt_cons = opt_trgt + rt
        adj_cons = [*constraints, {'type': 'eq', 'fun': lambda w: init_func(w, df, rf, scaling_fact) - trgt_cons}]

        r = sco.minimize(
            prim_func, w, 
            (df, rf, scaling_fact),
            method='SLSQP', 
            bounds=bounds, constraints=adj_cons
        ) 

        opt_w = r.x
        prim_trgt = abs(prim_func(opt_w, df, rf, scaling_fact))
        init_trgt = abs(init_func(opt_w, df, rf, scaling_fact))

        res.append([prim_trgt, init_trgt])
        res_w.append(opt_w)

    res = pd.DataFrame(res, columns=[prim_func.__name__, init_func.__name__])
    res_w = pd.DataFrame(res_w, columns=df.columns).round(4)
    final_res = pd.concat([res, res_w], axis=1)
    final_res.index.name = 'step'

    end = time.time()


    print('Total time: {} secs'.format(end - start))

    return final_res