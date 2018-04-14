import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import time
import datetime as dt

__all__ = ['port_optimisation', 'dual_target_optimisation']

def port_optimisation(func, df, rf=0, scaling_fact=252, conf_lvl=0.95, bounds=None, constraints=(), v=True):
    """
    Generates optimised weights based on the func entered.
    Returns a dictionary.

    Parameters
    ----------
    func: objective function for optimisation. max_er, min_vol, max_sr, risk_parity, max_dr from pyportopt module
    df: DataFrame of prices
    rf: risk free rate used within calculations, scalar
    scaling_fact: annualisation factor used within calculations, scalar
    bounds: list of tuples for weight boundaries
    constraints: list of dictionary of constraints for scipy.optimize.minimize()
    v: Boolean to print out status of optimisation.
    """

    non_opt_func = ['equal_weights', 'inv_volatility', 'inv_variance']

    if func.__name__ in non_opt_func:
        print("{} not supported".format(func.__name__))
        return None

    start = time.time()
    w = [1/len(df.columns) for x in df.columns]
    
    r = sco.minimize(
        func, w, 
        (df, rf, scaling_fact, conf_lvl, True),
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

def dual_target_optimisation(prim_func, init_func, df, relax_tol=0.1, steps=10, rf=0, scaling_fact=252, conf_lvl=0.95, bounds=None, constraints=[]):
    """
    Dual optimisation function. Initially optimises on init_func then re-optimises on prim_func until relaxation tolerence is breached.
    Returns a DataFrame.
    
    Parameters
    ----------
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

    non_opt_func = ['equal_weights', 'inv_volatility', 'inv_variance']

    if [prim_func.__name__, init_func.__name__] in non_opt_func:
        print("{} or {} not supported".format(prim_func.__name__, init_func.__name__))
        return None

    start = time.time()
    w = [1/len(df.columns) for x in df.columns]
    res, res_w = [], []
    
    init_r = sco.minimize(
        init_func, w, 
        (df, rf, scaling_fact, conf_lvl, True),
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
            (df, rf, scaling_fact, conf_lvl, True),
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