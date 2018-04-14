import time
import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import datetime as dt

__all__ = ['max_er', 'min_vol', 'max_sr', 'risk_parity', 'max_dr']

def equal_weights(df):
    """
    Returns a list of equal weights.

    Parameters
    ----------
    df: DataFrame of prices
    """

    return [1/len(df.columns) for x in df.columns]

def inv_volatility(df):
    """
    Returns a list of weights inversely proportional to position standard deviation.

    Parameters
    ----------
    df: DataFrame of prices
    """

    r = df.pct_change()
    return list(1/r.std() / sum(1/r.std()))

def inv_variance(df):
    """
    Returns a list of weights inversely proportional to position variance.

    Parameters
    ----------
    df: DataFrame of prices
    """

    r = df.pct_change()
    return list(1/r.var() / sum(1/r.var()))

def max_er(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates expected return,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """ 

    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
    ln_ret = np.log(df / df.shift(1))
    er = np.exp(np.dot(w, ln_ret.mean()) * scaling_fact) - 1

    if opt:
        return -er
    else:
        return er

def min_vol(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates portfolio volatility,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """ 

    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
    ln_ret = np.log(df / df.shift(1))
    pvol = (np.dot(np.dot(w, ln_ret.cov().T), w) * scaling_fact) ** 0.5

    return pvol

def max_sr(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates portfolio sharpe ratio,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """ 

    sr = (max_er(w, df, scaling_fact=252, opt=False) - rf) / min_vol(w, df, scaling_fact=252, opt=False)

    if opt:
        return -sr 
    else:
        return sr

def risk_parity(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates risk parity objective function,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """   

    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
    ln_ret = np.log(df / df.shift(1))
    er = np.exp(np.dot(w, ln_ret.mean()) * scaling_fact) - 1

    """Marginal risk, risk contribution, and risk parity target."""
    mvar = np.dot(w, ln_ret.cov().T) * w * scaling_fact
    pvar = sum(mvar)
    rp_target = sum((pvar/len(w) - mvar)**2) * 10000 #Scale up by 10000 to make the optimisation easier

    return rp_target

def max_dr(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates diversification ratio,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """  

    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
    ln_ret = np.log(df / df.shift(1))
    er = np.exp(np.dot(w, ln_ret.mean()) * scaling_fact) - 1

    pvol = min_vol(w, df, scaling_fact=252)
    dr_target = sum((w * ln_ret.std() * scaling_fact ** 0.5)) / pvol

    if opt:
        return -dr_target
    else:
        return dr_target

def min_mdd(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates maximum drawdown,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """  

    df = (df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0] * w).sum(axis=1)
    dd = df.div(df.cummax()).sub(1)
    mdd = dd.min()

    if opt:
        return -mdd
    else:
        return mdd

def max_skew(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates portfolio skew,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """ 

    df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
    ln_ret = (np.log(df / df.shift(1)) * w).sum(axis=1)
    skew = ln_ret.skew()

    if opt:
        return -skew
    else:
        return skew

def min_hVaR(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates 1 period historical portfolio VaR,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """   

    ln_ret = (np.log(df / df.shift(1)) * w).sum(axis=1)
    hvar = ln_ret.quantile(1 - conf_lvl)

    if opt:
        return -hvar
    else:
        return hvar

def min_hcVaR(w, df, rf=0, scaling_fact=252, conf_lvl=0.95, opt=True):
    """
    Calculates 1 period historical portfolio cVaR,
    Returns a scalar.

    Parameters
    ----------
    w: List of weights
    df: DataFrame of prices
    scaling_fact: annualisation factor, scalar
    conf_lvl: confidence level for VaR calculations, scalar (only applicable to min_hVaR and min_cVaR)
    opt: Determines if calculation is for optimisation or descriptions, Boolean
    """   

    ln_ret = (np.log(df / df.shift(1)) * w).sum(axis=1)
    hcvar = ln_ret[ln_ret <= ln_ret.quantile(1 - conf_lvl)].mean()

    if opt:
        return -hcvar
    else:
        return hcvar


