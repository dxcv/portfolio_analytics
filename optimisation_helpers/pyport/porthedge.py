import pandas as pd 
import numpy as np 
import scipy.optimize as sco

def hedged_er(er, vol, hedge_cost, hedge_strategy='put', leg1_strike=None, leg2_strike=None, years=1, paths=10000):
    """
    Simulates asset returns based on a geometric brownian motion (Black-Scholes-Merton setup)
    Returns a distribution of returns, pandas Series.

    Parameters
    ----------        
    er: mean expected return at end of years, scalar
    vol: volatility (annualised), scalar
    hedge_cost: cost of hedge_stategy, e.g. if it cost 1.5% enter 0.015, scalar
    hedge_strategy: hedge strategy, either 'put', 'put_spread', 'risk_reversal', string
    leg1_strike: long put leg, if 10% OTM strike then enter -0.10, scalar
    leg2_strike: short option leg, if 25% OTM strike then enter -0.25, scalar
    years: duration of hedge in years, e.g. if 3 months then enter 0.25, scalar
    paths: number of paths for monte carlo, scalar
    """

    S0 = 100
    ST = S0 * np.exp((er - 0.5 * vol ** 2) * T + vol * T ** 0.5 * np.random.standard_normal((iters))
    R_gbm = pd.Series(np.sort(ST / S0 - 1))

    if hedge_strategy == 'put':
        hedged_R_gbm = R_gbm.apply(lambda x: x - hedge_cost + max(leg1_strike - x, 0))
    elif hedge_strategy == 'put_spread':
        hedged_R_gbm = R_gbm.apply(lambda x: x - hedge_cost + max(min(leg1_strike - x, leg2_strike - leg1_strike), 0)
    elif hedge_strategy == 'risk_reversal':
        print('TODO')
        return None
    else:
        print('{} not supported. Please select {}, {}, or {}'.format(hedge_stategy, 'put', 'put_spread', 'risk_reversal'))


    return None