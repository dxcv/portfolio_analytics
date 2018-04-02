import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import time

class Opt_Helpers(object):
    """
    Helpers for basic portfolio optimisation techniques,
    Expects a df of prices **NOT RETURNS** with a datetime index.
    """

    def __init__(self, df, rf=0, scaling_fact=252):
        self.df = df
        self.rf = 0
        self.scaling_fact = 252    
        self.columns = df.columns  
        self.opt_method = None
        self.weights = None  
        self.stats = None

    def summary_stats(self, w):
        """Calculates summary statistics needed for optimisation target."""

        ln_ret = np.log(self.df / self.df.shift(1))

        """Expected return target (based on historical mean)."""
        er = np.dot(w, ln_ret.mean()) * self.scaling_fact

        """Historical max drawdown."""
        ret = ((np.exp(ln_ret) - 1) * w).sum(axis=1).add(1).cumprod()
        dd = ret.div(ret.cummax()).sub(1)
        mdd = dd.min()

        """Marginal risk, risk contribution, and risk parity target."""
        mvar = np.dot(w, ln_ret.cov().T) * w * self.scaling_fact
        pvar = sum(mvar)
        r_con = mvar / pvar
        rp_target = sum((pvar/len(w) - mvar)**2) * 10000 #Scale up by 10000 to make the optimisation easier

        """Portfolio volatility target and diversification ratio target."""
        pvol_target = pvar ** 0.5
        dr_target = sum((w * ln_ret.std() * 252 ** 0.5)) / pvol_target

        """Sharpe ratio target."""
        sr = (er - self.rf) / pvol_target

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

    def max_er(self, w):
        self.opt_method = 'max_er'
        return -self.summary_stats(w)['exp_return']

    def min_vol(self, w):
        self.opt_method = 'min_vol'
        return self.summary_stats(w)['volatility']

    def max_sr(self, w):
        self.opt_method = 'max_sr'
        return -self.summary_stats(w)['sharpe_ratio']

    def risk_parity(self, w):
        self.opt_method = 'risk_parity'
        return self.summary_stats(w)['risk_parity']

    def max_dr(self, w):
        self.opt_method = 'max_dr'
        return -self.summary_stats(w)['diversification_ratio']

    def port_optimisation(self, func, bounds=None, constraints=()):
        """
        Create wrapper for scipy optimisation.
        Not really needed, just makes working with scipy a little easier.
        SLSQP selected as method as we will be working with equality constraints.
        
        """
        start = time.time()
        w = [1/len(self.df.columns) for x in self.df.columns]
        
        r = sco.minimize(
            func, w, 
            method='SLSQP', 
            bounds=bounds, constraints=constraints
        ) 
        
        w = r.x
        
        s = self.summary_stats(w)
        
        stats = {
            'exp_return': s['exp_return'],
            'volatility': s['volatility'],
            'sharpe_ratio': s['sharpe_ratio'],
            'max_drawdown': s['max_drawdown']
        }
        
        results = {
            'weights': w,
            'summary_stats': stats,
            'success': r.success    
        }
        
        end = time.time()
        print('{} Success=={} after {} iterations.'.format(r.message, r.success, r.nit))
        print('Total time: {} secs'.format(end - start))

        self.weights = pd.DataFrame(w, index=self.columns, columns=['pos_wgt'])
        self.weights['vol_contrib'] = pd.DataFrame(s['risk_contribution'], index=self.columns)
        self.stats = pd.DataFrame([stats], index=['summary_stats']).T
        
        return results