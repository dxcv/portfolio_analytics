import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import time
import datetime as dt

class Opt_Helpers(object):
    """
    Helpers for basic portfolio optimisation techniques,
    Expects a df of prices **NOT RETURNS** with a datetime index.
    """

    def __init__(self, df, rf=0, scaling_fact=252):
        self.df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
        self.rf = rf
        self.scaling_fact = 252    
        self.columns = df.columns  
        self.opt_method = None
        self.weights = None  
        self.risk_contribution = None
        self.success = None
        self.stats = None

    def summary_stats(self, w):
        """Calculates summary statistics needed for optimisation target."""

        ln_ret = np.log(self.df / self.df.shift(1))

        """Expected return target (based on historical mean)."""
        er = np.exp(np.dot(w, ln_ret.mean()) * self.scaling_fact) - 1

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

    def port_optimisation(self, func, bounds=None, constraints=(), v=True):
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

        self.stats = pd.DataFrame([stats], index=['summary_stats']).T
        
        end = time.time()
        if v:
            print('{} Success=={} after {} iterations.'.format(r.message, r.success, r.nit))
            print('Total time: {} secs'.format(end - start))

        self.weights = w
        self.risk_contribution = s['risk_contribution']
        self.success = r.nit > 1 and r.success

class BT_Helpers(object):

    def __init__(self, df, opt_period = 365, val_period = 90, rf=0, scaling_fact=252):
        self.df = df.sort_index(ascending=True)/df.sort_index(ascending=True).iloc[0]
        self.opt_period = opt_period
        self.val_period = val_period
        self.bt_calendar = self.bt_calendar()
        self.rf = rf
        self.scaling_fact = 252
        
    def bt_calendar(self):
        """Takes in a datetime series and returns a backtest calendar."""

        df = self.df.index
        opt_period = self.opt_period
        val_period = self.val_period
        
        start_dt, end_dt = min(df), max(df) - dt.timedelta(days=val_period + 1)
        in_sample_dt, val_sample_dt = [], []
        
        idx = 0
        in_e = start_dt
        
        while in_e < end_dt:
            in_s = start_dt + dt.timedelta(days=idx * val_period)
            in_e = in_s + dt.timedelta(days=opt_period)
            if in_e > end_dt - dt.timedelta(days=val_period + 1):
                in_e = end_dt

            in_sample_dt.append([in_s, in_e])
            val_sample_dt.append([in_e + dt.timedelta(days=1), in_e + dt.timedelta(days=val_period)])

            idx += 1

        result = [in_sample_dt, val_sample_dt]
            
        return result    

    def bt_optimisation(self, func, bounds=None, constraints=(), v=False):
        """Returns a list dates and weights."""
        
        start = time.time()
        bt_weights = []
        for opt_dt, val_dt in zip(self.bt_calendar[0], self.bt_calendar[1]):
            opt_s, opt_e = opt_dt
            val_s, val_e = val_dt

            df = self.df[(self.df.index >= opt_s) & (self.df.index <= opt_e)]     
            opt = Opt_Helpers(df, rf=self.rf, scaling_fact=self.scaling_fact)

            func_dict = {
                'max_er': opt.max_er,
                'min_vol': opt.min_vol,
                'max_sr': opt.max_sr,
                'risk_parity': opt.risk_parity,
                'max_dr': opt.max_dr
                }

            opt.port_optimisation(func_dict[func], bounds=bounds, constraints=constraints, v=v)
            bt_weights.append([[val_s, val_e], opt.weights])

        rb_dts = {}
        for (s, e), wgt in bt_weights:
            rb_dts[self.df.loc[s:e].index[0]] = wgt

        self.weights = rb_dts
        self.init_dt = list(rb_dts.keys())[0]

        end = time.time()

        print('Total time: {} secs'.format(end - start))

        return rb_dts

    def bt_timeseries(self):
        df = self.df[self.df.index >= self.init_dt].copy()
        df = df/df.iloc[0]

        for idx, row in enumerate(df.itertuples()):
            dt = row[0]
            if dt in self.weights.keys():
                w = self.weights[dt]
                df.iloc[idx] = np.multiply(np.dot(row[1:], w), w)
            else:
                df.iloc[idx] = np.multiply(row[1:], w)
                
        df['NAV'] = df.sum(axis=1)

        return df

class Stats_Helpers(object):

    def __init__(self):
        return None

    def stats_summary(self, df, rf=0, scaling_fact=252):
        ln_rets = np.log(df / df.shift(1))

        res = []
        for col in ln_rets.columns:
            ln_ret = ln_rets[col]

            """Expected return target (based on historical mean)."""
            er = np.exp(ln_ret.mean() * scaling_fact) - 1

            """Historical max drawdown."""
            ret = (np.exp(ln_ret) - 1).add(1).cumprod()
            dd = ret.div(ret.cummax()).sub(1)
            mdd = dd.min()

            """Portfolio volatility."""
            pvol = ln_ret.std() * (scaling_fact ** 0.5)

            """Sharpe ratio target."""
            sr = (er - rf) / pvol

            res.append([er, pvol, sr, mdd])
        
        results = pd.DataFrame(res, columns=['exp_return', 'volatility', 'sharpe_ratio', 'max_drawdown'], index=ln_rets.columns).T

        return results