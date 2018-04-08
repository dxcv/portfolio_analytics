import pandas as pd 
import numpy as np 
import scipy.optimize as sco
import pyportopt as opt
import time
import datetime as dt

class PyBacktest(object):

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

            res = opt.port_optimisation(func, df=df, rf=self.rf, scaling_fact=self.scaling_fact, bounds=bounds, constraints=constraints, v=v)
            if not res['success']:
                return print('Optimisation Failed')
            bt_weights.append([[val_s, val_e], res['weights']])

        rb_dts = {}
        for (s, e), wgt in bt_weights:
            rb_dts[self.df.loc[s:e].index[0]] = wgt

        self.weights = rb_dts
        self.init_dt = list(rb_dts.keys())[0]

        end = time.time()

        print('Total time: {} secs'.format(end - start))

        return rb_dts

    def bt_timeseries(self, invested=1):
        """
        Walkforward optimisation based on bt_optimisation results,
        Returns timeseries using the backtested weights, 
        Assumes rebalancing is done at the end of the rebalance date.
        """

        df = self.df[self.df.index >= self.init_dt].copy()
        df = df.pct_change().fillna(0)
        mv = 1

        for idx, (dt, row) in enumerate(zip(df.index, df.values)):
            if dt in self.weights.keys():
                w = self.weights[dt]
                df.iloc[idx] = np.multiply(np.multiply(mv, (1+row)), w)
            else:
                df.iloc[idx] = np.multiply(df.iloc[idx-1], (1+row))
            mv = df.iloc[idx].sum()
                
        df['MV'] = df.sum(axis=1)

        return df * invested