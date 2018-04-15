import pandas as pd 
import numpy as np 
import scipy.optimize as sco
from pyport.portopt import opt, objfunc
import time
import datetime as dt

__all__ = ['PyBacktest']

class PyBacktest(object):

    def __init__(self, df, opt_period = 365, val_period = 90, rf=0, scaling_fact=252):
        """
        Parameters
        ----------  
        df: DataFrame of prices
        opt_period: number of periods used for covar matrix and return expectations
        val_period: number of periods used for out of sample results
        rf: risk free rate, scalar
        """
        

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
        """
        Returns a list dates and weights.

        Parameters
        ----------        
        func: objective function from pyportopt package
        bounds: list of tuples defining boundaries within optimisation
        constraints: list of dictionaries defining constraints
        v: boolean for printing
        """

        non_opt_func = ['equal_weights', 'inv_volatility', 'inv_variance']
        
        start = time.time()
        bt_in_weights = [] # in sample weights
        bt_out_weights = [] # out of sample weights
        for opt_dt, val_dt in zip(self.bt_calendar[0], self.bt_calendar[1]):
            opt_s, opt_e = opt_dt
            val_s, val_e = val_dt

            df = self.df[(self.df.index >= opt_s) & (self.df.index <= opt_e)]     

            if func.__name__ in non_opt_func:
                if func.__name__ == 'equal_weights':
                    res = {'weights': objfunc.equal_weights(df)}
                if func.__name__ == 'inv_volatility':
                    res = {'weights': objfunc.inv_volatility(df)}
                if func.__name__ == 'inv_variance':
                    res = {'weights': objfunc.inv_variance(df)}
            else:
                res = opt.port_optimisation(func, df=df, rf=self.rf, scaling_fact=self.scaling_fact, bounds=bounds, constraints=constraints, v=v)
                if not res['success']:
                    return print('Optimisation Failed')
            bt_in_weights.append([[opt_s, opt_e], res['weights']])
            bt_out_weights.append([[val_s, val_e], res['weights']])

        rb_in_dts = {}
        for (s, e), wgt in bt_in_weights:
            rb_in_dts[self.df.loc[s:e].index[0]] = wgt

        rb_out_dts = {}
        for (s, e), wgt in bt_out_weights:
            rb_out_dts[self.df.loc[s:e].index[0]] = wgt

        self.in_weights = rb_in_dts
        self.in_init_dt = list(rb_in_dts.keys())[0]

        self.out_weights = rb_out_dts
        self.out_init_dt = list(rb_out_dts.keys())[0]

        end = time.time()

        print('Algorithm: {}\nTotal time: {} secs\n'.format(func.__name__, round(end - start, 4)))

        return [rb_in_dts, rb_out_dts]

    def bt_timeseries(self, invested=1, bt_type='out', start_date='out'):
        """
        Walkforward optimisation based on bt_optimisation results,
        Returns timeseries using the backtested weights, 
        Assumes rebalancing is done at the end of the rebalance date.

        Parameters
        ----------
        invested: monetary value of initial investment, scalar
        bt_type: backtest type: 'in' is in sample, 'out' is out of sample. Defaults to 'out'
        purpose: start date of output DataFrame: 'in' is in sample, 'out' is out of sample. Defaults to 'out'
        """
        mv = 1
        if bt_type == 'out':
            df = self.df[self.df.index >= self.out_init_dt].copy()
            df = df.pct_change().fillna(0)

            for idx, (dt, row) in enumerate(zip(df.index, df.values)):
                if dt in self.out_weights.keys():
                    w = self.out_weights[dt]
                    df.iloc[idx] = np.multiply(np.multiply(mv, (1+row)), w)
                else:
                    df.iloc[idx] = np.multiply(df.iloc[idx-1], (1+row))
                mv = df.iloc[idx].sum()
            print('Type: out-of-sample data\n')

        elif bt_type == 'in':
            df = self.df[self.df.index >= self.in_init_dt].copy()
            df = df.pct_change().fillna(0)

            for idx, (dt, row) in enumerate(zip(df.index, df.values)):
                if dt in self.in_weights.keys():
                    w = self.in_weights[dt]
                    df.iloc[idx] = np.multiply(np.multiply(mv, (1+row)), w)
                else:
                    df.iloc[idx] = np.multiply(df.iloc[idx-1], (1+row))
                mv = df.iloc[idx].sum()
            print('Type: in-sample data\n')
        else:
            print('[] not supported. Please use "in" or "out"'.format(bt_type))
            return None
                               
        df['MV'] = df.sum(axis=1)
        
        if start_date == 'out':
            df = df[df.index >= self.out_init_dt] / df.loc[self.out_init_dt]

        return df * invested