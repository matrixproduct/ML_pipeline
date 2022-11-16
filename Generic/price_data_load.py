# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:36:39 2022

@author: aossipov
"""

import pandas as pd


import os, sys


sys.path.append('C:/tsLocal/folib/ML_tools')

from  train_validation_test import create_binary_return


##############################################################################
import folib.core.tsArctic as ts_arctic
db = ts_arctic.ArcticCross('Arctic.trader')
from folib.core.CachePlus import CachePlus
dbc = CachePlus(db , 'disk', 'C:/temp/arctic', ex=60*60*12)

import tsg.odin.Odin as odin
mdw = odin.MarketData()




import tsg.core.Dates as tsDates


##############################################################################
# correct way to convert price for returns from one currency to another
def return_fx_conversion(price, price_ret, fx_rate):
    return price/fx_rate - price.shift(1)/fx_rate.shift(1) 

# converts both prices and returns
def fx_conversion(df, sym, fx_rate):
    for col in df.columns:
        if sym in col and col.endswith(' return'):
                price_col = col.replace(' return', '')
                df[col] = return_fx_conversion(df[price_col], df[col], fx_rate)
                df[price_col] = df[price_col] / fx_rate 
                

##############################################################################



def get_dayahead_data(symbols_names, start_date, end_date, filename=None):
    
    if filename is not None and os.path.exists(filename):  # load data from file if it exists already
        print(f'Loading day ahead data from file: {filename}')
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index) 
        
        return df
    
    
    # MarketData series names for different commodities
    mdw_series = {'nbp': 'vNG_UK_NBP.GBP|DA', 'ttf': 'vNG_NL_TTF.EUR|DA'}
    
    
    start_date, end_date = tsDates.GetDate(start_date + '-1d'),  tsDates.GetDate(end_date)
    
    df = pd.DataFrame()
    
    for symbol in symbols_names:
        if symbol in mdw_series.keys():
            col = symbol + '_da' 
            df[col] = mdw.series(mdw_series[symbol], startDate=start_date, endDate=end_date)
            df[col + ' return'] =  df[col] - df[col].shift(1)
        
    df.dropna(inplace=True)
    
    
    # save data to file
    if filename is not None:
        df.to_csv(filename)
   
    return df      

##############################################################################

def get_symbol_data(symbols, eff_start, eff_end, num_days, 
                    cal='nb01', cal2='nb02', drop_nan=True, 
                    filename=None, **kwargs):
    
    
    if filename is not None and os.path.exists(filename):  # load data from file if it exists already
        print(f'Loading price data from file: {filename}')
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index) 
        
        return df
    
    df = pd.DataFrame()

   
    
    for sym_name, sym_exchange in symbols.items():
        
        if 'EUR' in sym_name or 'USD' in sym_name:
            price = dbc.read(sym_exchange, 'trader', eff_start=eff_start, eff_end=eff_end) 
            price.rename(columns={price.columns[0]: sym_name}, inplace=True)    
            df = df.join(price, how='outer')            
            continue
        
              
          
        price = dbc.slice_with_calendar(sym_exchange,'trader', cal, 
                                         eff_start=eff_start, eff_end=eff_end)
       
        price.rename(columns={price.columns[0]: sym_name}, inplace=True)
        
        df = df.join(price, how='outer') 
        
        
        price_change_suffix = '.pxchg' 
        price_change = dbc.slice_with_calendar(sym_exchange + price_change_suffix,'trader', cal, 
                                                 eff_start=eff_start, eff_end=eff_end)
        
        price_change.rename(columns={price_change.columns[0]: sym_name + ' return'}, inplace=True)
        
        df = df.join(price_change, how='outer') 
        
        
        
        if cal2 is not None:
        
            price2 = dbc.slice_with_calendar(sym_exchange,'trader', cal2, 
                                             eff_start=eff_start, eff_end=eff_end)
            
            price2 = price2.join(price, how='inner')
            # price - price2
            price2[sym_name + '_spread'] = price2[price2.columns[1]]-price2[price2.columns[0]]
            
            df = df.join(price2[[sym_name + '_spread']], how='outer')  
            
            price2_change = dbc.slice_with_calendar(sym_exchange + price_change_suffix,'trader', cal2, 
                                                     eff_start=eff_start, eff_end=eff_end)
            
            price2_change.rename(columns={price2_change.columns[0]: sym_name + ' cal2 return'}, inplace=True)
            price2_change = price2_change.join(price_change, how='inner')
            # price_change - price2_change
            price2_change[sym_name + '_spread return'] = price2_change[price2_change.columns[1]] \
                                                             - price2_change[price2_change.columns[0]]
                                                             
            df = df.join(price2_change[[sym_name + '_spread return']], how='outer')                                                   
            
        
       
        volume = kwargs.get('volume', None) 
        if volume is not None:
            
            volume = dbc.slice_with_calendar(sym_exchange + '.volume','trader', cal, 
                                                     eff_start=eff_start, eff_end=eff_end)  
            volume.rename(columns={volume.columns[0]: sym_name + ' Volume'}, inplace=True)
            df = df.join(volume, how='outer') 
            
        volatility = kwargs.get('volatility', None) 
        if volatility is not None:     
            
            volatility = dbc.slice_with_calendar(sym_exchange + '.atmvol', 'trader', cal, 
                                                     eff_start=eff_start, eff_end=eff_end)           
            if len(volatility) > 0:
                volatility.rename(columns={volatility.columns[0]: sym_name + ' Volatility'}, inplace=True)
            df = df.join(volatility, how='outer')     
 
    # day ahead data
    day_ahead = kwargs.get('day_ahead', False)
    if day_ahead:
        df_dayahead = get_dayahead_data(symbols.keys(), eff_start, eff_end)
        df = df.join(df_dayahead, how='outer')
        

    
    
    df = create_binary_return(df, num_days=num_days)

    
    
    if 'nbp' in symbols and 'ttf' in symbols:
        fx_conversion(df, 'nbp', df['EUR/GBP'])
        thm_MWh = 0.029307107017222     
        for feature in df.columns:
            # # 'JKM', 'NBP', 'HenryHub'
            # if ('JKM' in feature) or ('HenryHub' in feature): 
            #     df[feature] = df[feature] / MMBtu_MWh
            if 'nbp' in feature and 'binary' not in feature:
                df[feature] = df[feature] / (100 * thm_MWh)
           
    if 'jkm' in symbols and 'ttf' in symbols:
        fx_conversion(df, 'nbp', df['EUR/USD'])
        MMBtu_MWh = 0.29307107017222
        for feature in df.columns:
            # # 'JKM', 'NBP', 'HenryHub'
            # if ('JKM' in feature) or ('HenryHub' in feature): 
            #     df[feature] = df[feature] / MMBtu_MWh
            if 'jkm' in feature and 'binary' not in feature:
                df[feature] = df[feature] / (df['EUR/USD'] * MMBtu_MWh)        
         
            

    
    
    if drop_nan:        
        df.dropna(inplace=True)  
        
    # save data to file
    if filename is not None:
        df.to_csv(filename)
                            
    
    return df
