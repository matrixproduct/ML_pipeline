# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:51:28 2021

@author: aossipov

Module containing functions for loading data for COT project.

"""

import numpy as np
import pandas as pd


import os, sys


sys.path.append('C:/tsLocal/folib/ML_tools')

from  train_validation_test import create_binary_return




from ldz_prediction import  build_ldz_forecast_df



##############################################################################
import folib.core.tsArctic as ts_arctic
db = ts_arctic.ArcticCross('Arctic.trader')
from folib.core.CachePlus import CachePlus
dbc = CachePlus(db , 'disk', 'C:/temp/arctic', ex=60*60*12)
# dbc.read = CachePlus(dbc.read, 'disk', 'C:/temp/arctic', ex=60*60*12)


import tsg.mktdata.sources.CommodityEssentials as tsCommodityEssentials 
eugas = tsCommodityEssentials.REST()


import tsg.odin.Odin as odin
mdw = odin.MarketData()




import tsg.core.Dates as tsDates
##############################################################################

path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Gas_short_term'

##############################################################################


###############################################################################
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
                

###############################################################################

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
        
               
          
        if 'coal' in sym_name:      
            price = dbc.slice_with_calendar(sym_exchange,'trader', cal, 
                                             eff_start=eff_start, eff_end=eff_end)
            price.rename(columns={price.columns[0]: sym_name}, inplace=True)    
            df = df.join(price, how='outer')            
            continue
        
        if 'wti' in sym_name:      
            price = dbc.slice_with_calendar(sym_exchange,'trader', 'nb12', 
                                             eff_start=eff_start, eff_end=eff_end)
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

##############################################################################

def get_storage_data(start_date, end_date, countries=None, filename=None):
    
    
    if filename is not None and os.path.exists(filename):  # load data from file if it exists already
        print('Loading storage data from file: {filename}')
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index) 
        
        return df
    
    start_date, end_date = tsDates.GetDate(start_date),  tsDates.GetDate(end_date)

     
    if countries is not None and not isinstance(countries, (list, tuple)):
        countries = [countries]    

    EUStorage = {}
    
    EUStorage['TotalEUIncUKr'] = {
            'AssetIDs': 
                {
                    'UK': 468, 
                    'FR': 458, 
                    'DE': 688, 
                    'NL': 477, 
                    'BE': 677,
                    'AT': 1768,  # Austria
                    'IT': 1780, 
                    'ES': 1910,
                    'CZ': 1865,
                    'SL': 1880, # slovakia
                    'PT': 2269, # portugal
                    'RO': 2555, # romania
                    'HU': 2251,
                    'BG': 2582, # Bulgaria
                    'DK': 2001, # Denmark
                    #'TK': 2706,  # Turkey..not much data here
                    'Ukr':1968  # Ukraine
                }
            }
    
    EUStorage['EULiquidMarkets'] = {
            'AssetIDs': 
                {
                    'UK': 468, 
                    'FR': 458, 
                    'DE': 688, 
                    'NL': 477, 
                    'BE': 677,
                    'AT': 1768,  # Austria
                    'IT': 1780
                }
            }
    EUStorage['EULiquidMarketsexUK'] = {
            'AssetIDs': 
                {
                    'FR': 458, 
                    'DE': 688, 
                    'NL': 477, 
                    'BE': 677,
                    'AT': 1768,  # Austria
                    'IT': 1780
                }
            }
    EUStorage['EULiquidMarketsIncUkr'] = {
            'AssetIDs': 
                {
                    'UK': 468, 
                    'FR': 458, 
                    'DE': 688, 
                    'NL': 477, 
                    'BE': 677,
                    'AT': 1768,  # Austria
                    'IT': 1780,
                    'Ukr':1968  # Ukraine
                }
            }
    
    EUStorage['Norg'] = {
            'AssetIDs': 
                {
                    'Norg': 216
                }
            }
    
    EUStorage['GazpromStorageExCZ'] = {
            'AssetIDs': 
                {
                    'Rehden': 617, 
                    'HaidachAstora': 1760, 
                    'HaidachGSA': 1820,                 
                    'Jemgum': 589, 
                    'EtzelEKB': 580, 
                    'Katharina': 626, 
                    'Bergermeer': 220  ## only half of it gazprom
                }
            }
                                           
    
    
    df_all = pd.DataFrame()
    df_all_totals = pd.DataFrame()
    
    
    for i in EUStorage:
        
        # print(f'{i=}')
    
        for country, x in EUStorage[i]['AssetIDs'].items():
            
            # print(f'{x=}')
            
            df = eugas.eugasopdata(assets=x,unit='GWh',variable='Stock')
            df = df.set_index('Date')
            df.rename(columns ={df.columns[0]: country}, inplace=True)
            df_all = pd.concat([df_all,df],axis = 1)
    
        df_all = df_all.dropna()
        df_all[str(i)] = df_all.sum(axis=1) 
        #breakpoint()
        
        # return df_all
        
        df_all_totals = pd.concat([df_all_totals,df_all[str(i)]],axis = 1)
        
        if countries is not None and i == 'TotalEUIncUKr':
            df_all_totals = pd.concat([df_all_totals,df_all[countries]],axis = 1)
        df_all = pd.DataFrame()
    
    df_all_totals = df_all_totals/11.0  # convert gwh to mcm
    df_all_totals = df_all_totals.dropna()
    
    
    df_storage = df_all_totals.copy()
    df_storage.index = pd.to_datetime(df_storage.index)
    df_storage
    
    
    df_storage['dayofyear'] = df_storage.index.dayofyear
    
    
    
    years_range = np.arange(2014, 2019)
    avg_values = df_storage[df_storage.index.year.isin(years_range)].groupby('dayofyear').mean().reset_index().add_prefix('avg_')
    avg_values.rename(columns={'avg_dayofyear':'dayofyear'}, inplace=True)
    avg_values
    
    
    df_new = pd.merge(df_storage.reset_index(), avg_values, on='dayofyear', sort=False)
    df_new = df_new.sort_values('index').set_index('index')
    df_new
    
    
    original_cols = df_all_totals.columns
    df_storage_diff = pd.DataFrame()
    for col in original_cols:
        df_storage_diff[col] = df_new[col] - df_new['avg_' + col]
    
    df_storage_diff = df_storage_diff[start_date: end_date]
    
    # save data to file
    if filename is not None:
        df_storage_diff.to_csv(filename)
                            
    
    
    return df_storage_diff

###############################################################################

def get_uk_pipes_regas_data(start_date, end_date, filename=None):
    
    def cecall(seriesID, columnName = None):
        series = eugas.Series(seriesID)
        series.set_index('Date', inplace =True)
        if columnName is not None:
            series.columns = [columnName]
        return series
    
    
    if filename is not None and os.path.exists(filename):  # load data from file if it exists already
        print('Loading pipes and regas data from file: {filename}')
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index) 
        
        return df
    
    
    start_date, end_date = tsDates.GetDate(start_date),  tsDates.GetDate(end_date)
    
    
    no_to_uk = cecall(29616,'no')
    nl_to_uk = cecall(23105, 'nl')
    be_to_uk = cecall(23100, 'be') * -1
    uk_regas = eugas.eugasopdata(assets=467,unit='GWh', status='blend')
    uk_regas = uk_regas.set_index('Date')
    uk_regas = uk_regas.iloc[:,7:8]
    uk_regas.rename(columns={str('43439'): "regas"}, inplace=True)
    uk_regas /= 11
    df = pd.concat([no_to_uk,nl_to_uk,be_to_uk, uk_regas], axis=1)
    
   
    
    df['pipes'] = df[['no', 'nl', 'be']].sum(axis=1) 
    
    # change w.r.t. to the previous day
    # df['pipes_change'] = df['pipes'] - df['pipes'].shift(1)
    # df['regas_change'] = df['regas'] - df['regas'].shift(1)
    
    
    # change w.r.t. to 30 days moving average
    df['pipes_change'] = df['pipes'] - df['pipes'].rolling(window=30).mean() 
    df['regas_change'] = df['regas'] - df['regas'].rolling(window=30).mean() 
    
    df.drop(columns=['no', 'nl', 'be'], inplace=True)
    original_cols = df.columns
    
    df.dropna(inplace=True)
    
    df['dayofyear'] = df.index.dayofyear
        
    years_range = np.arange(2014, 2019)
    avg_values = df[df.index.year.isin(years_range)].groupby('dayofyear').mean().reset_index().add_prefix('avg_')
    avg_values.rename(columns={'avg_dayofyear':'dayofyear'}, inplace=True)
    avg_values
    
    
    df_new = pd.merge(df.reset_index(), avg_values, on='dayofyear', sort=False)
    df_new = df_new.sort_values('Date').set_index('Date')
    df_new
    
    
    df_diff = pd.DataFrame()
    for col in original_cols:
        df_diff[col] = df_new[col] - df_new['avg_' + col]
    
    df_diff = df_diff[start_date: end_date]
    
    
    
    # save data to file
    if filename is not None:
        df_diff.to_csv(filename)
   
    return df_diff     
    



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
   


###############################################################################

def get_data(symbols, eff_start, eff_end, num_days, 
             cal='nb01', cal2='nb02', drop_nan=True, price_only=False,
             countries=None,  # storage data for specific countries
             symbols_filename=None, ldz_forecast_filename=None, filename=None, **kwargs):

    if filename is not None and os.path.exists(filename):  # load data from file if it exists already
        print(f'Loading all data from file: {filename}')
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index) 
        
        return df
     
    # get symbol data
    df = get_symbol_data(symbols, eff_start, eff_end, num_days, 
                        cal=cal, cal2=cal2, drop_nan=drop_nan, price_only=price_only,
                        filename=symbols_filename, **kwargs)
    
    # get ldz forecast data 
    # currently we load ldz data only from file
    df_ldz =  build_ldz_forecast_df(country_list=None, start_date=None, end_date=None, 
                                    weather_fcst_filename_prefix=None, actual_ldz_filename_prefix=None,
                              ldz_fcst_filename=None, output_filename=ldz_forecast_filename)
    
    df_ldz = df_ldz.tz_localize(None)
    
    df = df.join(df_ldz, how='outer')
    
    
    # storage data
    df_storage = get_storage_data(eff_start, eff_end, countries=countries)
    df = df.join(df_storage, how='outer')
    
    
    # pipes and reas for UK
    if 'nbp' in symbols:
        pipes_filename = kwargs.get('pipe_filename', None)
        df_pipes = get_uk_pipes_regas_data(eff_start, eff_end, filename=pipes_filename)
        df = df.join(df_pipes, how='outer')
    
    # var data
    var_filename = kwargs.get('var_filename', None)
    if var_filename is not None:
        var_col_name = '1D VaR-'
        df_var = pd.read_csv(var_filename, index_col=0)
        df_var.index = pd.to_datetime(df_var.index)
        df = df.join(df_var[[var_col_name]], how='outer')
        
    
    if drop_nan:        
        df.dropna(inplace=True)  
        

                    
    # save data to file
    if filename is not None:
        df.to_csv(filename)
   
    return df      


##############################################################################

# def rename_cols(cols, sym_new, sym_old):
#     if sym_old is None:
#         return [sym_new + ' ' + col for col in cols]        
#     return [col.replace(sym_old, sym_new) for col in cols]   
                
        
# %% main


if __name__ == '__main__':
    # symbols = {'nbp':'ice.code.m', 'ttf': 'ice.code.tfm', 'EUR/GBP':'ecb.fx.gbp_eur'}   
    # symbols = {'nbp':'ice.code.m'}
    # symbols = {'ttf': 'ice.code.tfm'}
    symbols = {'ttf': 'ice.code.tfm', 'jkm': 'ice.code.tfm', 'EUR/USD':'ecb.fx.usd_eur'}
    # symbols = {'nbp':'cme.code.ukg', 'ttf': 'ice.code.tfm'}
    # symbols = {'ttf': 'ice.code.tfm'}
    # symbols = {'ttf': 'cme.code.ttf'}
    # symbols = {'nbp':'cme.code.ukg'}
    # symbols = {'nbp':'ice.code.m'}
    eff_start = '20150101' 
    eff_end = '20220101'  
    num_days = 1
    cal='nb01'
    cal2='nb02'
    # cal2=None
   
    
    symbols_names = ''
    for symbol in symbols:
        if 'EUR' not in symbol:
            symbols_names += f'_{symbol}'
        
    symbols_filename =  f'{path_prefix}/Output_Data/price_data{symbols_names}.csv'
    

    # df = get_symbol_data(symbols, eff_start, eff_end, num_days, 
    #                     cal=cal, cal2=cal2, drop_nan=True, 
    #                     # filename=symbols_filename
    #                     )
    
    
    symbols = {'ttf': 'ice.code.tfm'}
    eff_start = '20150101' 
    eff_end = '20220313'  
    symbols_filename =  f'{path_prefix}/Output_Data/price_data_ttf_da.csv'
    # symbols_filename = None
    df = get_symbol_data(symbols, eff_start, eff_end, num_days, 
                        cal=cal,   drop_nan=True,  day_ahead=True, 
                      filename=symbols_filename
                        )
    
    
    # symbols_names = ['nbp', 'ttf']
    # df = get_dayahead_data(symbols_names, eff_start, eff_end)
   
    countries = ['UK', 'DE']
    country ='DE'
    # df_storage = get_storage_data(eff_start, eff_end, countries=countries)
    
    # df = get_uk_pipes_regas_data(eff_start, eff_end)
    
    ldz_forecast_filename = f'{path_prefix}/Output_Data/ldz_fcst_and_actual_{country}_2015-01-01_2022-01-01.csv'
    
    var_filename = f'{path_prefix}/Input_Data/spark_spread_var_.csv'
    
    all_data_filename = f'{path_prefix}/Output_Data/ldz_fcst_and_actual_storage_{country}{symbols_names}_cal={cal}.csv'
   
    # df = get_data(symbols, eff_start, eff_end, num_days, 
    #           cal=cal, cal2=cal2, drop_nan=True,
    #           countries=countries,
    #           symbols_filename=symbols_filename, 
    #           ldz_forecast_filename=ldz_forecast_filename, 
    #           filename=all_data_filename, var_filename=var_filename)
    
    