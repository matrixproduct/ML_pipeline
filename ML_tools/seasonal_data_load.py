# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:28:48 2021

@author: AOssipov

Module containing functions for

--- getting data from tsArctic based on static future quarters and effective quarters

--- getting data from tsArctic based on rolling future quarters 

"""


import numpy as np
import pandas as pd
import tsg.core.tsArctic as ts_arctic
# import tsg.utils.tsVisualize as tsv



##############################################################################
# get data from tsArctic   
##############################################################################

def slice_effdates_fwddates(df, eff_start, eff_end, fwd_dates, freq='B'):
    '''
    
    Return futures prices with specified effective dates and futures dates for
    a specific symbol.
    
    Parameters
    ----------
    df : DataFrame
        futures prices for a specific symbol
    eff_start : str
        effective start date
    eff_end : str
       effective end date
    fwd_dates : DatetimeIndex
        date range for futures
    freq : str, optional
        frequency for effective dates. The default is 'B'.

    Returns
    -------
    DataFrame 

    '''
    
    eff_dates = pd.date_range(eff_start, eff_end, freq=freq)
    cut_index = pd.MultiIndex.from_product([eff_dates, fwd_dates], names=['date', 'fwdend'])
    
    cut_df = df.reindex(cut_index)
    cut_df.dropna(inplace=True)
    
    return cut_df



def slice_effdates_fwdquarters_years(df, eff_start, eff_end, fwd_quarters, fwd_years, freq='B'):
    '''
    
    Return futures prices with specified effective dates and futures dates with
    specified quarters and years for a specific symbol.
    
    Parameters
    ----------
    df : DataFrame
        futures prices for a specific symbol
    eff_start : str
        effective start date
    eff_end : str
       effective end date
    fwd_quarters : list of int
       quarters     
    fwd_years : list of int
       years        
    freq : str, optional
        frequency for effective dates. The default is 'B'.

    Returns
    -------
    DataFrame 
    
    '''
    
    
    eff_dates = pd.date_range(eff_start, eff_end, freq=freq)
    
    
    cut_df = df[df.index.get_level_values(0).isin(eff_dates) & 
                df.index.get_level_values(1).quarter.isin(fwd_quarters) & 
                df.index.get_level_values(1).year.isin(fwd_years)]
    
    return cut_df




def name_eff_quarter(year, eff_year, quarter):
    '''
    
    Return names of effective quarters relative to eff_year

    Parameters
    ----------
    year : int
        year of a date
    eff_year : int
        year realative to each the effective quarters are named
    quarter : int
        quarter of a date

    Returns
    -------
    eff_quarter : str
        name of the effective quarter

    '''
    if year == eff_year:
        eff_quarter_name = 'Q' + str(quarter)
    elif year == eff_year - 1:
            eff_quarter_name = 'Q' + str(quarter) + ' prev'
    else:
        k = eff_year - year
        eff_quarter_name = 'Q' + str(quarter) + ' prev_' + str(k)        
     
    return eff_quarter_name




def save_data_symbols(df_syms, sym_names, eff_start_md, eff_end_md, eff_start_years_back,
                            eff_years, fwd_quarters_current, fwd_quarters_next, filename=None, 
                            eua_required=False, usd_eur_required=False):
    '''
    
    Save the data for a list of symbols or for a single symbol, 
    effective time and future quarters 
    using the DataFrames containing data for these symbols and 
    return the required data as a DataFrame. 
    If filename is None just return the DataFrame.

    Parameters
    ----------

    df_syms : list of DataFrames with
        futures prices for a specific symbol
    sym_names: list of str
        names of the symbols to be used in the output DataFrame    
    eff_start_md : str
        effective start date: M-d
    eff_end_md : str
        effective end date: M-d   
    eff_start_years_back : int
        difference in years for eff_end and eff_start dates    
    eff_years: list of int
        effective years        
    fwd_quarters_current : list of int
        quarters in the current year  
    fwd_quarters_next : list of int
        quarters in the next year 
    filename : str
        name of the output file 
        The default is None
    eua_required : Boolean
        True if EUA data is required, then eua is the first in the list
        The default is False
    usd_eur_required : Boolean
        True if USD/EUR data is required,  then usd_eur follows eua in the list
        The default is False    
        

    Returns
    -------
    DataFrame

    '''
    
    if not isinstance(df_syms, (list, tuple, set)):
            df_syms = [df_syms]
            
    if not isinstance(sym_names, (list, tuple, set)):
            sym_names = [sym_names]        
    
    # pick up EUA df if required        
    if eua_required:
            eua = df_syms.pop(0)  # eua must be the first element
            eua_name = sym_names.pop(0) 
           
            
    # pick up USDEUR df if required 
    if usd_eur_required:   
       usdeur = df_syms.pop(0)  # usd_eur must be the first element now
       usdeur_name = sym_names.pop(0)
               
            
    data_all_years = pd.DataFrame()
    
    for eff_year in eff_years:
        start_year = eff_year - eff_start_years_back
        end_year = eff_year
        eff_start = str(start_year) + '-' + eff_start_md
        eff_end = str(end_year) + '-' + eff_end_md
        eff_dates = pd.date_range(eff_start, eff_end, freq='B')
    
        df = pd.DataFrame([])       
    
        # get EUA data if required
        if eua_required:
            fwd_dates = pd.to_datetime([str(eff_year) + '-12-31'])
            eua_slice = slice_effdates_fwddates(eua, eff_start, eff_end, fwd_dates)
            eua_slice.rename(columns={'close': eua_name}, inplace=True)
            df = df.join(eua_slice.reset_index(level=1, drop=True), how='outer') 
            
        # get USDEUR data if required
        if usd_eur_required:   
           usdeur_slice = usdeur[usdeur.index.isin(eff_dates)].copy() 
           usdeur_slice.rename(columns={'fx': usdeur_name}, inplace=True)
           df = df.join(usdeur_slice, how='outer')        
        
        # get all other symbols' data
        
        
        
        
        for df_sym, sym_name in zip(df_syms, sym_names):   
            
            for q in fwd_quarters_current:
                sym_slice = slice_effdates_fwdquarters_years(df_sym, eff_start, eff_end, [q], [eff_year])
                df2 = sym_slice.groupby(level=0).mean()
                df2.rename(columns={'close': sym_name + ' Q' +  str(q) + '-current'}, inplace=True)
                df = df.join(df2, how='outer')
                
            for q in fwd_quarters_next:
                sym_slice = slice_effdates_fwdquarters_years(df_sym, eff_start, eff_end, [q], [eff_year + 1])
                df2 = sym_slice.groupby(level=0).mean()
                df2.rename(columns={'close': sym_name + ' Q' +  str(q) + '-next'}, inplace=True)
                df = df.join(df2, how='outer')
                
      
        
        # add eff_year column
        df['eff_year'] = eff_year    
        
        data_all_years = pd.concat([data_all_years, df])   
        
    
    # add year, quarter, month and day 
    df =  data_all_years
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['quarter'] = pd.DatetimeIndex(df['date']).quarter
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
       
    # add effective quarter name
    df['eff_quarter'] = df.apply(lambda x: name_eff_quarter(x['year'], x['eff_year'], x['quarter']), axis=1)
    
    if filename is not None:
        data_all_years.to_csv(filename)
    
    return data_all_years





def get_quarterly_data(syms, sym_names, eff_start_md, eff_end_md, eff_start_years_back,
                            eff_years, fwd_quarters_current, fwd_quarters_next, filename=None):
    '''
    
    Load the data for a list of symbols or for a single symbol from tsArctic, 
    effective time and future quarters and 
    return the required data as a DataFrame.
    Save the resulting DataFrame if required. 
    If filename is None just return the DataFrame.

    Parameters
    ----------

    syms : list of symbols as used in tsArctic
    
    For example, use "eua" for EUA, "tfm" for TTF, "gab" for German Power
    "atw" for Coal, "usd_eur" for EUR/USD
    
    sym_names: list of str
        names of the symbols to be used in the output DataFrame    
    eff_start_md : str
        effective start date: M-d
    eff_end_md : str
        effective end date: M-d   
    eff_start_years_back : int
        difference in years for eff_end and eff_start dates    
    eff_years: list of int
        effective years        
    fwd_quarters_current : list of int
        quarters in the current year  
    fwd_quarters_next : list of int
        quarters in the next year 
    filename : str
        name of the output file 
        The default is None
        

    Returns
    -------
    DataFrame

    '''
    
    if not isinstance(syms, (list, tuple, set)):
            syms = [syms]
            
    if not isinstance(sym_names, (list, tuple, set)):
            sym_names = [sym_names]           
    
    df_syms = []   
    new_sym_names = [] 
    eua_required, usd_eur_required = False, False
    
    db = ts_arctic.ArcticCross('Arctic.trader')
    
    # check for eua
    if 'eua' in syms:
        ind = syms.index('eua')
        eua_required = True
        eua = db.read('ice.code.eua', 'trader')
        df_syms.append(eua)
        new_sym_names.append(sym_names.pop(ind))
        syms.pop(ind)
        
    # check for usdeur
    if 'usd_eur' in syms: 
        ind = syms.index('usd_eur')
        usd_eur_required = True
        usdeur = db.read('ecb.fx.usd_eur', 'trader') 
        df_syms.append(usdeur)
        new_sym_names.append(sym_names.pop(ind)) 
        syms.pop(ind)
        
    
    
    # all other symbols
    
    for sym, sym_name in zip(syms, sym_names):
        df_sym = db.read('ice.code.' + sym, 'trader')
        df_syms.append(df_sym)
        new_sym_names.append(sym_name) 

    # print(new_sym_names)    
    # return
       
    data_all_years = save_data_symbols(df_syms, new_sym_names, eff_start_md, eff_end_md, eff_start_years_back,
                            eff_years, fwd_quarters_current, fwd_quarters_next, filename, 
                            eua_required, usd_eur_required)   
       
     
    return data_all_years




def get_roll_data_symbols(syms, sym_names,  fwd_quarters_offsets,
                             eff_start=None, eff_end=None,
                             filename=None, normal_return=True):
    '''
    
    Load the data for a list of symbols or for a single symbol from tsArctic,
    effective time and rolling future quarters 
    using the DataFrames containing data for these symbols and 
    return the required data as a DataFrame. 
    Save DataFrame to file, if filename is not None.
    
    

    Parameters
    ----------

    syms : list of symbols as used in tsArctic
    
    For example, use "eua" for EUA, "tfm" for TTF, "gab" for German Power
    "atw" for Coal, "usd_eur" for EUR/USD
    
    sym_names: list of str
        names of the symbols to be used in the output DataFrame     
    fwd_quarterss_offsets : list of (int, int)
        future quarters, offsets
        offset = k means that the difference in months for the end of the start 
        months of the future quarter and effective time is not less than k months
    eff_start : str
        effective start date: Y-M-d
    eff_end : str
        effective end date: Y-M-d       
    filename : str
        name of the output file 
        The default is None
    eua_required : Boolean
        True if EUA data is required, then eua is the first in the list
        The default is False
    usd_eur_required : Boolean
        True if USD/EUR data is required,  then usd_eur follows eua in the list
        The default is False   
    normal_return : Boolean
        True for normal retur, False for lognormal
        The default is True
        

    Returns
    -------
    DataFrame
    
    
    '''


    if not isinstance(syms, (list, tuple, set)):
            syms = [syms]
            
    if not isinstance(sym_names, (list, tuple, set)):
            sym_names = [sym_names] 
            
    if not isinstance(fwd_quarters_offsets, (list)):
           fwd_quarters_offsets = [fwd_quarters_offsets]         
    
   
    price_change = '.pxchg' if normal_return else 'logpxchg'
    
    
    db = ts_arctic.ArcticCross('Arctic.trader')
    
    df = pd.DataFrame()
    
    # check for eua
    if 'eua' in syms:
        print('EUA')
        ind = syms.index('eua')
        eua = db.slice_with_calendar('ice.code.eua','trader', 'nb01', 
                                     eff_start=eff_start, eff_end=eff_end,
                                     allowed_months=12)
        eua_name = sym_names.pop(ind)
        eua.rename(columns={eua.columns[0]: eua_name}, inplace=True)
        
        eua_return = db.slice_with_calendar('ice.code.eua' + price_change,'trader', 'nb01', 
                                             eff_start=eff_start, eff_end=eff_end,
                                             allowed_months=12)
        eua_return.rename(columns={eua_return.columns[0]: eua_name + ' return'}, inplace=True)
        df = df.join(eua, how='outer') 
        df = df.join(eua_return, how='outer') 
        
    
        syms.pop(ind)
        
    # check for brent
    if 'b' in syms:
        print('Brent')
        ind = syms.index('b')
        brent = db.slice_with_calendar('ice.code.b','trader', 'nb12', 
                                     eff_start=eff_start, eff_end=eff_end)
        brent_name = sym_names.pop(ind)
        brent.rename(columns={brent.columns[0]: brent_name}, inplace=True)
        
        brent_return = db.slice_with_calendar('ice.code.b' + price_change,'trader', 'nb12', 
                                             eff_start=eff_start, eff_end=eff_end)
        brent_return.rename(columns={brent_return.columns[0]: brent_name + ' return'}, inplace=True)
        df = df.join(brent, how='outer') 
        df = df.join(brent_return, how='outer') 
        
    
        syms.pop(ind)    
        
    # check for usdeur
    if 'usd_eur' in syms: 
        print('USD_EUR')
        ind = syms.index('usd_eur')
        usdeur_name = sym_names.pop(ind)
        usdeur = db.read('ecb.fx.usd_eur', 'trader', eff_start=eff_start, eff_end=eff_end)         
        usdeur.rename(columns={'fx': usdeur_name}, inplace=True)
        df = df.join(usdeur, how='outer')        
        syms.pop(ind)
           
    # check for gbpeur
    if 'gbp_eur' in syms: 
        print('GBP_EUR')
        ind = syms.index('gbp_eur')
        gbpeur_name = sym_names.pop(ind)
        gbpeur = db.read('ecb.fx.gbp_eur', 'trader', eff_start=eff_start, eff_end=eff_end)         
        gbpeur.rename(columns={'fx': gbpeur_name}, inplace=True)
        df = df.join(gbpeur, how='outer')        
        syms.pop(ind)    
      
    # all other symbols
    for sym, sym_name in zip(syms, sym_names):
        print(sym_name)
        for quarter, offset in fwd_quarters_offsets:
            df_sym = get_quarter_mean(db, sym, quarter, offset, eff_start, eff_end)
            df_sym.rename(columns={df_sym.columns[0]: sym_name + ' Q' + str(quarter)}, inplace=True)
            
            df = df.join(df_sym, how='outer') 
            
            # price change
            
            if '.' not in sym:
                df_sym_return = get_quarter_mean(db, sym + price_change, quarter, offset, eff_start, eff_end)
                df_sym_return.rename(columns={df_sym_return.columns[0]: sym_name + ' Q' + str(quarter) + ' return'}, inplace=True)
               
                
                df = df.join(df_sym_return, how='outer')  
            

    print(df.columns)
    # add year, quarter, month and day 
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['quarter'] = pd.DatetimeIndex(df['date']).quarter
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
      
    
    if filename is not None:
        df.to_csv(filename,  index=False)         
                

    return df 



def get_quarter_mean(db, sym, quarter, offset, eff_start=None, eff_end=None):
    
    '''
    Return the mean value of a symbol for a quarter for all dates which are not 
    closer to the end of the first months of the quarter by offset=k months. More precisely offset=k
    correspond to 'nbk' calendar for the first month of the quarter
    
    '''
    
    
    months = [(quarter - 1) * 3 + i for i in range(1, 4)] 
    cals = ['nb0' + str(offset + i) if offset + i < 10 else 'nb' + str(offset + i) for i in range(0, 3)]
    df_list = []
    for cal, month in zip(cals, months):
           # print(cal, month) 
           df_sym = db.slice_with_calendar('ice.code.' + sym, 'trader', cal, eff_start=eff_start, eff_end=eff_end, allowed_months=month)
           df_list.append(df_sym)
           
    return sum(df_list) / 3  # mean value for a quarter    



#############################################################################
#############################################################################

if __name__ == "__main__":
    
    # TESTS

    # db = ts_arctic.ArcticCross('Arctic.trader')
    
    
    
    # gpw = db.read('ice.code.gab', 'trader')
    # eua = db.read('ice.code.eua', 'trader')
    # ttf = db.read('ice.code.tfm', 'trader')
    # atw = db.read('ice.code.atw', 'trader')
    # usdeur = db.read('ecb.fx.usd_eur', 'trader') 
    
    
    eff_start_md, eff_end_md = '01-01', '05-31' 
    eff_start, eff_end = '2018-01-01', '2020-05-31' 
    eff_start_years_back = 2
    
    
    eff_years = np.arange(2018, 2022)
    fwd_quarters_current = [3, 4]
    fwd_quarters_next = [1, 2, 3, 4]
    
           
    # data_all_years = save_data_sym(ttf, 'TTF', 
    #                                           eff_start_md,  eff_end_md, 
    #                                           eff_start_years_back, eff_years,
    #                                           fwd_quarters_current, fwd_quarters_next,
    #                                           # 'C:/Users/aossipov/Projects/EUA/Input_Data/gpw_multireg_data.csv',  
    #                                           # 'K:/Alexander Ossipov/Projects/Seasonal_comparison/Input_Data/ttf_etc_6_years.csv'
    #                                         'K:/Alexander Ossipov/Projects/temp.csv')
    
    # data_all_years = save_data_symbols(ttf, 'TTF', 
    #                                           eff_start_md,  eff_end_md, 
    #                                           eff_start_years_back, eff_years,
    #                                           fwd_quarters_current, fwd_quarters_next,
    #                                           # 'C:/Users/aossipov/Projects/EUA/Input_Data/gpw_multireg_data.csv',  
    #                                           # 'K:/Alexander Ossipov/Projects/Seasonal_comparison/Input_Data/ttf_etc_6_years.csv'
    #                                         'K:/Alexander Ossipov/Projects/temp.csv')
    
    # data_all_years = save_data_symbols([ttf, gpw, atw], ['TTF', 'GPW', 'ATW'], 
    #                                           eff_start_md,  eff_end_md, 
    #                                           eff_start_years_back, eff_years,
    #                                           fwd_quarters_current, fwd_quarters_next,
    #                                           # 'C:/Users/aossipov/Projects/EUA/Input_Data/gpw_multireg_data.csv',  
    #                                           # 'K:/Alexander Ossipov/Projects/Seasonal_comparison/Input_Data/ttf_etc_6_years.csv'
    #                                         'K:/Alexander Ossipov/Projects/temp.csv')
    
    
     # data_all_years = get_quarterly_data(['tfm', 'gab', 'eua', 'usd_eur'], ['TTF', 'GPW', 'EUA', 'EUR/USD' ], 
     #                                          eff_start_md,  eff_end_md, 
     #                                          eff_start_years_back, eff_years,
     #                                          fwd_quarters_current, fwd_quarters_next,
     #                                          # 'C:/Users/aossipov/Projects/EUA/Input_Data/gpw_multireg_data.csv',  
     #                                          # 'K:/Alexander Ossipov/Projects/Seasonal_comparison/Input_Data/ttf_etc_6_years.csv'
     #                                        'K:/Alexander Ossipov/Projects/temp.csv')

    # data_all_years = get_quarterly_data('usd_eur', 'EUR/USD' , 
    #                                           eff_start_md,  eff_end_md, 
    #                                           eff_start_years_back, eff_years,
    #                                           fwd_quarters_current, fwd_quarters_next,
    #                                           # 'C:/Users/aossipov/Projects/EUA/Input_Data/gpw_multireg_data.csv',  
    #                                           # 'K:/Alexander Ossipov/Projects/Seasonal_comparison/Input_Data/ttf_etc_6_years.csv'
    #                                         'K:/Alexander Ossipov/Projects/temp.csv')
    
    # data_all_years = get_roll_data_symbols(['tfm', 'gab', 'eua',  'usd_eur'], ['TTF', 'GPW', 'EUA' , 'EUR/USD' ], (2,1), eff_start, eff_end, 
    #                                             normal_return=True, filename='K:/Alexander Ossipov/Projects/temp.csv')
    
    # data_all_years = get_roll_data_symbols(['tfm', 'gab', 'eua',  'usd_eur'], ['TTF', 'GPW', 'EUA' , 'EUR/USD' ], (2,1),
    #                                        eff_start=eff_start, eff_end=None, 
    #                                             normal_return=True, filename='K:/Alexander Ossipov/Projects/temp.csv')
    
    # data_all_years = get_roll_data_symbols(['eua', 'usd_eur'], ['EUA' , 'EUR/USD' ], (1,1), eff_start, eff_end, 
    #                                       normal_return=True)
    
    
    # data_all_years = get_roll_data_symbols(['tfm'], ['TTF'], (2,6), eff_start='2013-01-01', eff_end='2021-06-30', 
    #                                            normal_return=True)
    
    syms_ice = ['tfm', 'atw' , 'eua', 'usd_eur', 'gbp_eur', 'jkm', 'm', 'h', 'tfm.atmvol', 'tfm.qdelta75', 'tfm.qdelta25', 'b' ]
    sym_names = ['TTF', 'Coal', 'EUA', 'EUR/USD', 'EUR/GBP', 'JKM', 'NBP', 'HenryHub', 'TTF Vol', 'TTF qdelta75', 'TTF qdelta25', 'Brent' ]

    data_all_years = get_roll_data_symbols(syms_ice, sym_names, (2,6), eff_start='2013-01-01', eff_end='2021-06-30', 
                                          normal_return=True)
    
    # db.slice_with_calendar('ice.code.' + 'gab', 'trader', 'nb01', eff_start=eff_start, eff_end=eff_end, allowed_months=1)
    
    print(data_all_years)
    print(data_all_years.columns)


