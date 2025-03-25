" Reads CSV file and save a csv file with the data if any transformation is required"
import pandas as pd
import numpy as np
import matplotlib as plt

def load_and_format_data(file_path):

    df = pd.read_csv(file_path)
    df['Date'] = pd.PeriodIndex(df['Date'], freq='Q').to_timestamp()   

    df = df.set_index('Date')

    return(df)

    
file_path = '/home/laptopubuntu/work/gpm_ideas/parser/data/usa_data.csv'
df = load_and_format_data(file_path=file_path)
df['l_gdp_obs'] = np.log(df['GDP'])
df['l_cpi_obs'] = np.log(df['CPI'])
df['dla_cpi_obs'] = 400*np.diff(df['l_cpi_obs'], prepend=True)
df['rs_obs'] = df['RS']
#Drop the first observation 
df = df.iloc[1:,:]
df = df[['l_gdp_obs', 'dla_cpi_obs', 'rs_obs']]
df.to_csv('/home/laptopubuntu/work/gpm_ideas/parser/data/transformed_data_us.csv')

df[['dla_cpi_obs', 'rs_obs']].plot()