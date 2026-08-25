import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data():
    data_df = pd.read_csv(r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\data\household_power_consumption.txt",
                           sep=';', na_values=['?', ' ?'])
    
    ## combined date and time 
    data_df['datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Time'],
                                         format="%d/%m/%Y %H:%M:%S", errors='coerce')
    data_df = data_df.drop(columns=['Date', 'Time']).set_index('datetime').sort_index()

    ## convert all columns to numeric
    data_df = data_df.apply(pd.to_numeric, errors='coerce')

    return data_df

data_df = load_data()


def missing_values(data_df):
    ## Fill null values with median
    num_cols_null = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                     'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    data_df[num_cols_null] = data_df[num_cols_null].fillna(data_df[num_cols_null].median())

    ## create hourly column of datetime
    hourly = data_df.resample('h').mean()

    ## Remove hours where all measurements are missing
    hourly = hourly.dropna(how='all')

    hourly['hour'] = hourly.index.hour
    hourly['day_of_week'] = hourly.index.day_name()
    hourly['month'] = hourly.index.month_name()
    hourly['is_weekend'] = hourly.index.day_name().isin(['Saturday', 'Sunday'])

    return hourly


if __name__ == "__main__":
    hourly = missing_values(data_df)
    print(hourly)