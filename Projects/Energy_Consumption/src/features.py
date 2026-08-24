import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data():
    data_df = pd.read_csv(r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\data\hourly_cleaned_power_data.csv", sep=',')
    data_df.describe
    data_df.dtypes
    return data_df

data_df = load_data()

def create_features(data_df):

    dup_df = data_df.copy()

    num_cols = dup_df.select_dtypes(include=['int64', 'float64']).columns
    impute = SimpleImputer(strategy='median')
    dup_df[num_cols] = impute.fit_transform(dup_df[num_cols])

    cat_cols = dup_df.select_dtypes(include=['object', 'bool']).columns
    impute = SimpleImputer(strategy='most_frequent')
    dup_df[cat_cols] = impute.fit_transform(dup_df[cat_cols])

    ## feature 1
    ## `Datetime' column to datetime object
    dup_df['datetime'] = pd.to_datetime(dup_df['datetime'])
    dup_df = dup_df.set_index('datetime')
    

    ## feature 2
    ## Lag features for Global_active_power at 1hr, 24hr, and 168hr (1 week) intervals
    dup_df['Global_active_power_lag_1'] = dup_df['Global_active_power'].shift(1)
    dup_df['Global_active_power_lag_24'] = dup_df['Global_active_power'].shift(24)
    dup_df['Global_active_power_lag_168'] = dup_df['Global_active_power'].shift(168)


    ## feature 3
    ## Rolling mean features for Global_active_power over a 24hr window
    dup_df['Global_active_power_roll_mean_24'] = dup_df['Global_active_power'].rolling(window=24).mean()


    ## feature 4
    ## Lag features for voltage and global_intensity
    dup_df['Voltage_lag_1'] = dup_df['Voltage'].shift(1)
    dup_df['Global_intensity_lag_1'] = dup_df['Global_intensity'].shift(1)
    dup_df['Global_intensity_rolling_mean_24'] = dup_df['Global_intensity'].rolling(window=24).mean()
    dup_df.dropna(inplace=True)

    ## feature 5
    ## Extracting time-based features from the datetime index
    dup_df['hour_sin'] = np.sin(2 * np.pi * dup_df['hour'] / 24)
    dup_df['hour_cos'] = np.cos(2 * np.pi * dup_df['hour'] / 24)

    dup_df['day_of_week_num'] = pd.Categorical(dup_df['day_of_week'], 
                                               categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                            'Saturday', 'Sunday'], ordered=True).codes
    dup_df['day_of_week_sin'] = np.sin(2 * np.pi * dup_df['day_of_week_num'] / 7)
    dup_df['day_of_week_cos'] = np.cos(2 * np.pi * dup_df['day_of_week_num'] / 7)

    dup_df['month_num'] = pd.Categorical(dup_df['month'], categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                                                          'September', 'October', 'November', 'December'], ordered=True).codes
    dup_df['month_sin'] = np.sin(2 * np.pi * dup_df['month_num'] / 12)
    dup_df['month_cos'] = np.cos(2 * np.pi * dup_df['month_num'] / 12)

    dup_df.drop(columns=['hour', 'day_of_week', 'month', 'day_of_week_num', 'month_num'], inplace=True)

    ## feature 6
    ## Create a 24-hour rolling standard deviation for 'Global_active_power'
    dup_df['Global_active_power_rolling_std24'] = dup_df['Global_active_power'].rolling(window=24).std()


    dup_df = pd.get_dummies(dup_df, drop_first=True)
    dup_df.dropna(inplace=True)

    return dup_df

dup_df = create_features(data_df)

feature_list = dup_df.columns.tolist()


if __name__ == "__main__":
    print(dup_df.head())
    print(dup_df)
    print(feature_list)