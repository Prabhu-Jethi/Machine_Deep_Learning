import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"D:\Python\ML\30_DAYS_ML_DL\Project_1\data\raw\loan_data.xls")

def preprocess(df):

    ## remove duplicates
    df = df.drop_duplicates()

    ## Numerical cols
    num_cols = df.select_dtypes(exclude='object').columns

    ## Categorical cols
    cat_cols = df.select_dtypes(include='object').columns

    ## Fill missing numerical values
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')  
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    ## Fill missing categorical values
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

if __name__ == "__main__":
    print(preprocess(df))