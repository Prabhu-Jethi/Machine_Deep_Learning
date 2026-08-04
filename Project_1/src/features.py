import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.impute import SimpleImputer


## Raw data
df = pd.read_csv(r"D:\Python\ML\30_DAYS_ML_DL\Project_1\data\raw\loan_data.xls")

def create_features(df):

    ## create a copy of original data
    df_fe = df.copy()

    ## Numerical cols
    num_cols = df_fe.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy='median')
    df_fe[num_cols] = num_imputer.fit_transform(df_fe[num_cols])

    ## Categorical cols
    cat_cols = df_fe.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_fe[cat_cols] = cat_imputer.fit_transform(df_fe[cat_cols])

    ## Creating new features

    ## 1. Debt_income_ration: 
    df_fe['debt_income_ratio'] = (
        df['installment'] / df['log.annual.inc']
    )

    ## 2. Inquiries >= 5 months
    df_fe['many_inquiries'] = (
        df_fe['inq.last.6mths'] >= 5
    ).astype(int)

    ## 3. Delinquency
    df_fe['has_delinquency'] = (
        df_fe['delinq.2yrs'] > 0
    ).astype(int)

    ## 4. Public records
    df_fe['has_public_record'] = (
        df_fe['pub.rec'] > 0
    ).astype(int)

    ## 5. New FICO category
    bins = [300, 580, 670, 740, 800, 850]
    labels = [
        'poor', 'fair', 'good', 'very_good', 'excellent'
    ]
    df_fe['fico_category'] = pd.cut(
        df_fe['fico'],
        bins=bins,
        labels=labels
    )

    ### Encoding to numeric --> (One-hot encoding)
    df_fe = pd.get_dummies(df_fe, drop_first=True)

    ## All columns to list
    df_fe.columns.tolist()

    return df_fe

if __name__ == "__main__":
    df_fe = create_features(df)
    print(df_fe)
    