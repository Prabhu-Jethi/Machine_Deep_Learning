import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


df = pd.read_csv(r"D:\Python\ML\30_DAYS_ML_DL\Project_1\data\processed\loan_processed_data.csv")

def model_train(df):

    ## Spliting
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    #### Comparing 3 models: To get the better result ####

    ## 1. Logistic Regression
    lr_pipe = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(max_iter = 1000, class_weight='balanced'))
    ])
    lr_pipe.fit(X_train, y_train)

    ###### Cross validation
    lr_cv = cross_val_score(
        lr_pipe,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )
    print("\nLogistic_Regression_CV:")
    print(lr_cv.mean())
    print(lr_cv.std())
    

    ## 2. RandomForest Classifier
    rf_pipe = Pipeline([
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=6,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    rf_pipe.fit(X_train, y_train)

    ###### Cross validation
    rf_cv = cross_val_score(
        rf_pipe,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )
    print("\nRandom_Forest_CV:")
    print(rf_cv.mean())
    print(rf_cv.std())
    
    ## 3. DecisionTree Classifier
    tree_pipe = Pipeline([
        ('tree', DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=6,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    tree_pipe.fit(X_train, y_train)

    ###### Cross validation
    dt_cv = cross_val_score(
        tree_pipe,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )
    print("\nDecision_Tree_CV:")
    print(dt_cv.mean())
    print(dt_cv.std())

    ## 4. Xgboost Classifier
    xgb_pipe = Pipeline([
        ('xgb', XGBClassifier(
            eval_metric='logloss',
            max_depth=4,
            n_estimators=200,
            learning_rate=0.03,
            random_state=42,
            class_weight='scale_pos_weight'
        ))
    ])
    xgb_pipe.fit(X_train, y_train)
    
    ###### Cross validation
    xgb_cv = cross_val_score(
        xgb_pipe,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )
    print("\nXGBoost CV:")
    print(xgb_cv.mean())
    print(xgb_cv.std())

    #### Evaluating #####

    def evaluate(model, X_test, y_test):
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]

        return {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }
    lr_result = evaluate(lr_pipe, X_test, y_test)
    rf_result = evaluate(rf_pipe, X_test, y_test)
    dt_result = evaluate(tree_pipe, X_test, y_test)
    xgb_result = evaluate(xgb_pipe, X_test, y_test)
    
    ###### Comaparing all models to find the best one ###
    comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        lr_result["Accuracy"],
        dt_result["Accuracy"],
        rf_result["Accuracy"],
        xgb_result["Accuracy"]
    ],
    "Precision": [
        lr_result["Precision"],
        dt_result["Precision"],
        rf_result["Precision"],
        xgb_result["Precision"]
    ],
    "Recall": [
        lr_result["Recall"],
        dt_result["Recall"],
        rf_result["Recall"],
        xgb_result["Recall"]
    ],
    "F1 Score": [
        lr_result["F1 Score"],
        dt_result["F1 Score"],
        rf_result["F1 Score"],
        xgb_result["F1 Score"]
    ],
    "ROC-AUC": [
        lr_result["ROC-AUC"],
        dt_result["ROC-AUC"],
        rf_result["ROC-AUC"],
        xgb_result["ROC-AUC"]
    ],
    "CV ROC-AUC (Mean)": [
        lr_cv.mean(),
        dt_cv.mean(),
        rf_cv.mean(),
        xgb_cv.mean()
    ],
    "CV Std": [
        lr_cv.std(),
        dt_cv.std(),
        rf_cv.std(),
        xgb_cv.std()
    ]
    })

    comparison = comparison.sort_values(
    by="CV ROC-AUC (Mean)",
    ascending=False)

    print("\nComparision_between_metrics_of_models:\n", comparison)


    ## Grid search
    param_grid = {
        'lr__max_iter': [1000, 2000, 3000]
    }
    grid = GridSearchCV(
        lr_pipe,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc'
    )
    print(grid.fit(X_train, y_train))

    #### Best model 
    best_model = grid.best_score_
    print(best_model)


if __name__ == "__main__":
    model_train(df)



    


    