import datetime

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def main():
    print('Car Subscribe Prediction Pipeline')

    df = pd.read_csv('data/df_feaches.csv')

    # month, weekday do not have numeric nature,
    # transform numeric values to str
    df['month'] = df['month'].astype(str)
    df['weekday'] = df['weekday'].astype(str)

    # balance df providing same number of users with useful actions and not
    # delete random samples with useless actions
    np.random.seed(10)

    remove_n = df.shape[0] - len(df[df['useful_action'] == 1]) * 2
    drop_indices = np.random.choice(df[df['useful_action'] != 1].index, remove_n, replace=False)
    df = df.drop(drop_indices)

    X = df.drop('useful_action', axis=1)
    y = df['useful_action']

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(max_iter=400)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.2f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X,y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump({
        'model': best_pipe,
        'metadata':{
            'name': 'car prediction pipelne',
            'author': 'Anna Perepelitsa',
            'version': 1.0,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'accuracy': best_score
        }
    }, 'car_subscribe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
