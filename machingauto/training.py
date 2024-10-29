import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_and_prepare_data 
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def train_linear_regression_model(file_path):
 
    df = load_and_prepare_data('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    X = df.drop('Prix', axis=1) 
    y = df['Prix']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('linear_regression', LinearRegression())  
    ])

    
    param_grid = {}

    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    
    best_model_lr = grid_search.best_estimator_

  
    dump(best_model_lr, 'modele_entraine_regression_lineaire.joblib')
    
    
    dump(X_test, 'X_test.joblib')
    dump(y_test, 'y_test.joblib')
    dump(X_train, 'X_train.joblib')








def train_lasso_model(file_path,alpha=0.1):
    df = load_and_prepare_data(file_path)
    X = df.drop('Prix', axis=1)
    y = df['Prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso())
    ])

    
    param_grid = {
        'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

   
    best_model_lasso = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Meilleurs paramètres:", best_params)
    best_model_lasso.fit(X_train, y_train) 

    
    dump(best_model_lasso, 'modele_entraine_lasso_pipeline.joblib')

def train_random_forest_model(file_path):
    df = load_and_prepare_data(file_path)
    X = df.drop('Prix', axis=1)
    y = df['Prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestRegressor())
    ])

    
    param_grid = {
        'random_forest__n_estimators': [50, 100, 200],
        'random_forest__max_depth': [None, 10, 20, 30],
        'random_forest__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

   
    best_model_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Meilleurs paramètres:", best_params)

   
    best_model_rf.fit(X_train, y_train)

  
    dump(best_model_rf, 'modele_entraine_random_forest_pipeline.joblib')



def train_gradient_boosting_model(file_path):
    df = load_and_prepare_data(file_path)
    X = df.drop('Prix', axis=1)
    y = df['Prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gradient_boosting', GradientBoostingRegressor())
    ])

    
    param_grid = {
        'gradient_boosting__n_estimators': [50, 100, 200],
        'gradient_boosting__learning_rate': [0.01, 0.1, 0.2],
        'gradient_boosting__max_depth': [3, 4, 5],
        'gradient_boosting__min_samples_split': [2, 5, 10]
    }

   
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model_gb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Meilleurs paramètres:", best_params)

    
    best_model_gb.fit(X_train, y_train)

    
    dump(best_model_gb, 'modele_entraine_gradient_boosting_pipeline.joblib')


def train_knn_model_with_cross_validation(file_path, n_splits=5):
    df = load_and_prepare_data(file_path)
    X = df.drop('Prix', axis=1)
    y = df['Prix']

    params = {
        'n_neighbors': range(5, 30, 5),
        'weights': ['uniform', 'distance']
    }

    model_knn = KNeighborsRegressor(metric='manhattan')
    grid_search = GridSearchCV(model_knn, params, cv=n_splits, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    print("Meilleurs paramètres:", grid_search.best_params_)

    # Sauvegarder le modèle
    dump(best_model, 'modele_entraine_knn.joblib')


def train_mlp_model(file_path):
    df = load_and_prepare_data(file_path)
    X = df.drop('Prix', axis=1)
    y = df['Prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('mlp', MLPRegressor(max_iter=3000))  
    ])

    
    param_grid = {
        'mlp__hidden_layer_sizes': [(100, 100), (50, 50, 50)],
        'mlp__alpha': [0.001, 0.01, 0.1]
    }

    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    
    best_params = grid_search.best_params_
    best_model_mlp = grid_search.best_estimator_
    print("Meilleurs paramètres:", best_params)

    dump(best_model_mlp, 'modele_entraine_mlp.joblib')



def train_svm_model(file_path):
    df=load_and_prepare_data(file_path)
    X=df.drop('Prix',axis=1)
    y=df['Prix']
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('svm', SVR())  
    ])
    
    param_grid = {
        'svm__kernel': ['linear', 'rbf'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print("Meilleurs paramètres:", best_params)
    
    
    dump(best_model, 'modele_svm_regression.joblib')







if __name__ == "__main__":
    train_linear_regression_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_random_forest_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_gradient_boosting_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_knn_model_with_cross_validation('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_mlp_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_lasso_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    train_svm_model('C:\\Users\\Youssef\\Desktop\\M2\\MACHING\\Projet\\vehicules.csv')
    