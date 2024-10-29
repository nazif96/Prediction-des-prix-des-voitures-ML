import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from data import load_and_prepare_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import load
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.svm import SVR


def calculate_evaluation_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def evaluate_model():
    X_train = load('X_train.joblib')
    X_test = load('X_test.joblib')
    y_test = load('y_test.joblib')


    model_lr = load('modele_entraine_regression_lineaire.joblib')
    model_rf = load('modele_entraine_random_forest_pipeline.joblib')
    model_gb = load('modele_entraine_gradient_boosting_pipeline.joblib')
    model_knn = load('modele_entraine_knn.joblib') 
    model_mlp=load('modele_entraine_mlp.joblib')
    model_lasso=load('modele_entraine_lasso_pipeline.joblib')
    model_svm=load('modele_svm_regression.joblib')

    
 

    rmse_lr, r2_lr = calculate_evaluation_metrics(model_lr, X_test, y_test)
    rmse_rf, r2_rf = calculate_evaluation_metrics(model_rf, X_test, y_test)
    rmse_gb, r2_gb = calculate_evaluation_metrics(model_gb, X_test, y_test)
    rmse_knn, r2_knn = calculate_evaluation_metrics(model_knn, X_test, y_test)
    rmse_mlp,r2_mlp=calculate_evaluation_metrics(model_mlp,X_test, y_test)
    rmse_lasso,r2_lasso=calculate_evaluation_metrics(model_lasso,X_test,y_test)
    rmse_svm,r2_svm=calculate_evaluation_metrics(model_svm,X_test,y_test)


    resultats = {
        "Modèle": ["Linéaire", "Forêt Aléatoire", "Boosting", "KNN", "MLP", "Lasso", "SVM"],
        "RMSE": [rmse_lr, rmse_rf, rmse_gb, rmse_knn, rmse_mlp, rmse_lasso, rmse_svm],
        "R2": [r2_lr, r2_rf, r2_gb, r2_knn, r2_mlp, r2_lasso, r2_svm]
    }

    df_resultats = pd.DataFrame(resultats)
    df_resultats.to_csv('resultats_modeles.csv', index=False)
    
    rmse_lasso,r2_lasso=calculate_evaluation_metrics(model_lasso,X_test, y_test)
    rmse_svm,r2_svm=calculate_evaluation_metrics(model_svm,X_test, y_test)
    print("Régression Linéaire: RMSE =", rmse_lr, ", R² =", r2_lr)
    residuals_lr = y_test - model_lr.predict(X_test)
    
    print("Forêt Aléatoire: RMSE =", rmse_rf, ", R² =", r2_rf)
    residuals_rf = y_test - model_rf.predict(X_test)

    print("Boosting: RMSE =", rmse_gb, ", R² =", r2_gb)
    residuals_gb = y_test - model_gb.predict(X_test)

    print("K Nearest Neighbors: RMSE =", rmse_knn, ", R² =", r2_knn)

    print("MLPRegressor : RMSE "  , rmse_mlp,", R² =",r2_mlp )


    print("lasso"  , rmse_lasso,", R² =",r2_lasso )

    print("svm", rmse_svm,",R²=",r2_svm)
    
    voiture_exemple = {
        'Modèle': ['bentley'],
        'Kilométrage': [10000],
        'Puissance_CH': [500],
       'Transmission_Boîte automatique': [True]
    }
    df_voiture = pd.DataFrame(voiture_exemple)

    
    cols_manquantes = [col for col in X_test.columns if col not in df_voiture.columns]
    df_cols_manquantes = pd.DataFrame(0, index=np.arange(len(df_voiture)), columns=cols_manquantes)
   
    df_voiture_encoded = pd.concat([df_voiture, df_cols_manquantes], axis=1)
    df_voiture_encoded = df_voiture_encoded.reindex(columns=X_test.columns, fill_value=0)

    prix_predit = model_knn.predict(df_voiture_encoded)
    print("Prix prédit pour la voiture Bentley spécifiée:", prix_predit[0])






if __name__ == "__main__":
    evaluate_model()


