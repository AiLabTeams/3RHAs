
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from catboost import CatBoostRegressor


# List of Excel files to load
file_paths_all = ['Au_Ag 3RHA.xlsx', 'Ag_Au 3RHA.xlsx']
file_name_all = ['Au_Ag 3RHA', 'Ag_Au 3RHA']
# for ii, file_path in enumerate(file_paths_all):  # 同时获取索引和文件路径
#     print(f"正在处理第 {ii} 个文件: {file_path}")
for ii, file_path in enumerate(file_paths_all):
    file_name = file_name_all[ii]
    # Define the columns and y_column
    x_columns = ['Oad1', 'Inc1', 'dOad', 'dInc', 'Oad2', 'Inc2']
    y_column = 'CD_abs'

    # Create empty lists to store various importance values and scores
    r2_scores_catboost = []
    rmse_scores_catboost = []


    # Dictionaries to hold feature importances
    feature_importances_dict_cat = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
    perm_importances_dict_cat = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}

    # Loop through each Excel file
    # for file_path in file_paths:
        # Read the Excel file
    df = pd.read_excel(file_path)

    # Feature engineering: add interaction features
    df['Oad1_Inc1'] = df['Oad1'] * df['Inc1']
    df['Oad2_Inc2'] = df['Oad2'] * df['Inc2']

    # Update feature columns
    x_columns_extended = x_columns + ['Oad1_Inc1', 'Oad2_Inc2']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[x_columns_extended], df[y_column], random_state=42)

    # CatBoost model
    cat_model = CatBoostRegressor(random_seed=42, verbose=0)
    param_grid_cat = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [100, 200, 300]
    }
    grid_search_cat = GridSearchCV(estimator=cat_model, param_grid=param_grid_cat, cv=5, n_jobs=-1, scoring='r2')
    grid_search_cat.fit(X_train, y_train)
    best_model_cat = grid_search_cat.best_estimator_

    # Fit the model and predict
    best_model_cat.fit(X_train, y_train)
    y_pred_cat = best_model_cat.predict(X_test)

    # Calculate R² and RMSE for CatBoost
    r2_cat = r2_score(y_test, y_pred_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
    r2_scores_catboost.append(r2_cat)
    rmse_scores_catboost.append(rmse_cat)

    # Get feature importances for CatBoost
    feature_importances_cat = best_model_cat.get_feature_importance()
    for feature, importance in zip(x_columns_extended, feature_importances_cat):
        feature_importances_dict_cat[feature].append(importance)

    # Permutation importance for CatBoost
    perm_importance_cat = permutation_importance(best_model_cat, X_test, y_test, n_repeats=10, random_state=42)
    for feature, importance in zip(x_columns_extended, perm_importance_cat.importances_mean):
        perm_importances_dict_cat[feature].append(importance)


    print("\nCatBoost Results:")
    summary_df_cat = pd.DataFrame({
        'File': file_path,
        'R²': r2_scores_catboost,
        'RMSE': rmse_scores_catboost
    })
    print(summary_df_cat)

    # Save the summary results to CSV
    summary_df_cat.to_csv(f'catboost_summary_results_{file_name}.csv', index=False)

    # Create dataframes for feature importances
    feature_importances_df_cat = pd.DataFrame(feature_importances_dict_cat, index=[file_path])
    perm_importances_df_cat = pd.DataFrame(perm_importances_dict_cat, index=[file_path])

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_importances_df_cat.T, annot=True, cmap='coolwarm')
    plt.title(f'CatBoost Feature Importance Heatmap_{file_name}')
    plt.tight_layout()
    plt.savefig(f'catboost_feature_importance_heatmap_{file_name}.png')
    # plt.show()
    plt.close()


    plt.figure(figsize=(12, 8))
    sns.heatmap(perm_importances_df_cat.T, annot=True, cmap='coolwarm')
    plt.title(f'CatBoost Permutation Importance Heatmap_{file_name}')
    plt.tight_layout()
    plt.savefig(f'catboost_permutation_importance_heatmap_{file_name}.png')
    # plt.show()
    plt.close()

    # Save the feature importances to CSV
    feature_importances_df_cat.to_csv(f'catboost_feature_importances_{file_name}.csv')
    perm_importances_df_cat.to_csv(f'catboost_permutation_importances_{file_name}.csv')