import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Load data
train_df = pd.read_csv('/Users/aidin07/Desktop/AI/floar_dask_graph/train.csv')
test_df = pd.read_csv('/Users/aidin07/Desktop/AI/floar_dask_graph/test.csv')

# Initial data inspection
print("\n=== Data Preview ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# EDA: Plot numeric relationships
plt.figure(figsize=(12, 6))
sns.regplot(x='LotArea', y='GrLivArea', data=train_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Lot Area vs Living Area (with Regression Line)')
plt.tight_layout()
plt.show()

# Preprocessing setup
numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
categorical_features = train_df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X = train_df.drop('SalePrice', axis=1)
y = np.log1p(train_df['SalePrice'])  # log(1+x) transform for robustness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Stacking Regressor
stackreg = StackingRegressor(
    estimators=[
        ('gbr', GradientBoostingRegressor(
            learning_rate=0.01, max_depth=12, max_features=0.1,
            min_samples_leaf=25, n_estimators=1000)),
        ('xgb', XGBRegressor(
            colsample_bytree=0.8, gamma=0, learning_rate=0.1,
            max_depth=3, min_child_weight=2, n_estimators=300)),
        ('cat', CatBoostRegressor(verbose=0)),
        ('lgb', lgb.LGBMRegressor(
            learning_rate=0.05, n_estimators=200, num_leaves=20)),
        ('rfr', RandomForestRegressor(
            max_depth=15, min_samples_split=3, n_estimators=500))
    ],
    final_estimator=VotingRegressor(
        estimators=[
            ('gbr', GradientBoostingRegressor(
                learning_rate=0.01, max_depth=12, max_features=0.1,
                min_samples_leaf=25, n_estimators=1000)),
            ('xgb', XGBRegressor(
                colsample_bytree=0.8, gamma=0, learning_rate=0.1,
                max_depth=3, min_child_weight=2, n_estimators=300)),
            ('ridge', Ridge(alpha=10, solver='lsqr'))
        ],
        weights=[2, 3, 1]
    )
)

# Training
print("\n=== Training Stacking Regressor ===")
stackreg.fit(X_train_transformed, y_train)

# Prediction and inverse log transform
y_pred = stackreg.predict(X_test_transformed)
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Enhanced Evaluation Metrics
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'R² (%)': r2_score(y_true, y_pred) * 100,
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    print("\n" + "="*50)
    print("REGRESSION PERFORMANCE METRICS".center(50))
    print("="*50)
    for name, value in metrics.items():
        if name in ['R² (%)', 'MAPE']:
            print(f"{name:<10}: {value:>10.2f}%")
        else:
            print(f"{name:<10}: {value:>10.2f}")
    print("="*50)
    
    return metrics

metrics = evaluate_regression(y_test_original, y_pred_original)
input("Press Enter to continue to residual plots...")  # Pause to see metrics

# Residual Analysis
residuals = y_test_original - y_pred_original

print("\n" + "="*70)
print("ОЦЕНКА КАЧЕСТВА РЕГРЕССИОННОЙ МОДЕЛИ".center(70))
print("="*70)
print("R-квадрат (Коэффициент детерминации):".ljust(45), f"{metrics['R² (%)']:.2f}%")
print("Среднеквадратичная ошибка (MSE):".ljust(45), f"{metrics['MSE']:.2f}")
print("Корень MSE (RMSE):".ljust(45), f"{metrics['RMSE']:.2f}")
print("Средняя абсолютная ошибка (MAE):".ljust(45), f"{metrics['MAE']:.2f}")
print("Средняя абсолютная % ошибка (MAPE):".ljust(45), f"{metrics['MAPE']:.2f}%")
print("="*70)

print("\nФАКТОРЫ ВЛИЯНИЯ НА ТОЧНОСТЬ:")
print("-"*70)
print("1. Мультиколлинеарность:".ljust(20), "Высокая корреляция между признаками → ненадёжные коэффициенты")
print("2. Выбросы:".ljust(20), "Искажают предсказания, особенно в MSE/RMSE")
print("3. Переобучение:".ljust(20), "Слишком сложная модель → плохая обобщающая способность")
print("="*70)

# Residual Plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residual Distribution')
plt.xlabel('Prediction Error ($)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_original, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_original, y=y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted.Price ($)')
plt.title(f'Actual vs Predicted Prices (R² = {metrics["R² (%)"]:.2f}%)')
plt.tight_layout()
plt.show()