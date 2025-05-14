# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Загрузка данных с обработкой ошибок
# try:
#     train_df = pd.read_csv('/Users/aidin07/Desktop/AI/floar_dask_graph/train.csv')
# except FileNotFoundError:
#     print("Ошибка: файл train.csv не найден!")
#     exit()

# # Проверяем наличие необходимых колонок
# required_columns = ['SalePrice', 'LotArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'YearBuilt']
# if not all(col in train_df.columns for col in required_columns):
#     print("Ошибка: в данных отсутствуют необходимые колонки!")
#     print("Ожидаются колонки:", required_columns)
#     exit()

# df = train_df.copy()

# # Целевая переменная
# y = df['SalePrice']

# # Выбираем признаки
# features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'YearBuilt']
# df = df.dropna(subset=features + ['SalePrice'])  # Удаляем строки с пропусками в признаках и целевой переменной

# # Преобразование данных (если нужно)
# # LotArea уже в числовом формате (площадь участка в квадратных футах)
# # BedroomAbvGr — количество спален над землёй, числовое
# # FullBath — количество ванных комнат, числовое
# # OverallQual — общая оценка качества, числовое
# # YearBuilt — год постройки, числовое
# X = df[features]

# # Разделение данных
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Обучение модели
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Предсказание
# y_pred = model.predict(X_test)

# # Оценка модели
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R² Score: {r2}")

# # График: Площадь участка vs. Цена
# plt.figure(figsize=(10, 6))
# plt.scatter(df['LotArea'], df['SalePrice'], color='blue', alpha=0.5)
# plt.xlabel("Площадь участка (кв. футы)")
# plt.ylabel("Цена продажи (доллары)")
# plt.title("Площадь участка vs. Цена продажи")
# plt.grid(True)
# plt.show()


