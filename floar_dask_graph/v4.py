import pandas as pd  # Работа с табличными данными  
import matplotlib.pyplot as plt  # Визуализация данных  
from sklearn.linear_model import LinearRegression  # Линейная регрессия  
from sklearn.metrics import r2_score  # Оценка качества модели  
import numpy as np  # Работа с массивами и матрицами  

# Загружаем набор данных
file_path = "apple_products.csv"  # Путь к файлу с данными о продуктах Apple
df = pd.read_csv(file_path)

# Преобразуем столбцы в числовой формат, заменяя ошибки на NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Mrp'] = pd.to_numeric(df['Mrp'], errors='coerce')

# Удаляем строки с пропущенными значениями в столбцах Price и Mrp
df = df.dropna(subset=['Price', 'Mrp'])

# Создаем модель линейной регрессии
X = df[['Mrp']].values  # Независимая переменная: максимальная розничная цена (MRP)
y = df['Price'].values  # Зависимая переменная: фактическая цена (Price)

model = LinearRegression()
model.fit(X, y)  # Обучаем модель

# Генерируем предсказания для диапазона значений MRP
mrp_range = np.linspace(df['Mrp'].min(), df['Mrp'].max(), 100).reshape(-1, 1)
price_predictions = model.predict(mrp_range)

# Строим график результатов
plt.figure(figsize=(10, 6))
plt.scatter(df['Mrp'], df['Price'], alpha=0.5, label="Фактические данные")  # Точки данных
plt.plot(mrp_range, price_predictions, color='red', linewidth=2, label="Прогноз")  # Линия предсказаний
plt.xlabel('MRP (Максимальная розничная цена)')
plt.ylabel('Price (Фактическая цена)')
plt.title('Прогноз цены iPhone на основе MRP')
plt.legend()
plt.grid(True)

# Вычисляем и выводим коэффициент детерминации R²
accuracy = r2_score(y, model.predict(X))
print(f"Accuracy (R² score): {accuracy:.4f}")

# Отображаем график
plt.show()