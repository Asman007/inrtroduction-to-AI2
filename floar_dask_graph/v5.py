import pandas as pd  # Работа с табличными данными  
import matplotlib.pyplot as plt  # Визуализация данных  
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Полиномиальные признаки и нормализация  
from sklearn.linear_model import LinearRegression  # Линейная регрессия  
from sklearn.metrics import r2_score  # Оценка качества модели  
import numpy as np  # Работа с массивами и матрицами  

# Загружаем набор данных
file_path = "train.csv"  # Путь к файлу с данными о домах
df = pd.read_csv(file_path)

# Выбираем необходимые столбцы и преобразуем их в числовой формат, заменяя ошибки на NaN
columns_to_use = ['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt']
df = df[columns_to_use].apply(pd.to_numeric, errors='coerce')

# Удаляем строки с пропущенными значениями, чтобы избежать проблем в анализе
df = df.dropna()

# Фильтруем выбросы: исключаем 1% самых дорогих домов и домов с экстремально большой площадью
df = df[(df['SalePrice'] < df['SalePrice'].quantile(0.99)) &
        (df['GrLivArea'] < df['GrLivArea'].quantile(0.99))]

# Определяем признаки (независимые переменные) и целевую переменную (зависимая переменная)
X = df[['GrLivArea', 'OverallQual', 'YearBuilt']]  # Площадь, качество постройки, год постройки
y = df['SalePrice']  # Цена продажи дома

# Масштабируем признаки, чтобы привести их к единому диапазону значений (улучшает точность модели)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создаем полиномиальные признаки степени 3 для учета нелинейных зависимостей
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_scaled)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_poly, y)

# Генерируем предсказания для диапазона значений жилой площади
area_range = np.linspace(df['GrLivArea'].min(), df['GrLivArea'].max(), 100).reshape(-1, 1)

# Дополняем область значений средними значениями других признаков для корректного предсказания
area_range_scaled = scaler.transform(np.hstack([area_range, np.full_like(area_range, df['OverallQual'].mean()),
                                                 np.full_like(area_range, df['YearBuilt'].mean())]))
area_range_poly = poly.transform(area_range_scaled)
price_predictions = model.predict(area_range_poly)

# Строим график зависимости цены дома от жилой площади
plt.figure(figsize=(10, 6))
plt.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.5, label="Фактические данные")  # Точки фактических данных
plt.plot(area_range, price_predictions, color='red', linewidth=2, label="Прогноз (полиномиальная регрессия)")  # Линия предсказаний
plt.xlabel('GrLivArea (Жилая площадь, кв. футы)')
plt.ylabel('SalePrice (Цена продажи, $)')
plt.title('Прогноз цены дома на основе жилой площади и других факторов')
plt.legend()
plt.grid(True)

# Вычисляем и выводим коэффициент детерминации R² (показывает, насколько хорошо модель объясняет данные)
accuracy = r2_score(y, model.predict(X_poly))
print(f"Accuracy (R² score): {accuracy:.4f}")

# Отображаем график с результатами предсказаний
plt.show()