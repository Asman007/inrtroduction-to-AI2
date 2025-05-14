import pandas as pd  # Для работы с данными в формате таблиц (чтение CSV, создание DataFrame)
import matplotlib.pyplot as plt  # Для создания графиков (например, scatter-графиков)
import seaborn as sns  # Для улучшенной визуализации данных (например, стилизованные scatter-графики)
from sklearn.cluster import KMeans  # Для выполнения кластеризации методом K-Means
from sklearn.preprocessing import StandardScaler  # Для нормализации данных (приведения признаков к единому масштабу)

# Загрузка данных
df = pd.read_csv('penguins.csv')

# Удаляем строки с пропусками в числовых признаках
df = df.dropna(subset=["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"])

# Удаляем аномалии (например, отрицательное значение в flipper_length_mm)
df = df[df["flipper_length_mm"] > 0]

# Используем все числовые признаки
X = df.select_dtypes(include='number')

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Финальная модель
k_opt = 3  # Фиксированное число кластеров
kmeans = KMeans(n_clusters=k_opt, random_state=10000)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Печать кластеров и признаков
print("\nДанные с кластерами и признаками:")
print(df[['Cluster', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']])

# Подготовка данных точек для визуализации
points = df[['culmen_length_mm', 'flipper_length_mm', 'Cluster', 'sex']].copy()
print("\nДанные точек для графика (culmen_length_mm, flipper_length_mm, Cluster, sex):")
print(points.head(10))

# Визуализация на паре признаков (culmen_length_mm vs flipper_length_mm)
sns.scatterplot(data=df, x='culmen_length_mm', y='flipper_length_mm', hue='Cluster', style='sex', palette='Set2', s=100)
plt.title('Кластеры пингвинов по длине клюва и плавника')
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()