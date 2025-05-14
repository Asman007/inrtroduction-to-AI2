import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка и предобработка данных
data = pd.read_csv("penguins.csv")

# Удаляем строки с пропусками в числовых признаках
data = data.dropna(subset=["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"])

# Удаляем аномалии (например, отрицательные значения в flipper_length_mm)
data = data[data["flipper_length_mm"] > 0]

# Выбираем числовые признаки для кластеризации
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
X = data[features]

# Нормализуем данные, чтобы все признаки имели одинаковый масштаб
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Метод локтя для выбора числа кластеров
inertias = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# График метода локтя
plt.plot(range(1, 7), inertias, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Choosing Number of Clusters")
plt.show()

# 3. Применяем K-Means с 5 кластерами (можно изменить после анализа графика)
kmeans = KMeans(n_clusters=5, random_state=42)
data["cluster"] = kmeans.fit_predict(X_scaled)

# 4. Визуализация кластеров (используем два признака для простоты)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["culmen_length_mm"], y=data["flipper_length_mm"], hue=data["cluster"], palette="viridis")
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.title("Penguin Clusters Based on Physical Features")
plt.show()

# 5. Анализ распределения пола по кластерам
# Удаляем строки с NA в колонке sex для чистоты анализа
data_sex = data.dropna(subset=["sex"])
sex_summary = data_sex.groupby("cluster")["sex"].value_counts(normalize=True).unstack().fillna(0)
print("\nРаспределение пола по кластерам (доли):")
print(sex_summary)

# Визуализация распределения пола
sex_summary.plot(kind="bar", stacked=True, figsize=(8, 6))
plt.xlabel("Cluster")
plt.ylabel("Proportion")
plt.title("Sex Distribution Across Clusters")
plt.legend(title="Sex")
plt.show()