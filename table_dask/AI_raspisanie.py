import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import time

# Устанавливаем стиль графика
plt.style.use("dark_background")

# Глобальные определения
groups = ["CS-21", "CS-22", "CS-23", "CS-25"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
slots_per_day = 3
total_slots = 15

subject_multiset = (
    ["DSA"] * 3 +
    ["IAI"] * 2 +
    ["APython"] * 2 +
    ["English"] * 5 +
    ["Geografy"] * 1 +
    ["Manas"] * 1 +
    ["Kyrgyz"] * 1
)
    # Декодирует вектор одной группы в упорядоченное расписание
    # 1. Определяет порядок предметов с помощью сортировки индексов.
    # 2. Формирует расписание, разбивая предметы на 5 дней (по slots_per_day в день).
def decode_group_schedule(vector):
    permutation = np.argsort(vector)
    ordered_subjects = [subject_multiset[i] for i in permutation]
    schedule = [ordered_subjects[i * slots_per_day:(i + 1) * slots_per_day] for i in range(5)]
    return schedule

def decode_solution(solution_vector):
        # Декодирует весь вектор решения в расписание для всех групп
    # 1. Преобразует вектор в матрицу, где каждая строка — вектор группы.
    # 2. Для каждой группы вызывает decode_group_schedule и собирает в словарь.
    solution_matrix = solution_vector.reshape((len(groups), total_slots))
    group_schedules = {}
    for i, group in enumerate(groups):
        group_schedules[group] = decode_group_schedule(solution_matrix[i])
    return group_schedules
# Проверяем повторяющиеся предметы в расписании каждой группы
def fitness(solution_vector, penalty_per_duplicate=10):
    schedules = decode_solution(solution_vector)
    penalty = 0
    
    for group, sched in schedules.items():
        for day_sched in sched:
            counts = {}
            for subj in day_sched:
                counts[subj] = counts.get(subj, 0) + 1
            penalty += sum((cnt - 1) * penalty_per_duplicate for cnt in counts.values() if cnt > 1)
# Проверяем повторяющиеся предметы в расписании всех групп по дням
    for d in range(5):
        subj_counts = {}
        for group in groups:
            for subj in schedules[group][d]:
                subj_counts[subj] = subj_counts.get(subj, 0) + 1
        penalty += sum((cnt - 1) * penalty_per_duplicate for cnt in subj_counts.values() if cnt > 1)
    
    return penalty

def pso_optimize_with_history(n_particles=50, max_iter=200):
    # Определяем размерность пространства решений (количество параметров в одной частице)
    # dim — общее число слотов для всех групп (группы × временные слоты)
    dim = len(groups) * total_slots
    lower_bounds = np.zeros(dim)  
    upper_bounds = np.ones(dim)   
    
    particles = np.random.uniform(lower_bounds, upper_bounds, (n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    personal_best = particles.copy()
    personal_best_scores = np.array([fitness(p) for p in particles])
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    
    w, c1, c2 = 0.5, 2, 2
    history = [global_best_score]
    
    for iter in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                              c1 * r1 * (personal_best[i] - particles[i]) +
                              c2 * r2 * (global_best - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], lower_bounds, upper_bounds)
            score = fitness(particles[i])
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                if score < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
        history.append(global_best_score)
        if global_best_score == 0:
            break
    
    return global_best, global_best_score, history

best_solution, best_score, history = pso_optimize_with_history()

plt.figure(figsize=(12, 6))
plt.plot(history, color='#00FFAA', marker='o', linestyle='-', linewidth=2, markersize=5, label='Best Score')
plt.xlabel('Iterations', fontsize=14, fontweight='bold', color='white')
plt.ylabel('Global Best Score', fontsize=14, fontweight='bold', color='white')
plt.title('PSO Convergence', fontsize=16, fontweight='bold', color='#FFD700')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show(block=False)

time.sleep(5)

# Устанавливаем стиль текста без цвета и эффектов
console = Console(style="bold white on black")
schedule = decode_solution(best_solution)

console.print("\nЛучшее расписание:\n")

for group in groups:
    console.print(f"\n{group}:\n")
    table = Table(show_header=True, header_style="bold white", show_lines=True)
    table.add_column("Day", justify="center", style="white")
    table.add_column("Lessons", justify="left", style="white")
    
    for day, subjects in zip(days, schedule[group]):
        table.add_row(day, ", ".join(subjects))
    
    console.print(table)
    console.print()

plt.show()

