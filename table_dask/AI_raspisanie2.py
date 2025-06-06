import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
# from scipy.signal import savgol_filter
import time  # Для задержки

# Глобальные определения
groups = ["CS-21", "CS-22", "CS-23", "CS-25"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
slots_per_day = 3
total_slots = 15  # Количество слотов для группы за неделю

# Список предметов с требуемой частотой
subject_multiset = (
    ["DSA"] * 3 +
    ["IAI"] * 2 +
    ["APython"] * 2 +
    ["English"] * 5 +
    ["Geografy"] * 1 +
    ["Manas"] * 1 +
    ["Kyrgyz"] * 1
)

# Декодирование расписания для одной группы
def decode_group_schedule(vector):
    permutation = np.argsort(vector)  # Индексы для сортировки
    ordered_subjects = [subject_multiset[i] for i in permutation]  # Упорядоченные предметы
    schedule = [ordered_subjects[i * slots_per_day:(i + 1) * slots_per_day] for i in range(5)]  # Разбивка по дням
    return schedule

# Декодирование полного решения для всех групп
def decode_solution(solution_vector):
    solution_matrix = solution_vector.reshape((len(groups), total_slots))
    group_schedules = {}
    for i, group in enumerate(groups):
        group_schedules[group] = decode_group_schedule(solution_matrix[i])
    return group_schedules

# Функция приспособленности (fitness)
def fitness(solution_vector, penalty_per_duplicate=10):
    schedules = decode_solution(solution_vector)
    penalty = 0

    # Ограничение внутри группы: уникальность предметов в каждом дне
    for group, sched in schedules.items():
        for day_sched in sched:
            counts = {}
            for subj in day_sched:
                counts[subj] = counts.get(subj, 0) + 1
            penalty += sum((cnt - 1) * penalty_per_duplicate for cnt in counts.values() if cnt > 1)

    # Ограничение между группами: предмет не должен преподаваться более чем в одной группе в один день
    for d in range(5):
        subj_counts = {}
        for group in groups:
            for subj in schedules[group][d]:
                subj_counts[subj] = subj_counts.get(subj, 0) + 1
        penalty += sum((cnt - 1) * penalty_per_duplicate for cnt in subj_counts.values() if cnt > 1)

    return penalty

# Оптимизация PSO с историей
def pso_optimize_with_history(n_particles=30, max_iter=100):
    dim = len(groups) * total_slots  # 60
    lower_bounds = np.zeros(dim)
    upper_bounds = np.ones(dim)

    # Инициализация частиц
    particles = np.random.uniform(lower_bounds, upper_bounds, (n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    personal_best = particles.copy()
    personal_best_scores = np.array([fitness(p) for p in particles])
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # Параметры PSO
    w = 0.5  # Инерционный вес
    c1 = 1.0  # Когнитивный коэффициент
    c2 = 1.0  # Социальный коэффициент

    history = [global_best_score]  # История лучших значений

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

# Запуск PSO
best_solution, best_score, history = pso_optimize_with_history(n_particles=30, max_iter=100)



# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(history, label='Исходные значения', alpha=0.5)
plt.xlabel('iterations')
plt.ylabel('global Best')
plt.title('Сходимость PSO')
plt.legend()
plt.grid(True)
plt.show(block=False)  # Не блокируем выполнение программы

# Задержка для просмотра графика
time.sleep(5)

# Вывод расписания в терминал
console = Console()
schedule = decode_solution(best_solution)

console.print("\n[bold magenta]Лучшее расписание:[/bold magenta]\n")

for group in groups:
    console.print(f"[bold cyan]{group}:[/bold cyan]")
    table = Table(show_header=True, header_style="bold green")
    table.add_column("day", justify="center")
    table.add_column("lessons", justify="left")

    for day, subjects in zip(days, schedule[group]):
        table.add_row(day, ", ".join(subjects))

    console.print(table)
    console.print()

plt.show()