import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import scipy.integrate as integrate
import pandas as pd
from scipy.special import jacobi
from numpy.linalg import eig

import matplotlib.pyplot as plt

k = 15.7503
l = 19.58201

x0 = -1
x1 = -0.7978525


# Определяем коэффициенты перед u
def p(x):
    return k * x + l


def q(x):
    return k ** 2 / (k * x + l) - k ** 3 * x


# Вычисляем численную производную вектора vec по x
def compute_derivative(vec, dx):
    dvec = np.zeros_like(vec)
    # Для внутренних элементов вычисляем значения производных методом центральной разности
    # Производная вычисляется как разность значений соседних элементов vec / на двойной шаг dx
    dvec[1:-1] = (vec[2:] - vec[:-2]) / (2 * dx)
    # Для 1-го элемента используем одностороннюю разность
    # Разности между 2-ым и 1-ым элементами vec / на шаг dx
    dvec[0] = (vec[1] - vec[0]) / dx
    # Для последнего: разность между последним и предпоследним элементами vec / на шаг dx
    dvec[-1] = (vec[-1] - vec[-2]) / dx
    return dvec


# Вычисляем L^2 норму вектора vec
# Интегрируем квадрат значений вектора с использованием метода трапеций и берем sqrt из результата
def l2_norm_integral(vec, dx):
    return np.sqrt(np.trapz(vec ** 2, dx))


# Вычисляем скалярное произведение f1(x) и f2(x) на [x_0, x_1], используя quad для численного интегрирования
def dotL2(f1, f2):
    return integrate.quad(lambda x: f1(x) * f2(x), x0, x1)[0]


# Вычисляем интеграл с весом для пары функций w1(x) и w2(x) с их производными dw1(x), dw2(x)
# а также весами p(x), q(x), используя quad
def braces(w1, w2, dw1, dw2):
    return integrate.quad(lambda x: p(x) * dw1(x) * dw2(x) + q(x) * w1(x) * w2(x), x0, x1)[0]


# Вычисляем производную функции fun(x) в точке x, используя метод центральной разности с шагом h
def d_fun(fun, x):
    h = 1e-5
    return (fun(x + h) - fun(x - h)) / (2 * h)


# Оценка функций p(x), q(x) (оцениваем диапазоны значений p(x) и q(x) на [x_0, x_1])
x = np.linspace(x0, x1, 1000)

# Вычисляем min и max функции p(x), построив массив значений p(x)
p_min = np.min(p(np.linspace(x0, x1, 1000)))
p_max = np.max(p(np.linspace(x0, x1, 1000)))

# Вычисляем min и max функции q(x), построив массив значений q(x)
q_min = np.min(q(np.linspace(x0, x1, 1000)))
q_max = np.max(q(np.linspace(x0, x1, 1000)))


# Вычисляем собственное значение при заданном k
def eigen_val(k, p, q):
    return np.pi ** 2 * k ** 2 / (x1 - x0) ** 2 * p + q


# Вычисляем собственный вектор для заданного k
def eigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    # Возвращаем собственный вектор, нормированный на единичную длину
    # Нормируем при помощи деления fun(x) на sqrt из скалярного произведения fun(x) на саму себя, используя dotL2
    return lambda x: fun(x) / np.sqrt(dotL2(fun, fun))


# Вычисляем производную собственного вектора для заданного k
def deigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    # Вычисляем норму собственной функции, чтобы найти длину вектора в пространстве функций
    c = np.sqrt(dotL2(fun, fun))
    # Вычисляем значение a, используемое в производной собственной функции
    a = np.pi / (x1 - x0) * k
    # Возвращаем производную собственной функции и нормализуем на значение c
    return lambda x: (a * np.cos(a * x) - ck * a * np.sin(a * x)) / c


# Вычисляем вторую производную собственного вектора для заданного k
def ddeigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    c = np.sqrt(dotL2(fun, fun))
    a = np.pi / (x1 - x0) * k
    return lambda x: (-a * a * np.sin(a * x) - ck * a * a * np.cos(a * x)) / c


for i in range(2):
    plt.plot(x, eigen_vec(i + 1)(x), label=f'Собственная функция {i + 1}')
plt.legend()
plt.show()


# Вычисляем невязку между левой и правой частями уравнения для СФ
def loss(i, pm, qm, x):
    # Определяем левую часть уравнения для СФ
    left_side = lambda x: -k * deigen_vec(i)(x) - p(x) * ddeigen_vec(i)(x) + q(x) * eigen_vec(i)(x)
    # Определяем правую часть уравнения для СФ
    right_side = lambda x: eigen_val(i, pm, pm) * eigen_vec(i)(x)
    # Находим max абсолютное значение разности между левой и правой частями
    # Оцениваем, насколько близки эти части (чем меньше значение, тем ближе СФ к удовлетворяющей уравнению форме)
    return np.max(np.abs(right_side(x) - left_side(x)))


# Таблица с результатами вычислений СЗ и их невязки для 1-ых двух СФ при разных p и q
df_mm = pd.DataFrame()

df_mm['p'] = ['min', 'max']
df_mm['λ_1'] = [eigen_val(1, p_min, q_min), eigen_val(1, p_max, q_max)]
df_mm['λ_1 невязка'] = [loss(1, p_min, q_min, x), loss(1, p_max, q_max, x)]

df_mm['λ_2'] = [eigen_val(2, p_min, q_min), eigen_val(2, p_max, q_max)]
df_mm['λ_2 невязка'] = [loss(2, p_min, q_min, x), loss(2, p_max, q_max, x)]

print(df_mm)

print(
    f'Первое собственное число {braces(eigen_vec(1), eigen_vec(1), deigen_vec(1), deigen_vec(1)) / dotL2(eigen_vec(1), eigen_vec(1))}')
print(
    f'Второе собственное число {braces(eigen_vec(2), eigen_vec(2), deigen_vec(2), deigen_vec(2)) / dotL2(eigen_vec(2), eigen_vec(2))}')

N = 7


# Используем jacobi для вычисления полиномов Якоби (нужны для создания базисных функций)
# Вычисляем базисные функции и нормируем их
def basic_func(k):
    fun = lambda x: (1 - ((2 * x - x0 - x1) / (x1 - x0)) ** 2) * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))
    c = np.sqrt(dotL2(fun, fun))
    return lambda x: fun(x) / c


# Вычисляем производные базисных функций и нормируем их
# Если k=0, используем формулу для вычисления производной базисной функции 1-го порядка
# Иначе 2-го порядка
def dbasic_func(k):
    fun = lambda x: (1 - ((2 * x - x0 - x1) / (x1 - x0)) ** 2) * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))
    c = np.sqrt(dotL2(fun, fun))
    if k == 0:
        return lambda x: (-4 * (2 * x - x0 - x1) / (x1 - x0) ** 2 * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))) / c
    return lambda x: (-4 * (2 * x - x0 - x1) / (x1 - x0) ** 2 * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0)) + 2 / (
            x1 - x0) * (1 - ((2 * x - x0 - x1) / (x1 - x0)) ** 2) * (k + 5) / 2 * jacobi(k - 1, 3, 3)(
        (2 * x - x0 - x1) / (x1 - x0))) / c


# Создаем матрицу Галеркина и вычисляем ее элементы
G_l = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        G_l[i, j] = braces(basic_func(i), basic_func(j), dbasic_func(i), dbasic_func(j))

# Вычисляем СЗ и СВ матрицы
vals, vecs = eig(G_l)
# Возвращаем индексы элементов массива vals в порядке возрастания
sorted_indices = np.argsort(vals)
# Упорядочиваем СЗ в порядке возрастания
vals = vals[sorted_indices]
# Упорядочиваем СВ в соответствии с отсортированными индексами
# Каждый столбец матрицы vecs соответствует СВ и переупорядочивается так, чтобы они
# соответствовали новому порядку СЗ
vecs = vecs[:, sorted_indices]

print(f'Первое собственное число {vals[0]}')
print(f'Второе собственное число {vals[1]}')


# Создаем СФ на основе полученных в методе Галеркина коэффициентов
def eig_vec_r(n, coef):
    fun = lambda x: sum([basic_func(i)(x) * coef[i] for i in range(n)])
    return lambda x: fun(x)


for i in range(2):
    plt.plot(x, eig_vec_r(N, vecs[:, i])(x), label=f'Собственная функция {i + 1}')
plt.legend()
plt.show()


# Метод минимальных невязок для поиска минимального СЗ и соответствующего СВ матрицы Гамма
def scalar_product_method(Gamma_L, epsilon=1e-4):
    # Получаем размерность матрицы Гамма
    n = Gamma_L.shape[0]
    # Генерируем случайный начальный вектор z
    z = np.random.rand(n)
    # Нормируем вектор z
    z /= np.linalg.norm(z)

    while True:
        # Решаем систему линейных уравнений Гамма для нахождения нового вектора
        z_new = np.linalg.solve(Gamma_L, z)

        # Вычисляем норму нового вектора и нормируем его
        z_new_norm = np.linalg.norm(z_new)
        z_new /= z_new_norm

        # Если разница между предыдущим и новым векторами меньше эпсилон, то завершаем цикл
        if np.linalg.norm(z_new - z) < epsilon:
            break

        # Обновляем вектор z
        z = z_new

    # Вычисляем min СЗ
    lambda_min = np.dot(z, np.dot(Gamma_L, z)) / np.dot(z, z)
    # Возвращаем min СЗ и соответствующий СВ
    return lambda_min, z


# Задаем min СЗ матрицы Галеркина и соответствующий СВ
lambd, coefs = scalar_product_method(G_l)

for i in range(2):
    plt.plot(x, sum([basic_func(i)(x) * coefs[i] for i in range(N)]), label=f'Собственная функция {i + 1}')
plt.legend()
plt.show()


# Таблица сравнения найденного min СЗ с точным значением
# Ваша функция table
def table(Nn):
    df = pd.DataFrame()
    df['n'] = np.array(Nn)
    lambd_list = []
    lambd_loss = []

    for n in Nn:
        G_l = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                G_l[i, j] = braces(basic_func(i), basic_func(j), dbasic_func(i), dbasic_func(j))
        lambd, coefs = scalar_product_method(G_l)
        lambd_list.append(lambd)
        lambd_loss.append(lambd - vals[0])
    df[r'$\lambda_1^{(n)}$'] = lambd_list
    df[r'$\lambda_1^{(n)}$-$\lambda_1^*$'] = lambd_loss

    return df

# Вывод таблицы в консоль
print(table([3, 4, 5, 6, 7]))
