from sympy import solve, series, factorial, oo, sinh, symbols
import scipy.integrate as integrate
from numpy.linalg import cond, solve
import matplotlib.pyplot as plt
import math

import pandas as pd

import numpy as np

import warnings
warnings.filterwarnings("ignore")

N = 10
lambd = -0.5

a=0
b=1

f = lambda x: x + 0.5
delta = lambda x,y: 1 if x==y else 0

# Вычисляем скалярное произведение f1(x) и f2(x) на [0,1] путем численного интегрирования
def dotL2(f1,f2):
    return integrate.quad(lambda x: f1(x)*f2(x), 0, 1)[0]

# Вычисляем ряд Тейлора для sh(xy) относительно y, начиная с y=0 и ограничиваемся N
# Удаляем члены более высокого порядка и проделываем то же самое относительно x
x, y = symbols('x y')

sinh_taylor = series(sinh(x*y), y, 0, N).removeO().series(x, 0, N).removeO()

print(sinh_taylor)


# Решение уравнения интегрального типа методом сингулярного интегрального уравнения
def singular_kernel(N):
    # Создаем список анонимных функций An, которые представляют собой члены разложения в ряд
    # Тейлора функции x^k/k! для аппроксимации ядра интегрального уравнения
    An = [lambda x, k=k: x ** k / math.factorial(k) for k in range(1, 2 * N, 2)]
    # Создаем список анонимных функций Bn, которые представляют собой многочлены степени k для
    # аппроксимации функции правой части уравнения
    Bn = [lambda x, k=k: x ** k for k in range(1, 2 * N, 2)]

    # Нулевая матрица (матрица системы уравнений) и нулевой вектор (вектор правой части уравнения)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Заполняем матрицу A и вектор b
    for i in range(N):
        for j in range(N):
            A[i, j] = delta(i, j) - lambd * dotL2(Bn[i], An[j])

    for i in range(N):
        b[i] = dotL2(Bn[i], f)

    # Решаем систему линейных уравнений, где an - вектор коэффициентов аппроксимации An(x)
    an = np.linalg.solve(A, b)

    # u(x) - сумма функции правой части f(x) и линейной комбинации аппроксимацией An(x) с
    # найденными коэффициентами an
    u = lambda x: f(x) + lambd * sum([an[i] * An[i](x) for i in range(N)])

    # Создаем массив точек на [0,1] для оценки u(x)
    x = np.linspace(0, 1, 1000)

    # Вычисляем невязку между левой и правой частью уравнению для найденного приближенного решения u(x)
    def loss(u):
        left_side = lambda x: u(x) - lambd * dotL2(lambda y: np.sinh(x * y), u)
        right_side = f
        left_side_val = np.array([left_side(xi) for xi in x])
        return np.max(np.abs(left_side_val - f(x)))

    # Возвращаем приближенное решение u(x) и функцию для оценки невязки
    return u, loss(u)

# Выводим невязки
data = {'N': [], 'Невязка': []}

N_values = range(1, 10)

for N in N_values:
    _, loss = singular_kernel(N)
    data['N'].append(N)
    data['Невязка'].append(loss)

df = pd.DataFrame(data)

print(df)

# Приближенное решение уравнения (функция u(x))
u_true, _ = singular_kernel(40)

# Определяем ядро уравнения интегрального типа
def K(x, y):
    return np.sinh(x * y)

# Создаем равномерную сетку на [a,b] с использованием N узлов
def create_grid(a, b, N):
    return np.linspace(a, b, N)

# Вычисляем дискретное скалярное произведение функции k(x,y) и вектора u на сетке x с шагом h
def dotL2dis(k, u, x, h):
    integral_approx = 0
    for i, xi in enumerate(x):
        integral_approx += k(xi, x) * u[i] * h
    return integral_approx

# Численный метод для решения уравнения с использованием квадратурных формул
# Создаем сетку x, строим матрицу системы уравнений, решаем систему и возвращаем вектор решения
def mech_quad(a, b, N, lambd):
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)
    # Создаем единичную матрицу
    A = np.eye(N)
    # Вычисляем значение функции правой части уравнения в узлах сетки
    F = f(x)

    # Заполняем матрицу A, учитывая ядро K(x,y) и параметр lambd. m - индекс столбца, n - индекс строки
    # m соответствует краевым узлам - квадратная формула с весом h/3
    # m нечетный - квадратная формула с весом 4h/3
    # m четный - квадратная формула с весом 2h/3
    for n in range(N):
        for m in range(N):
            if m == 0 or m == N - 1:
                cm = h / 3
            elif m % 2 == 0:
                cm = 2 * h / 3
            else:
                cm = 4 * h / 3
            A[n, m] -= lambd * cm * K(x[n], x[m])

    # Решаем систему линейных уравнений и возвращаем U - численное решение уравнения
    U = solve(A, F)
    return U

# Вычисляем ошибку между численным решением u и точным решением u_true, используя максимальное абсолютное отклонение
def loss(u, x):
    return np.max(np.abs(u - u_true(x)))

# Решаем уравнение интегрального типа на [0,1], используя 10 узлов сетки и параметр lambd
U = mech_quad(0, 1, 10, lambd)

# Численное решение для различных N и оценка ошибки для каждого решения
Ns = [11, 21, 41, 81, 101, 201]

results = pd.DataFrame(columns=['N', 'Оценка ошибки'])

# Вычисляем шаг сетки для текущего N; создаем равномерную сетку на [a,b]; численное решение уравнения;
# вычисляем оценку ошибки и выводим результат
for N in Ns:
    h = (b - a) / (N - 1)
    x = np.linspace(a, b, N)
    u = mech_quad(a, b, N, lambd)
    current_loss = loss(u, x)
    results = pd.concat([results, pd.DataFrame({'N': [N], 'Оценка ошибки': [current_loss]})], ignore_index=True)

print(results)