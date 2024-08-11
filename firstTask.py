import numpy as np
import sympy
import pandas as pd

from sympy.abc import x

# Строит массив многочленов Якоби
def Jac(k, n):
    pj = [x] * (n + 1)

    for j in range(n + 1):
        if j == 0:
            pj[j] = 1
        elif j == 1:
            pj[j] = (1 + k) * x
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            pj[j] = (tmp_1 * x * pj[j - 1] - tmp_2 * pj[j - 2]) / tmp_3
    return pj

# Строит массив значение многочленов Якоби в точке y
def Jacval(k, n, y):
    vals = [0] * (n + 1)
    for j in range(n + 1):
        if j == 0:
            vals[j] = 1
        elif j == 1:
            vals[j] = (1 + k) * y
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            vals[j] = (tmp_1 * y * vals[j - 1] - tmp_2 * vals[j - 2]) / tmp_3
    return vals

# Строит массивы координатных функций и их производных
def fun(k, n):
    phi = [x] * (n)
    dphi = [x] * (n)
    ddphi = [x] * (n)


    jacs = Jac(k, n)
    djacs = Jac(k - 1, n + 1)
    for i in range(n):
        phi[i] = (1 - x ** 2) * jacs[i]
        phi[i] = sympy.simplify(phi[i])

        dphi[i] = (-2) * (i + 1) * (1 - x ** 2) ** (k - 1) * djacs[i + 1]
        dphi[i] = sympy.simplify(dphi[i])

        tmp1 = (k - 1) * (1 - x ** 2) ** (k - 2) * djacs[i + 1]
        tmp2 = (1 - x ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacs[i]
        ddphi[i] = (-2) * (i + 1) * ( tmp1 + tmp2 )
        ddphi[i] = sympy.simplify(ddphi[i])

    return phi, dphi, ddphi

# Строит массивы значений координатных функций и их производных
def fun_vals(k, n, y):
    phi = [0] * n
    dphi = [0] * n
    ddphi = [0] * n

    jacs = Jacval(k, n, y)
    djacs = Jacval(k - 1, n + 1, y)
    for i in range(n):
        phi[i] = (1 - y ** 2) * jacs[i]

        dphi[i] = (-2) * (i + 1) * (1 - y ** 2) ** (k - 1) * djacs[i + 1]

        tmp1 = (1 - y ** 2) ** (k - 2) * (k - 1) * djacs[i + 1]
        tmp2 = (1 - y ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacs[i]
        ddphi[i] = (-2) * (i + 1) * ( tmp1 + tmp2 )

    return phi, dphi, ddphi


# Метод Ритца
def Ritz(k, n):
    # Вызываем функция Якоби и метода координатых функций и их производных
    pols = Jac(k, n)
    phis, dphis, ddphis = fun(k, n)

    # Создаем матрицу и вектор для построения системы линейных уравнений метода Ритца
    A = np.zeros((n, n))
    b = np.zeros((n, 1))

    x = sympy.symbols('x')
    # Определяем функцию f(x)
    f = 2 - x

    # 1) Вычисляем произведение f(x) на одну из координатных функций
    # 2) Вычисляем значение интеграла от h на [-1,1] (элементы вектора правой части)
    for i in range(3):
        h = f * phis[i]
        b[i] = sympy.integrals.integrate(h, (x, -1, 1))

    # Задаем значения узлов и соответствующие веса для формулы Гаусса
    x1 = 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7))
    x2 = 1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))
    c1 = (322 + 13 * np.sqrt(70)) / 900
    c2 = (322 - 13 * np.sqrt(70)) / 900
    x_i = [-x2, -x1, 0, x1, x2]
    c_i = [c2, c1, 128 / 225, c1, c2]

    # Вычисляем значения координатных функция и их производных в этом узле
    arr = [fun_vals(k, n, x_i[i]) for i in range(5)]

    # Вычисление интеграла по формуле Гаусса
    def Gauss(nodes, coefs, i, j):
        s = 0
        # Перебираем все узлы формулы Гаусса
        for k in range(len(nodes)):
            tmp_1 = (((nodes[k] + 4) / (nodes[k] + 5)) * arr[k][1][j] * arr[k][1][i] + np.exp(nodes[k] ** 4 / 4) *
                     arr[k][0][i] * arr[k][0][j])
            s += coefs[k] * tmp_1
        return s

    for i in range(n):
        for j in range(n):
            A[i][j] = Gauss(x_i, c_i, i, j)
    # Вычисляем коэффициенты для решения методом Ритца
    coeffs = np.linalg.solve(A, b)
    return coeffs, A, b

# Строит узлы многочлена Чебышева первого рода
def Cheb_nodes(n, a, b):
    arr = []
    for i in range(1, n + 1):
        tmp1 = (1 / 2) * (a + b)
        tmp2 = (1 / 2) * (b - a)
        arr.append(np.cos((2 * i - 1) * np.pi / (2 * n)))
    return arr

# Метод коллокации
def colloc(k, n):
    # Генерируем узлы метода коллокации с помощью Чебышева
    nodes = Cheb_nodes(n, -1, 1)
    # Функции, которые представляют правую часть дифура и коэффициенты при производных в уравнении
    f = lambda x: 2 - x
    p = lambda x: (x + 4) / (x + 5)
    dp = lambda x: 1 / (x + 5) ** 2
    r = lambda x: np.exp(x**4/4)

    # Матрица и вектор для построения системы линейных уравнений метода коллокации
    A = np.zeros((n, n))
    b = np.zeros((n, 1))

    # 1) Вычисляем значение правой части дифура в узле
    # 2) Вычисляем значения координатных функций и их производных в узле
    for i in range(n):
        b[i] = f(nodes[i])
        phi, dphi, ddphi = fun_vals(k, n, nodes[i])
        for j in range(n):
            # Вычисляем значения произведения и второй, первой производной и координатной функции в узле
            tmp1 = p(nodes[i]) * ddphi[j]
            tmp2 = dp(nodes[i]) * dphi[j]
            tmp3 = r(nodes[i]) * phi[j]
            A[i][j] = (-1) * (tmp1 + tmp2) + tmp3
    # Решаем систему линейных уравнений, чтобы найти коэффициенты приближенного решения
    coeffs = np.linalg.solve(A, b)
    return coeffs, A, b

# Сравнивает с точным решением
def final_solution(coeffs, dots):
    dot1, dot2, dot3 = dots[0], dots[1], dots[2]

    exact_value = [0.721373, 0.813764, 0.541390]
    res = [0.0] * 3
    n = len(coeffs)

    phi_dot1 = fun_vals(1, n, dot1)[0]
    phi_dot2 = fun_vals(1, n, dot2)[0]
    phi_dot3 = fun_vals(1, n, dot3)[0]

    for i in range(3):
        res[0] += coeffs[i] * phi_dot1[i]
        res[1] += coeffs[i] * phi_dot2[i]
        res[2] += coeffs[i] * phi_dot3[i]

    errs = [exact_value[k] - res[k] for k in range(3)]
    arr = [np.round(res[0], 5),
           np.round(res[1], 5),
           np.round(res[2], 5),
           np.round(errs[0], 5),
           np.round(errs[1], 5),
           np.round(errs[2], 5)]
    return arr

# Строит таблицу по значениям
def make_table(values):
    column = [
        "y(-0.5)",
        "y(0)",
        "y(0.5)",
        "y* - y(-0.5)",
        "y* - y(0)",
        "y* - y(0.5)"
    ]
    indexes = [3, 4, 5, 6, 7]
    table = pd.DataFrame(data = values, columns=column, index=indexes)
    table.columns.name = "n"
    return(table)

# Строим таблицу результатов для метода Ритца
dots = [-0.5, 0.0, 0.5]
val_Ritz = []
coeffs, A, b = [], [], []
for i in range(3, 8):
    coeffs, A, b = Ritz(1, i)
    val_Ritz.append(final_solution(coeffs, dots))
result_table = make_table(val_Ritz)
print("Метод Ритца")
print("Расширенная матрица системы:")
print("А = ", A)
print("Число обусловленности матрицы А = ", np.linalg.cond(A))
print("b = ", b)
print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
result_table

# Строим таблицу результатов для метода коллокации
dots = [-0.5, 0.0, 0.5]
val_colloc = []
coeffs, A, b = [], [], []
for i in range(3, 8):
    coeffs, A, b = colloc(1, i)
    val_colloc.append(final_solution(coeffs, dots))
result_table = make_table(val_colloc)
print("Метод коллокации")
print("Расширенная матрица системы:")
print("А = ", A)
print("Число обусловленности матрицы А = ", np.linalg.cond(A))
print("b = ", b)
print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
result_table

