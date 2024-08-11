from math import cos

import numpy as np
import pandas as pd
import sympy
from sympy.abc import x

# Строит массив многочленов Якоби степени от 0 до n
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

# Генерирует массивы координатных функций и их производных, используя многочлены Якоби
# Генерирует массивы координатных функций и их производных, используя многочлены Якоби
def fun(k, n):
    phi = [x] * n
    dphi = [x] * n
    ddphi = [x] * n
    jacs = Jac(k, n)
    for i in range(n):
        phi[i] = (1 - x ** 2) * jacs[i]
        phi[i] = sympy.simplify(phi[i])
        dphi[i] = -2 * x * jacs[i] + (1 - x ** 2) * sympy.diff(jacs[i], x)
        dphi[i] = sympy.simplify(dphi[i])
        ddphi[i] = -2 * jacs[i] + 2 * x * sympy.diff(jacs[i], x) - 2 * x * sympy.diff(jacs[i], x) + (1 - x ** 2) * sympy.diff(jacs[i], x, x)
        ddphi[i] = sympy.simplify(ddphi[i])
    return phi, dphi, ddphi
# Вычисляет значения координатных функций и их производных в точке y
def fun_vals(k, n, y):
    phi = [0] * n
    dphi = [0] * n
    ddphi = [0] * n
    jacs = Jacval(k, n, y)
    for i in range(n):
        phi[i] = (1 - y ** 2) * jacs[i]
        dphi[i] = -2 * y * jacs[i] + (1 - y ** 2) * jacs[i]
        ddphi[i] = -2 * jacs[i] + 2 * y * jacs[i] - 2 * y * jacs[i] + (1 - y ** 2) * jacs[i]
    return phi, dphi, ddphi

# Реализует метод Ритца для решения дифференциальных уравнений
def Ritz(k, n):
    pols = Jac(k, n)
    phis, dphis, ddphis = fun(k, n)
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    f = 1 + x
    for i in range(n):
        h = f * phis[i]
        b[i] = sympy.integrals.integrate(h, (x, -1, 1))
        for j in range(n):
            A[i][j] = sympy.integrals.integrate(ddphis[i] * ddphis[j] / (2 + x), (x, -1, 1))
    coeffs = np.linalg.solve(A, b)
    return coeffs, A, b

# Реализует метод коллокации для решения дифференциальных уравнений
def colloc(k, n):
    nodes = Cheb_nodes(n, -1, 1)
    f = lambda x: 1 + x
    p = lambda x: 1 / (2 + x)
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] = f(nodes[i])
        phi, dphi, ddphi = fun_vals(k, n, nodes[i])
        for j in range(n):
            A[i][j] = -ddphi[j] / (2 + nodes[i]) + np.cos(nodes[i]) * phi[j]
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

# Сравнивает приближенное решение с точным решением в заданных точках.
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

# Создает таблицу из полученных значений.
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
    table = pd.DataFrame(data=values, columns=column, index=indexes)
    table.columns.name = "n"
    return (table)


def main():
    dots = [-0.5, 0.0, 0.5]

    # Метод Ритца
    print("Метод Ритца")
    val_Ritz = []
    for i in range(3, 8):
        coeffs, A, b = Ritz(1, i)
        val_Ritz.append(final_solution(coeffs, dots))
    result_table = make_table(val_Ritz)
    print("Расширенная матрица системы:")
    print("А = ", A)
    print("Число обусловленности матрицы А = ", np.linalg.cond(A))
    print("b = ", b)
    print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
    print(result_table)

    # Метод коллокации
    print("Метод коллокации")
    val_colloc = []
    for i in range(3, 8):
        coeffs, A, b = colloc(1, i)
        val_colloc.append(final_solution(coeffs, dots))
    result_table = make_table(val_colloc)
    print("Расширенная матрица системы:")
    print("А = ", A)
    print("Число обусловленности матрицы А = ", np.linalg.cond(A))
    print("b = ", b)
    print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
    print(result_table)


if __name__ == "__main__":
    main()
