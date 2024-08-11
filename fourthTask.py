import numpy as np
import matplotlib.pyplot as plt

def u_0(x):
    return x

def f(x, t):
    return 2

def explicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    for i in range(1, k + 1):
        for j in range(1, n + 1):
            sol[i, j] = sol[i - 1, j] + tau * (-c * (sol[i - 1, j] - sol[i - 1, j - 1]) / h + f(x[j], t[i - 1]))

    return sol

def pure_implicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 1 + c * tau / h
    off_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(off_diag * np.ones(n - 1), k = -1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = sol[i - 1, j] + tau * f(x[j], t[i - 1])

        sol[i, 1 : ] = np.linalg.solve(A, b[1 : ])

    return sol


def implicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 1 + c * tau / h
    under_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(under_diag * np.ones(n - 1), k=-1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = sol[i - 1, j - 1] + tau * f(x[j], t[i - 1])

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol


def symmetric_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 2 + 2 * c * tau / h
    under_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(under_diag * np.ones(n - 1), k=-1) + np.diag(under_diag * np.ones(n - 1),
                                                                                          k=1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = 2 * sol[i - 1, j - 1] + 2 * tau * f((x[j] + h / 2), (t[i] + tau / 2))

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol


def surface_drawing(t, x, sol):
    T, X = np.meshgrid(t, x)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, sol.T, cmap='viridis')

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(t, x)')
    ax.set_title('Solution of the PDE')
    plt.colorbar(surf)
    plt.show()

a = 1
T = 1
N = M = 100
h = tau = 1 / N
c = 1.0

if (c * tau / h <= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, M + 1)

u = explicit_scheme(f, c, tau, h, N, M, x, t)

surface_drawing(t, x, u)
plt.plot(x, u[6].T)

a = 1
T = 1
N = 700
M = 100
h = 1 / N
tau = 1 / M
c = 1.0

if (c * tau / h <= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, M + 1)

u = explicit_scheme(f, c, tau, h, N, M, x, t)

surface_drawing(t, x, u)
plt.plot(x, u[6].T)

a = 10
T = 3
N = 100
K = 200
h = a / N
tau = T / K
c = 1.0


x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = pure_implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[15].T)

a = 10
T = 3
N = 200
K = 100
h = a / N
tau = T / K
c = 1.0

if (c * tau / h >= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[6].T)

a = 10
T = 3
N = 1000
K = 100
h = a / N
tau = T / K
c = 1.0

if (c * tau / h >= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[8].T)

a = 10
T = 3
N = 100
K = 200
h = a / N
tau = T / K
c = 1.0

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = symmetric_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[100].T)

