"""
Краевая задача для ОДУ 2-го порядка на отрезке [1, 2]:

        x(x + 1) y'' + (x + 2) y' - y = x + 1/x,
        y'(1) = 3/2,
        4 y'(2) + y(2) = 13 + 4 ln 2.

Точное решение:
        y(x) = x + 7/2 + 1/x + (x/2 + 1) ln|x|.

Нужно: реализовать метод стрельбы и конечно-разностный метод, решить задачу,
оценить погрешность по Рунге-Ромбергу и сравнением с точным решением.
Шаг в задании не указан - берём h = 0.1, как в 4.1 (и 2h = 0.2 для Р-Р).
"""

import math

# Метод Рунге-Кутты 4-го порядка уже написан и проверен в задании 4.1.
from lab4_1_cauchy import runge_kutta4, grid_size



# Делим исходное уравнение на x(x + 1) (на [1, 2] это не ноль):
#   y'' + p·y' + q·y = f

def p(x):
    return (x + 2) / (x * (x + 1))


def q(x):
    return -1 / (x * (x + 1))


def f(x):
    return (x ** 2 + 1) / (x ** 2 * (x + 1))


def second_derivative(x, y, dy):
    """
    Для метода стрельбы то же уравнение в виде (4.28): y'' = F(x, y, y'),
        y'' = f(x) - p(x) y' - q(x) y.
    """
    return f(x) - p(x) * dy - q(x) * y


def exact(x):
    """Точное решение y = x + 7/2 + 1/x + (x/2 + 1) ln|x|."""
    return x + 3.5 + 1 / x + (x / 2 + 1) * math.log(abs(x))


A_END, B_END = 1.0, 2.0
LEFT = (0.0, 1.0, 1.5)                         # 0*y(1) + 1*y'(1) = 3/2
RIGHT = (1.0, 4.0, 13 + 4 * math.log(2))       # 1*y(2) + 4*y'(2) = 13 + 4 ln 2


# МЕТОД СТРЕЛЬБЫ
def initial_state(left, eta):
    """
    Превращаем левое граничное условие α y(a) + β y'(a) = A в начальные
    условия задачи Коши с одним неизвестным параметром η .

    - Если β = 0, то неизвестна производная: y'(a) = η.
    - Иначе неизвестным считаем само значение y(a) = η, а производную
      выражаем из условия: y'(a) = (A - α η) / β.
    """
    alpha, beta, A = left
    if beta == 0:
        return [A / alpha, eta]
    return [eta, (A - alpha * eta) / beta]


def solve_cauchy(F, a, b, left, eta, h):
    """
    Решает задачу Коши для системы
        y' = z,
        z' = F(x, y, z)
    с начальными условиями из initial_state методом Рунге-Кутты 4-го порядка.
    """
    rhs = lambda x, Y: [Y[1], F(x, Y[0], Y[1])]
    return runge_kutta4(rhs, a, b, initial_state(left, eta), h)


def residual(F, a, b, left, right, eta, h):
    """
     Φ(η) = δ y(b; η) + γ y'(b; η) - B.
    промах
    """
    delta, gamma, B = right
    xs, Ys = solve_cauchy(F, a, b, left, eta, h)
    y_b, dy_b = Ys[-1]
    return delta * y_b + gamma * dy_b - B, xs, Ys


def shooting(F, a, b, left, right, h, eta0=1.0, eta1=0.8, eps=1e-10,
             max_iter=50, verbose=True):
    """

    метод секущих:

        η_{j+2} = η_{j+1} - (η_{j+1} - η_j) / (Φ(η_{j+1}) - Φ(η_j)) * Φ(η_{j+1}),

    т.е. проводим прямую через две последние точки (η, Φ(η)) и берём её
    пересечение с нулём. 

    """
    phi0, _, _ = residual(F, a, b, left, right, eta0, h)
    phi1, xs, Ys = residual(F, a, b, left, right, eta1, h)
    history = [(eta0, phi0), (eta1, phi1)]

    while abs(phi1) > eps:
        if len(history) > max_iter:
            raise RuntimeError("Метод стрельбы не сошёлся")
        if phi1 == phi0:
            raise ZeroDivisionError("Φ(η_j) = Φ(η_{j+1}) - секущая горизонтальна")
        eta2 = eta1 - (eta1 - eta0) / (phi1 - phi0) * phi1
        phi2, xs, Ys = residual(F, a, b, left, right, eta2, h)
        history.append((eta2, phi2))
        eta0, phi0, eta1, phi1 = eta1, phi1, eta2, phi2

    if verbose:
        print("  Итерации метода секущих:")
        print(f"  {'j':>3s}  {'η_j':>16s}  {'Φ(η_j)':>14s}")
        for j, (eta, phi) in enumerate(history):
            print(f"  {j:3d}  {eta:16.10f}  {phi:14.6e}")
        print()

    return xs, [Y[0] for Y in Ys], eta1, history


# КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД
def tridiagonal_solve(a, b, c, d):
    """
    Метод прогонки
    """
    n = len(b)
    P, Q = [0.0] * n, [0.0] * n
    P[0], Q[0] = -c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if denom == 0:
            raise ZeroDivisionError(f"Нулевой знаменатель в прогонке, i = {i}")
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    x = [0.0] * n
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def finite_difference(p, q, f, a, b, left, right, h, verbose=True):
    """
    Привожу уравнение к виду y'' + p·y' + q·y = f

    2) Во внутренниъ узлах 
           y'_k  ≈ (y_{k+1} - y_{k-1}) / (2h),
           y''_k ≈ (y_{k+1} - 2 y_k + y_{k-1}) / h^2.
       Подставляем в уравнение и умножаем на h^2:
           (1 - p_k h/2) y_{k-1} + (-2 + h^2 q_k) y_k + (1 + p_k h/2) y_{k+1} = h^2 f_k.
    
    3) На границах есть производная
           y'_0 ≈ (-3 y_0 + 4 y_1 - y_2) / (2h),
           y'_N ≈ (y_{N-2} - 4 y_{N-1} + 3 y_N) / (2h).

    """
    n = grid_size(a, b, h)
    xs = [a + k * h for k in range(n + 1)]

    # Коэффициенты внутренних уравнений: lo*y_{k-1} + mid*y_k + up*y_{k+1} = rhs.
    lo = [1 - p(x) * h / 2 for x in xs]
    mid = [-2 + h * h * q(x) for x in xs]
    up = [1 + p(x) * h / 2 for x in xs]
    rhs = [h * h * f(x) for x in xs]

    # Диагонали системы из N+1 уравнений для y_0..y_N.
    A = [0.0] * (n + 1)   # поддиагональ (при y_{k-1})
    Bd = [0.0] * (n + 1)  # главная диагональ (при y_k)
    C = [0.0] * (n + 1)   # наддиагональ (при y_{k+1})
    D = [0.0] * (n + 1)   # правая часть

    # Внутренние узлы
    for k in range(1, n):
        A[k], Bd[k], C[k], D[k] = lo[k], mid[k], up[k], rhs[k]

    # Левая граница, k = 0.
    # Условие α y_0 + β (-3y_0 + 4y_1 - y_2)/(2h) = A_l, умножаем на 2h:
    #     (2hα - 3β) y_0 + 4β y_1 - β y_2 = 2h A_l.
    # y_2 = (rhs_1 - lo_1 y_0 - mid_1 y_1) / up_1.
    # Подставляем y_2 и собираем коэффициенты при y_0 и y_1:
    alpha, beta, A_l = left
    Bd[0] = 2 * h * alpha - 3 * beta + beta * lo[1] / up[1]
    C[0] = 4 * beta + beta * mid[1] / up[1]
    D[0] = 2 * h * A_l + beta * rhs[1] / up[1]

    # Правая граница, k = N.
    # Условие δ y_N + γ (y_{N-2} - 4y_{N-1} + 3y_N)/(2h) = B_r, умножаем на 2h:
    #     γ y_{N-2} - 4γ y_{N-1} + (2hδ + 3γ) y_N = 2h B_r.
    # y_{N-2} = (rhs - mid y_{N-1} - up y_N) / lo.
    delta, gamma, B_r = right
    m = n - 1
    A[n] = -4 * gamma - gamma * mid[m] / lo[m]
    Bd[n] = 2 * h * delta + 3 * gamma - gamma * up[m] / lo[m]
    D[n] = 2 * h * B_r - gamma * rhs[m] / lo[m]

    if verbose:
        print("  Трёхдиагональная система  a_k y_{k-1} + b_k y_k + c_k y_{k+1} = d_k:")
        print(f"  {'k':>3s}  {'x_k':>5s}  {'a_k':>10s}  {'b_k':>10s}  {'c_k':>10s}  {'d_k':>10s}")
        for k in range(n + 1):
            print(f"  {k:3d}  {xs[k]:5.2f}  {A[k]:10.5f}  {Bd[k]:10.5f}  {C[k]:10.5f}  {D[k]:10.5f}")
        print()

    return xs, tridiagonal_solve(A, Bd, C, D)


# РУНГЕ-РОМБЕРГ
def runge_romberg(ys_h, ys_2h, p):
    """
    Формула Рунге-Ромберга: погрешность решения с шагом h в общих узлах
    сеток h и 2h (узел j сетки 2h = узел 2j сетки h):
        R_h ≈ (y_h - y_2h) / (2^p - 1).
    """
    return [(ys_h[2 * j] - y2) / (2 ** p - 1) for j, y2 in enumerate(ys_2h)]


# ГРАФИК
def plot_results(filename, xs, sols):
    """Сверху - решения и точное, снизу - |погрешность| в логарифмическом масштабе."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    style = {
        "стрельба": dict(color="#2a78d6", marker="o", linestyle="-"),
        "конечные разности": dict(color="#eb6834", marker="s", linestyle="--"),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    fig.patch.set_facecolor(surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink2)
        ax.tick_params(colors=ink2)

    px = [xs[0] + (xs[-1] - xs[0]) * i / 300 for i in range(301)]
    ax1.plot(px, [exact(x) for x in px], color=ink, linewidth=1.5, label="точное решение")
    for name, ys in sols.items():
        st = style[name]
        ax1.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=6,
                 linestyle="none", label=name)
    ax1.set_ylabel("y", color=ink2)
    ax1.set_title("Краевая задача, h = 0.1 (вариант 26)", loc="left", color=ink)
    ax1.legend(frameon=False)

    for name, ys in sols.items():
        st = style[name]
        errs = [abs(exact(x) - y) for x, y in zip(xs, ys)]
        ax2.semilogy(xs, errs, linewidth=2, markersize=6, label=name, **st)
    ax2.set_xlabel("x", color=ink2)
    ax2.set_ylabel("|y_точн − y_k|", color=ink2)
    ax2.set_title("Абсолютная погрешность (лог. шкала)", loc="left", color=ink)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График сохранён: {filename}\n")


# ОСНОВНАЯ ПРОГРАММА
def main(h=0.1):
    a, b = A_END, B_END

    print("Задание 4.2. Краевая задача (вариант 26)")
    print("  x(x+1) y'' + (x+2) y' - y = x + 1/x,  x ∈ [1, 2]")
    print("  y'(1) = 3/2,   4y'(2) + y(2) = 13 + 4 ln 2")
    print(f"  Шаг h = {h}\n")

    # Метод стрельбы
    print("1) Метод стрельбы: y(1) = η неизвестно, y'(1) = 3/2,")
    print("   Φ(η) = y(2; η) + 4 y'(2; η) - (13 + 4 ln 2),  задача Коши - РК4.")
    xs, y_shoot, eta, _ = shooting(second_derivative, a, b, LEFT, RIGHT, h)
    print(f"   Найдено η* = y(1) = {eta:.10f}   (точно: y(1) = {exact(1.0):.10f})\n")

    # Конечно-разностный метод
    print("2) Конечно-разностный метод (2-й порядок, в т.ч. на границах):")
    _, y_fd = finite_difference(p, q, f, a, b, LEFT, RIGHT, h)

    # Таблица решений
    print(f"Решение с шагом h = {h}:")
    print(f"  {'x':>5s}  {'точное':>12s}  {'стрельба':>12s}  {'кон. разн.':>12s}"
          f"  {'|ε| стр.':>10s}  {'|ε| к.р.':>10s}")
    for k, x in enumerate(xs):
        ye = exact(x)
        print(f"  {x:5.2f}  {ye:12.8f}  {y_shoot[k]:12.8f}  {y_fd[k]:12.8f}"
              f"  {abs(ye - y_shoot[k]):10.2e}  {abs(ye - y_fd[k]):10.2e}")
    print()

    # Рунге-Ромберг: те же методы с шагом 2h
    _, y_shoot2, _, _ = shooting(second_derivative, a, b, LEFT, RIGHT, 2 * h, verbose=False)
    xs2, y_fd2 = finite_difference(p, q, f, a, b, LEFT, RIGHT, 2 * h, verbose=False)
    R_shoot = runge_romberg(y_shoot, y_shoot2, p=4)   # РК4 - 4-й порядок
    R_fd = runge_romberg(y_fd, y_fd2, p=2)            # разности - 2-й порядок

    print(f"Оценка погрешности по Рунге-Ромбергу (h = {h} и 2h = {2 * h:g}):")
    print(f"  {'x':>5s}  {'R стрельба':>11s}  {'ε стрельба':>11s}  {'R кон.р.':>11s}  {'ε кон.р.':>11s}")
    for j, x in enumerate(xs2):
        ye = exact(x)
        print(f"  {x:5.2f}  {R_shoot[j]:11.2e}  {ye - y_shoot[2 * j]:11.2e}"
              f"  {R_fd[j]:11.2e}  {ye - y_fd[2 * j]:11.2e}")
    print("  R - оценка (y_h - y_2h)/(2^p - 1), ε - истинная погрешность y_точн - y_h.\n")

    print("Итог: максимальная погрешность на отрезке")
    for name, ys, R in (("стрельба (p = 4)", y_shoot, R_shoot),
                        ("кон. разности (p = 2)", y_fd, R_fd)):
        true_max = max(abs(exact(x) - y) for x, y in zip(xs, ys))
        print(f"  {name:>22s}:  max|ε| = {true_max:.2e},  max|R| = {max(map(abs, R)):.2e}")
    print()
   
    print("  У конечных разностей R занижает ошибку: при делении h пополам ошибка")
    print("  падает в 3.0, 3.45, 3.70, 3.84 раза (-> 4 = 2^2), т.е. при h = 0.1")
    print("  метод ещё не вышел на асимптотику C*h^2. Порядок при этом второй.\n")

    plot_results("lab4_2_solution.png", xs, {"стрельба": y_shoot, "конечные разности": y_fd})
    check_textbook()


# ПРОВЕРКА НА ПРИМЕРЕ ИЗ МЕТОДИЧКИ
def check_textbook():
    """
    Пример 4.9: y'' = e^x + sin y, y(0) = 1, y(1) = 2, h = 0.1, η0 = 1.0, η1 = 0.8,
    ε = 1e-4. В методичке: η4 = -0.160862503, y(0.5) = 1.17434 (табл. 4.15, 4.16).
    Здесь условия 1-го рода: α = 1, β = 0 слева и δ = 1, γ = 0 справа.
    """
    F = lambda x, y, dy: math.exp(x) + math.sin(y)
    xs, ys, eta, hist = shooting(F, 0.0, 1.0, (1.0, 0.0, 1.0), (1.0, 0.0, 2.0), 0.1,
                                 eta0=1.0, eta1=0.8, eps=1e-4, verbose=False)
    print("Проверка на примере 4.9 методички (y'' = e^x + sin y):")
    print(f"  итераций: {len(hist) - 1},  η = {eta:.9f}   (в методичке η4 = -0.160862503)")
    print(f"  y(0.5) = {ys[5]:.5f}   (в методичке 1.17434)\n")


if __name__ == "__main__":
    main()
