"""
    Вместо одного многочлена на всём отрезке строим НА КАЖДОМ отрезке
    [x_{i-1}, x_i] свой многочлен 3-й степени:
        S(x) = a_i + b_i (x - x_{i-1}) + c_i (x - x_{i-1})^2 + d_i (x - x_{i-1})^3,
        x_{i-1} <= x <= x_i,   i = 1..n   (у нас n = 4 отрезка).
    На концах - нулевая кривизна: S''(x0) = 0, S''(x4) = 0
    ("естественный" сплайн).

    Неизвестных 4n (a, b, c, d на каждом отрезке). Если из условий
    исключить a, b, d, остаётся система из n-1 уравнений только для
    c_2 .. c_n с ТРЁХДИАГОНАЛЬНОЙ матрицей (3.13), h_i = x_i - x_{i-1}:
        2(h1+h2) c2 + h2 c3                 = 3[(f2-f1)/h2 - (f1-f0)/h1],
        h_{i-1} c_{i-1} + 2(h_{i-1}+h_i) c_i + h_i c_{i+1}
                                            = 3[(f_i-f_{i-1})/h_i - (f_{i-1}-f_{i-2})/h_{i-1}],
        h_{n-1} c_{n-1} + 2(h_{n-1}+h_n) c_n = 3[(f_n-f_{n-1})/h_n - (f_{n-1}-f_{n-2})/h_{n-1}].
    Её решаем методом прогонки (как в лабе 1.2). Потом остальные
    коэффициенты по формулам (3.14):
        a_i = f_{i-1},
        b_i = (f_i - f_{i-1})/h_i - h_i (c_{i+1} + 2 c_i) / 3,   i < n,
        b_n = (f_n - f_{n-1})/h_n - 2 h_n c_n / 3,
        d_i = (c_{i+1} - c_i) / (3 h_i),                         i < n,
        d_n = -c_n / (3 h_n),
        c_1 = 0   (нулевая кривизна в x0).

"""

import os


# МЕТОД ПРОГОНКИ
def tridiagonal_solve(a, b, c, d):
    """
    Решение трёхдиагональной системы  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
    методом прогонки (подробно разобран в лабе 1.2).
    a - поддиагональ (a[0] = 0), b - главная диагональ,
    c - наддиагональ (c[-1] = 0), d - правые части.
    """
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n
    # Прямой ход: x_i = P_i x_{i+1} + Q_i.
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    # Обратный ход.
    x = [0.0] * n
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


# ПОСТРОЕНИЕ КУБИЧЕСКОГО СПЛАЙНА
def build_spline(xs, fs, verbose=True):
    """
    Строит естественный кубический сплайн по узлам xs и значениям fs.
    Возвращает (a, b, c, d) - списки коэффициентов с индексами 1..n.
    """
    n = len(xs) - 1

    # 1. Шаги сетки h_i = x_i - x_{i-1}, i = 1..n
    h = [0.0] * (n + 1)
    for i in range(1, n + 1):
        h[i] = xs[i] - xs[i - 1]

    """ 2. Трёхдиагональная система для c_2 .. c_n
    2.1 наклоны: sᵢ = (fᵢ − fᵢ₋₁) / hᵢ,    i = 1..n
    2.2 hᵢ₋₁·cᵢ₋₁ + 2(hᵢ₋₁ + hᵢ)·cᵢ + hᵢ·cᵢ₊₁ = 3·(sᵢ − sᵢ₋₁)
    low * c_{i-1} + diag * c_i + up * c_{i+1} = rhs. 
    коэффициенты: слева hᵢ₋₁, на диагонали 2(hᵢ₋₁ + hᵢ), справа hᵢ;
    правая часть: 3 × (разность соседних наклонов)."""
    low, diag, up, rhs = [], [], [], []
    for i in range(2, n + 1):
        low.append(h[i - 1] if i > 2 else 0.0)
        diag.append(2 * (h[i - 1] + h[i]))
        up.append(h[i] if i < n else 0.0)
        rhs.append(3 * ((fs[i] - fs[i - 1]) / h[i]
                        - (fs[i - 1] - fs[i - 2]) / h[i - 1]))

    if verbose:
        print("Система  для c_2 .. c_n:")
        for k in range(len(diag)):
            i = k + 2                      # номер неизвестной c_i
            terms = []
            if low[k]:
                terms.append(f"{low[k]:.4g}*c{i - 1}")
            terms.append(f"{diag[k]:.4g}*c{i}")
            if up[k]:
                terms.append(f"{up[k]:.4g}*c{i + 1}")
            print("   " + " + ".join(terms) + f" = {rhs[k]:.6f}")
        print()

    # Решаем прогонкой
    c_inner = tridiagonal_solve(low, diag, up, rhs)

    # 3. Коэффициенты c_i (c_1 = 0 - нулевая кривизна в x0)
    c = [0.0] * (n + 2)                   # c[n+1] нужен только как 0 в формулах
    for k in range(len(c_inner)):
        c[k + 2] = c_inner[k]
    c[1] = 0.0

    # 4. Остальные коэффициенты по формулам
    # aᵢ = fᵢ₋₁,    i = 1..n
    # bᵢ = sᵢ − hᵢ·(cᵢ₊₁ + 2cᵢ)/3,    i = 1..n−1
    # dᵢ = (cᵢ₊₁ − cᵢ) / (3hᵢ),
    a = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    d = [0.0] * (n + 1)
    for i in range(1, n + 1):
        a[i] = fs[i - 1]
        if i < n:
            b[i] = (fs[i] - fs[i - 1]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        else:
            # Последний отрезок: условие S''(x_n) = 0.
            b[i] = (fs[i] - fs[i - 1]) / h[i] - 2 * h[i] * c[i] / 3
            d[i] = -c[i] / (3 * h[i])


    return a, b, c[:n + 1], d


def find_segment(xs, x):
    """Номер отрезка i (1..n), такого что x_{i-1} <= x <= x_i."""
    n = len(xs) - 1

    tol = 1e-12
    for i in range(1, n + 1):
        if xs[i - 1] - tol <= x <= xs[i] + tol:
            return i
    raise ValueError(f"Точка {x} вне отрезка [{xs[0]}, {xs[-1]}]")


# считаем значение сплайна или его производной в точке x
def spline_value(xs, coefs, x, deriv=0):
    """
    Значение сплайна (deriv = 0) или его 1-й/2-й производной в точке x.
        S   = a + b t + c t^2 + d t^3,
        S'  = b + 2 c t + 3 d t^2,
        S'' = 2 c + 6 d t,
        t = X* − xᵢ₋₁,     
        S(X*) = aᵢ + bᵢ·t + cᵢ·t² + dᵢ·t³

    """
    a, b, c, d = coefs
    i = find_segment(xs, x)
    t = x - xs[i - 1]
    """ Sᵢ(x) = aᵢ + bᵢ·(x − xᵢ₋₁) + cᵢ·(x − xᵢ₋₁)² + dᵢ·(x − xᵢ₋₁)³,
"""
    if deriv == 0:
        return a[i] + b[i] * t + c[i] * t ** 2 + d[i] * t ** 3
    if deriv == 1:
        return b[i] + 2 * c[i] * t + 3 * d[i] * t ** 2
    return 2 * c[i] + 6 * d[i] * t


def piece_value(xs, coefs, i, x, deriv=0):
    """То же, но на заданном отрезке i."""
    a, b, c, d = coefs
    t = x - xs[i - 1]
    if deriv == 0:
        return a[i] + b[i] * t + c[i] * t ** 2 + d[i] * t ** 3
    if deriv == 1:
        return b[i] + 2 * c[i] * t + 3 * d[i] * t ** 2
    return 2 * c[i] + 6 * d[i] * t


def print_coefficients(xs, coefs):
    """Таблица коэффициентов сплайна"""
    a, b, c, d = coefs
    n = len(xs) - 1
    print("  i | [x_{i-1}, x_i] |      a_i     |      b_i     |      c_i     |      d_i")
    print("----+----------------+--------------+--------------+--------------+-------------")
    for i in range(1, n + 1):
        seg = f"[{xs[i - 1]}, {xs[i]}]"
        print(f"{i:3d} | {seg:14s} | {a[i]:12.5f} | {b[i]:12.5f} |"
              f" {c[i]:12.5f} | {d[i]:12.5f}")
    print()


def check_smoothness(xs, coefs):
    """
    Проверка, что сплайн построен правильно:
      * в узлах S(x_i) = f_i ,
      * во внутренних узлах соседние куски совпадают по значению,
        первой и второй производной,
      * на концах S'' = 0.
    Возвращает максимальный "скачок".
    """
    n = len(xs) - 1
    worst = 0.0
    for i in range(1, n):                 # внутренние узлы x_1 .. x_{n-1}
        x = xs[i]
        for k in range(3):
            left = piece_value(xs, coefs, i, x, k)       # кусок слева
            right = piece_value(xs, coefs, i + 1, x, k)  # кусок справа
            worst = max(worst, abs(left - right))
    ends = max(abs(piece_value(xs, coefs, 1, xs[0], 2)),
               abs(piece_value(xs, coefs, n, xs[n], 2)))
    return worst, ends


def sgn(v):
    """Знак для красивой печати формулы: '+' или '-'."""
    return "-" if v < 0 else "+"


def plot_spline(filename, xs, fs, coefs, x_star, value):
    """График сплайна и узлов интерполяции (сохраняется в PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return
    # Точки для гладкой кривой: 400 точек на [x0, xn].
    px = [xs[0] + (xs[-1] - xs[0]) * k / 400 for k in range(401)]
    py = [spline_value(xs, coefs, x) for x in px]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(px, py, color="#2a78d6", linewidth=2, label="кубический сплайн S(x)")
    ax.plot(xs, fs, "o", color="#0b0b0b", markersize=6, label="узлы (x_i, f_i)", zorder=5)
    ax.plot([x_star], [value], "o", color="#eb6834", markersize=7, zorder=6,
            label=f"S({x_star}) = {value:.4f}")
    ax.set_xlabel("x", color="#52514e")
    ax.set_ylabel("y", color="#52514e")
    ax.set_title("Естественный кубический сплайн (вариант 26)", loc="left", color="#0b0b0b")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График сохранён: {filename}\n")



def check_textbook():
    """Пример 3.3: x = 0..4, f = 0, 1.8415, 2.9093, 3.1411, 3.2432 -> f(1.5) = 2.4969."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    fs = [0.0, 1.8415, 2.9093, 3.1411, 3.2432]
    coefs = build_spline(xs, fs, verbose=False)
    print("Проверка на примере 3.3 методички:")
    print_coefficients(xs, coefs)
    print(f"  S(1.5) = {spline_value(xs, coefs, 1.5):.4f}   (в методичке 2.4969)\n")


# ОСНОВНАЯ ПРОГРАММА
def main():
    # Данные варианта 26
    xs = [0.1, 0.5, 0.9, 1.3, 1.7]
    fs = [100.01, 4.2500, 2.0446, 2.2817, 3.2360]
    x_star = 0.8

    print("Кубический сплайн X* = 0.8")
    print("  i  :  " + "  ".join(f"{i:8d}" for i in range(len(xs))))
    print("  x_i:  " + "  ".join(f"{x:8.4f}" for x in xs))
    print("  f_i:  " + "  ".join(f"{v:8.4f}" for v in fs))
    print()

    coefs = build_spline(xs, fs)
    print("Коэффициенты сплайна:")
    print_coefficients(xs, coefs)

    # Проверк
    node_err = max(abs(spline_value(xs, coefs, xs[i]) - fs[i]) for i in range(len(xs)))
    jump, ends = check_smoothness(xs, coefs)
    print("Проверки:")
    print(f"  max |S(x_i) - f_i| в узлах                  = {node_err:.2e}")
    print(f"  max скачок S, S', S'' во внутренних узлах   = {jump:.2e}")
    print(f"  |S''| на концах (должна быть 0)             = {ends:.2e}\n")

    # Значение в X*
    a, b, c, d = coefs
    i = find_segment(xs, x_star)
    t = x_star - xs[i - 1]
    value = spline_value(xs, coefs, x_star)
    x0 = xs[i - 1]
    print(f"Точка X* = {x_star} лежит на отрезке [{x0}, {xs[i]}] (i = {i}):")
    print(f"  S(x) = {a[i]:.5f} {sgn(b[i])} {abs(b[i]):.5f}(x - {x0})"
          f" {sgn(c[i])} {abs(c[i]):.5f}(x - {x0})^2 {sgn(d[i])} {abs(d[i]):.5f}(x - {x0})^3")
    print(f"  S({x_star}) = {a[i]:.5f} {sgn(b[i])} {abs(b[i]):.5f}*{t:.1f}"
          f" {sgn(c[i])} {abs(c[i]):.5f}*{t:.1f}^2 {sgn(d[i])} {abs(d[i]):.5f}*{t:.1f}^3"
          f" = {value:.5f}\n")

    # Таблица - это значения функции 1/x^2 + x^2 (та же, что в 3.1),
    # поэтому можно сравнить с "настоящим" значением.
    print(f"Для сравнения: таблица построена по y = 1/x^2 + x^2,"
          f" y({x_star}) = {1 / x_star ** 2 + x_star ** 2:.5f}.")


    plot_spline(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "lab3_2_spline.png"), xs, fs, coefs, x_star, value)

    check_textbook()


if __name__ == "__main__":
    main()
