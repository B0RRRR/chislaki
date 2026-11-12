"""
Задача Коши для ОДУ 2-го порядка:

        x^4 y'' + 2x^3 y' + y = 0,
        y(1) = 1,   y'(1) = 1,
        x ∈ [1, 2],   h = 0.1.

Точное решение:
        y(x) = (sin 1 + cos 1) cos(1/x) + (sin 1 - cos 1) sin(1/x).

Нужно: реализовать методы Эйлера, Рунге-Кутты 4-го порядка и Адамса
4-го порядка, решить задачу и оценить погрешность двумя способами:
по Рунге-Ромбергу и сравнением с точным решением.

        y' = z,
        z' = y'' = -(2x^3 z + y) / x^4 = -2z/x - y/x^4,
        y(1) = 1,  z(1) = 1.

Дальше работаем с вектором Y = (y, z)
"""

import math


def rhs(x, Y):
    """
    Правая часть системы Y' = F(x, Y), где Y = [y, z], z = y'.
    Возвращает [y', z'].
    """
    y, z = Y
    return [z, -2 * z / x - y / x ** 4]


S1, C1 = math.sin(1.0), math.cos(1.0)


def exact(x):
    """Точное решение y(x) = (sin1 + cos1) cos(1/x) + (sin1 - cos1) sin(1/x)."""
    return (S1 + C1) * math.cos(1 / x) + (S1 - C1) * math.sin(1 / x)



def axpy(a, c, b):
    """Возвращает вектор a + c * b (покомпонентно)."""
    return [ai + c * bi for ai, bi in zip(a, b)]


def lincomb(coefs, vectors):
    """Возвращает sum_j coefs[j] * vectors[j] (покомпонентно)."""
    return [sum(c * v[i] for c, v in zip(coefs, vectors))
            for i in range(len(vectors[0]))]


def grid_size(a, b, h):
    """Число шагов N = (b - a) / h; проверяем, что h укладывается целое число раз."""
    n = round((b - a) / h)
    if abs(a + n * h - b) > 1e-9:
        raise ValueError(f"Шаг h = {h} не делит отрезок [{a}, {b}] нацело")
    return n



def euler(f, a, b, Y0, h):
    """
    Явный метод Эйлера:

        Y_{k+1} = Y_k + h * F(x_k, Y_k).

    """
    n = grid_size(a, b, h)
    xs = [a + k * h for k in range(n + 1)]
    Ys = [list(Y0)]
    for k in range(n):
        Ys.append(axpy(Ys[k], h, f(xs[k], Ys[k])))
    return xs, Ys


def rk4_step(f, x, Y, h):
    """
        K1 = h F(x,Y)
        K2 = h F(x + h/2, Y + K1/2)
        K3 = h F(x + h/2, Y + K2/2)
        K4 = h F(x + h,   Y + K3)
        ΔY = (K1 + 2 K2 + 2 K3 + K4) / 6,     Y_{k+1} = Y_k + ΔY.
    усредняется с весами 1, 2, 2, 1.
    """
    K1 = [h * v for v in f(x, Y)]
    K2 = [h * v for v in f(x + h / 2, axpy(Y, 0.5, K1))]
    K3 = [h * v for v in f(x + h / 2, axpy(Y, 0.5, K2))]
    K4 = [h * v for v in f(x + h, axpy(Y, 1.0, K3))]
    dY = lincomb([1 / 6, 2 / 6, 2 / 6, 1 / 6], [K1, K2, K3, K4])
    return axpy(Y, 1.0, dY)


def runge_kutta4(f, a, b, Y0, h):
    """Метод Рунге-Кутты 4-го порядка на всём отрезке: N шагов rk4_step."""
    n = grid_size(a, b, h)
    xs = [a + k * h for k in range(n + 1)]
    Ys = [list(Y0)]
    for k in range(n):
        Ys.append(rk4_step(f, xs[k], Ys[k], h))
    return xs, Ys


# МНОГОШАГОВЫЙ МЕТОД АДАМСА
def adams4(f, a, b, Y0, h):
    """
    Явный метод Адамса 4-го порядка:

        Y_{k+1} = Y_k + h/24 * (55 F_k - 59 F_{k-1} + 37 F_{k-2} - 9 F_{k-3}),
        где F_j = F(x_j, Y_j).

    """
    n = grid_size(a, b, h)
    if n < 4:
        raise ValueError("Для метода Адамса нужно хотя бы 4 шага")
    xs = [a + k * h for k in range(n + 1)]

    # Разгон: первые три шага методом Рунге-Кутты.
    Ys = [list(Y0)]
    for k in range(3):
        Ys.append(rk4_step(f, xs[k], Ys[k], h))

    # Значения правой части в уже найденных узлах. Храним их, чтобы каждое
    # F_j считать ровно один раз.
    Fs = [f(xs[k], Ys[k]) for k in range(4)]

    for k in range(3, n):
        # Fs[k], Fs[k-1], Fs[k-2], Fs[k-3] - четыре последних значения.
        step = lincomb([55, -59, 37, -9], [Fs[k], Fs[k - 1], Fs[k - 2], Fs[k - 3]])
        Ys.append(axpy(Ys[k], h / 24, step))
        Fs.append(f(xs[k + 1], Ys[k + 1]))   # единственное новое вычисление F
    return xs, Ys


METHODS = [
    # (название, функция, порядок точности p)
    ("Эйлер", euler, 1),
    ("Рунге-Кутта 4", runge_kutta4, 4),
    ("Адамс 4", adams4, 4),
]


# ОЦЕНКА ПОГРЕШНОСТИ ПО РУНГЕ-РОМБЕРГУ
def runge_romberg(xs_h, ys_h, xs_2h, ys_2h, p):
    """
    Решаем задачу дважды: с шагом h и с шагом 2h.
    Каждый узел сетки 2h - это каждый второй узел сетки h, там есть оба
    решения. Если метод имеет порядок p, то в общем узле
        y_точн ≈ y_h  + C h^p,
        y_точн ≈ y_2h + C (2h)^p,
    откуда главный член погрешности решения с шагом h:
        R_h = (y_h - y_2h) / (2^p - 1).

    Возвращает список (x, R_h, y_h) по всем общим узлам.
    """
    out = []
    for j, x in enumerate(xs_2h):
        y_h = ys_h[2 * j]                # тот же x в мелкой сетке
        assert abs(xs_h[2 * j] - x) < 1e-9
        out.append((x, (y_h - ys_2h[j]) / (2 ** p - 1), y_h))
    return out


# ПЕЧАТЬ
def print_solutions(xs, sols):
    """Таблица y_k всех методов, точное решение и абсолютные погрешности."""
    names = [name for name, _, _ in METHODS]
    print(f"{'x':>5s}  {'точное':>11s}" + "".join(f"  {n:>14s}" for n in names)
          + "".join(f"  {'|ε| ' + n.split()[0]:>12s}" for n in names))
    for k, x in enumerate(xs):
        ye = exact(x)
        vals = "".join(f"  {sols[n][k]:14.8f}" for n in names)
        errs = "".join(f"  {abs(ye - sols[n][k]):12.2e}" for n in names)
        print(f"{x:5.2f}  {ye:11.8f}{vals}{errs}")
    print()


def print_runge_romberg(rr_all):
    """Сравнение оценки Рунге-Ромберга с истинной погрешностью в общих узлах."""
    names = [name for name, _, _ in METHODS]
    header = "".join(f"  {('R(' + n.split()[0] + ')'):>11s}  {'ε':>9s}" for n in names)
    print(f"{'x':>5s}{header}")
    rows = zip(*(rr_all[n] for n in names))
    for row in rows:
        x = row[0][0]
        cells = "".join(f"  {R:11.2e}  {exact(x) - y:9.2e}" for (_, R, y) in row)
        print(f"{x:5.2f}{cells}")
    print("  R - оценка погрешности по Рунге-Ромбергу (y_h - y_2h)/(2^p - 1),")
    print("  ε - истинная погрешность y_точн - y_h. Должны быть близки.\n")


# ГРАФИК
def plot_results(filename, xs, sols):
    """
    Два графика один под другим (общая ось x, отдельные оси y):
      сверху - численные решения и точное;
      снизу  - модуль погрешности |y_точн - y_k| в логарифмическом масштабе
               (у методов погрешности отличаются на порядки).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    # Цвет + маркер + тип линии: метод различим и без цвета.
    style = {
        "Эйлер": dict(color="#2a78d6", marker="o", linestyle="-"),
        "Рунге-Кутта 4": dict(color="#eb6834", marker="s", linestyle="--"),
        "Адамс 4": dict(color="#1baf7a", marker="^", linestyle=":"),
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

    # Точное решение - плотная кривая, численные - маркеры в узлах.
    px = [xs[0] + (xs[-1] - xs[0]) * i / 300 for i in range(301)]
    ax1.plot(px, [exact(x) for x in px], color=ink, linewidth=1.5, label="точное решение")
    for name, _, _ in METHODS:
        st = style[name]
        ax1.plot(xs, sols[name], color=st["color"], marker=st["marker"], markersize=6,
                 linestyle="none", label=name)
    ax1.set_ylabel("y", color=ink2)
    ax1.set_title("Задача Коши x⁴y'' + 2x³y' + y = 0, h = 0.1 (вариант 26)",
                  loc="left", color=ink)
    ax1.legend(frameon=False)

    # В узле x_0 погрешность нулевая - на логарифмической шкале её не показать.
    for name, _, _ in METHODS:
        st = style[name]
        errs = [abs(exact(x) - y) for x, y in zip(xs[1:], sols[name][1:])]
        ax2.semilogy(xs[1:], errs, linewidth=2, markersize=6, label=name, **st)
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
    # Данные варианта 26
    a, b = 1.0, 2.0
    Y0 = [1.0, 1.0]          # y(1) = 1, z(1) = y'(1) = 1

    print("Задание 4.1. Задача Коши (вариант 26)")
    print("  x^4 y'' + 2x^3 y' + y = 0,  y(1) = 1,  y'(1) = 1,  x ∈ [1, 2]")
    print("  Система: y' = z,  z' = -2z/x - y/x^4")
    print(f"  Шаг h = {h}\n")

    # Решаем всеми методами с шагом h и с шагом 2h (для Рунге-Ромберга).
    sols, rr_all = {}, {}
    xs = None
    for name, method, p in METHODS:
        xs, Ys = method(rhs, a, b, Y0, h)
        xs2, Ys2 = method(rhs, a, b, Y0, 2 * h)
        ys = [Y[0] for Y in Ys]            # нам нужна только первая компонента y
        ys2 = [Y[0] for Y in Ys2]
        sols[name] = ys
        rr_all[name] = runge_romberg(xs, ys, xs2, ys2, p)

    print(f"Решение с шагом h = {h}:")
    print_solutions(xs, sols)

    print(f"Оценка погрешности по Рунге-Ромбергу (шаги h = {h} и 2h = {2 * h:g}):")
    print_runge_romberg(rr_all)

    print("Итог: максимальная погрешность на отрезке")
    print(f"  {'метод':>14s}  {'p':>2s}  {'max|ε| (точн.)':>15s}  {'max|R| (Р-Р)':>13s}")
    for name, _, p in METHODS:
        true_max = max(abs(exact(x) - y) for x, y in zip(xs, sols[name]))
        rr_max = max(abs(R) for _, R, _ in rr_all[name])
        print(f"  {name:>14s}  {p:2d}  {true_max:15.2e}  {rr_max:13.2e}")
    print()
  
    print("  Для Адамса оценка Р-Р грубее: при шаге 2h = 0.2 узлы до x = 1.6 - это")
    print("  ещё разгон РК4, а сам метод при h = 0.1 ещё не вышел на асимптотику C*h^4")
    print("  (при уменьшении h вдвое ошибка падает в 8, 12, 14, затем ~16 раз).\n")

    plot_results("lab4_1_solution.png", xs, sols)
    check_textbook()


def check_textbook():
    """
    Примеры 4.1, 4.4, 4.7: y' = (y + x)^2, y(0) = 0, h = 0.1.
    Методичка: Эйлер         y(0.5) = 0.031513228   (табл. 4.1),
               Рунге-Кутта 4 y(0.3) = 0.009336039   (табл. 4.13, узлы разгона),
               Адамс         y(1.0) = 0.551159854   (табл. 4.13).
    Скалярное уравнение - это просто система из одного уравнения.
    """
    g = lambda x, Y: [(Y[0] + x) ** 2]
    _, ye = euler(g, 0.0, 0.5, [0.0], 0.1)
    _, yr = runge_kutta4(g, 0.0, 0.3, [0.0], 0.1)
    _, ya = adams4(g, 0.0, 1.0, [0.0], 0.1)
    print("Проверка на примерах методички (y' = (y + x)^2, y(0) = 0, h = 0.1):")
    print(f"  Эйлер    y(0.5) = {ye[-1][0]:.9f}   (в методичке 0.031513228)")
    print(f"  РК4      y(0.3) = {yr[-1][0]:.9f}   (в методичке 0.009336039)")
    print(f"  Адамс    y(1.0) = {ya[-1][0]:.9f}   (в методичке 0.551159854)\n")


if __name__ == "__main__":
    main()
