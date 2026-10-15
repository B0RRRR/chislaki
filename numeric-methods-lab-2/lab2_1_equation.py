"""
Лабораторная работа 2, задание 2.1. Вариант 26.
Решение нелинейного уравнения методом простой итерации и методом Ньютона.

Уравнение варианта 26:
    f(x) = lg(x + 1) - x + 0.5 = 0,
нужно найти его ПОЛОЖИТЕЛЬНЫЙ корень (lg - десятичный логарифм).

Этапы решения (методичка, п. 2.1):
    1. Отделение корня: находим отрезок [a, b], на котором ровно один корень.
       Делаем это графически: строим графики y = lg(x + 1) и y = x - 0.5,
       корень - абсцисса точки их пересечения. Дополнительно печатаем
       таблицу значений f(x) и ищем, где f меняет знак.
    2. Уточнение корня с точностью eps двумя методами:

       * Метод Ньютона (касательных), формула (2.2):
             x^(k+1) = x^(k) - f(x^(k)) / f'(x^(k)).
         Начальное приближение берём так, чтобы f(x0) * f''(x0) > 0
         (теорема 2.2) - тогда сходимость гарантирована.
         Стоп: |x^(k+1) - x^(k)| < eps.

       * Метод простой итерации, формулы (2.5)-(2.7):
             переписываем уравнение в виде x = phi(x) и считаем
             x^(k+1) = phi(x^(k)).
         Условия сходимости (теорема 2.3) на отрезке [a, b]:
             1) phi(x) лежит в [a, b] для всех x из [a, b];
             2) |phi'(x)| <= q < 1.
         Стоп: q / (1 - q) * |x^(k+1) - x^(k)| <= eps.

    3. Анализ: как погрешность зависит от номера итерации.

Запуск:
Графики сохраняются в PNG рядом с программой (нужен matplotlib;
если его нет - программа всё посчитает, просто без картинок).
"""

import math
import os
import sys

# Точность по умолчанию (если не указана при запуске).
DEFAULT_EPS = 1e-3

# Максимальное число итераций - защита от бесконечного цикла,
MAX_ITER = 1000

# ln(10) - понадобится для производной десятичного логарифма:
LN10 = math.log(10)



# Функция, её производные и функция для метода простой итерации.
def f(x):
    """f(x) = lg(x + 1) - x + 0.5. math.log10 - десятичный логарифм."""
    return math.log10(x + 1) - x + 0.5


def df(x):
    """
    Первая производная: f'(x) = 1 / ((x + 1) ln 10) - 1.
    """
    return 1 / ((x + 1) * LN10) - 1


def d2f(x):
    """
    Вторая производная: f''(x) = -1 / ((x + 1)^2 ln 10).
    Нужна только для выбора начального приближения Ньютона
    (условие f(x0) * f''(x0) > 0). Она всегда < 0.
    """
    return -1 / ((x + 1) ** 2 * LN10)


def phi(x):
    """
    Функция для метода простой итерации.
    Из уравнения lg(x + 1) - x + 0.5 = 0 выражаем x:
        x = lg(x + 1) + 0.5  =  phi(x).
    """
    return math.log10(x + 1) + 0.5


def dphi(x):
    """
    Производная phi'(x) = 1 / ((x + 1) ln 10).
    По ней проверяем условие сходимости |phi'(x)| <= q < 1.
    """
    return 1 / ((x + 1) * LN10)


# Отделение корня
def tabulate(a, b, h):
    """
    Печатает таблицу значений f(x) на [a, b] с шагом h и находит
    отрезки, на концах которых f меняет знак: на таком отрезке
    непрерывная функция обязательно проходит через ноль.
    Возвращает список таких отрезков [(x_left, x_right), ...].
    """
    print("   x     |    f(x)")
    intervals = []
    n = int(round((b - a) / h))           # число шагов
    x_prev = a
    f_prev = f(a)
    print(f" {a:7.3f} | {f_prev:9.5f}")
    for i in range(1, n + 1):
        x = a + i * h
        fx = f(x)
        mark = ""
        # Разные знаки на концах шага: f(x_prev) * f(x) < 0.
        if f_prev * fx < 0:
            intervals.append((x_prev, x))
            mark = "   <- смена знака, здесь корень"
        print(f" {x:7.3f} | {fx:9.5f}{mark}")
        x_prev, f_prev = x, fx
    print()
    return intervals


def plot_localization(filename, root=None):
    """
    Графическое отделение корня:
    уравнение lg(x + 1) - x + 0.5 = 0 переписываем как
        lg(x + 1) = x - 0.5
    и строим два графика: y = lg(x + 1) и y = x - 0.5.
    Абсциссы точек пересечения - корни уравнения.
    Картинка сохраняется в файл.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")            # рисуем в файл, без окна
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return

    # Точки для графика: x от -0.95 до 2 (при x <= -1 логарифм не определён).
    xs = []
    n = 400
    for i in range(n + 1):
        xs.append(-0.95 + (2.0 + 0.95) * i / n)
    y1 = [math.log10(x + 1) for x in xs]
    y2 = [x - 0.5 for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(fig, ax)
    ax.plot(xs, y1, color=BLUE, linewidth=2, label="y = lg(x + 1)")
    ax.plot(xs, y2, color=ORANGE, linewidth=2, label="y = x − 0.5")
    if root is not None:
        ax.plot([root], [root - 0.5], "o", color=INK, markersize=7, zorder=5)
        ax.annotate(f"x* ≈ {root:.4f}", (root, root - 0.5),
                    textcoords="offset points", xytext=(10, -18), color=INK)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Отделение корня: lg(x + 1) = x − 0.5", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График отделения корня сохранён: {filename}\n")


# Проверка условий сходимости
def check_newton_conditions(a, b, n=200):
    """
    если на [a, b]
        * f(a) * f(b) < 0,
        * f' и f'' не меняют знак,
    то метод Ньютона сходится из той точки x0, где f(x0) * f''(x0) > 0.
    """
    ok_ends = f(a) * f(b) < 0

    # Знаки f' и f'' во всех точках должны совпадать со знаком в точке a.
    sign_df = df(a) > 0
    sign_d2f = d2f(a) > 0
    const_df = True
    const_d2f = True
    for i in range(n + 1):
        x = a + (b - a) * i / n
        if (df(x) > 0) != sign_df:
            const_df = False
        if (d2f(x) > 0) != sign_d2f:
            const_d2f = False

    print("Проверка условий теоремы 2.2 (метод Ньютона):")
    print(f"  f(a) * f(b) = {f(a):.4f} * {f(b):.4f} < 0 : {ok_ends}")
    print(f"  f'(x) сохраняет знак на [a, b]  : {const_df}"
          f"  (f' {'> 0' if sign_df else '< 0'})")
    print(f"  f''(x) сохраняет знак на [a, b] : {const_d2f}"
          f"  (f'' {'> 0' if sign_d2f else '< 0'})")

    # Выбираем x0 среди концов отрезка: нужно f(x0) * f''(x0) > 0.
    x0 = None
    for candidate in (a, b):
        if f(candidate) * d2f(candidate) > 0:
            x0 = candidate
            break
    if x0 is not None:
        print(f"  f(x0) * f''(x0) > 0 выполняется при x0 = {x0}"
              f"  ({f(x0):.4f} * {d2f(x0):.4f} > 0)\n")
    else:
        print("  подходящий конец отрезка не найден\n")
    return x0

# Проверка условий сходимости метода простой итерации
def check_iteration_conditions(a, b, n=200):
    """
    на [a, b] должно быть
        1) phi(x) лежит в [a, b] для всех x из [a, b];
        2) |phi'(x)| <= q < 1.
    q - это максимум |phi'(x)| на отрезке; ищем его перебором n+1 точек.
    Возвращает q (или None, если условия не выполнены).
    """
    q = 0.0
    inside = True
    phi_min = phi(a)
    phi_max = phi(a)
    for i in range(n + 1):
        x = a + (b - a) * i / n
        q = max(q, abs(dphi(x)))
        p = phi(x)
        phi_min = min(phi_min, p)
        phi_max = max(phi_max, p)
        if p < a or p > b:
            inside = False

    print("Проверка условий теоремы 2.3 (метод простой итерации):")
    print(f"  phi(x) = lg(x + 1) + 0.5 принимает значения "
          f"[{phi_min:.4f}, {phi_max:.4f}] - внутри [{a}, {b}]: {inside}")
    print(f"  q = max |phi'(x)| = {q:.4f} < 1 : {q < 1}\n")
    if inside and q < 1:
        return q
    return None


# Метод Ньютона
def newton(x0, eps):
    """
    Метод Ньютона: x^(k+1) = x^(k) - f(x^(k)) / f'(x^(k)).

    в точке x^(k) проводим касательную к графику f и берём точку,
    где касательная пересекает ось x, - это и есть x^(k+1).

    f(x⁰)·f″(x⁰) > 0.
    """
    x = x0
    history = []
    for k in range(MAX_ITER):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("f'(x) = 0: касательная горизонтальна")

        # Шаг Ньютона: -f / f'.
        step = -fx / dfx
        history.append((k, x, fx, dfx, step))

        x_new = x + step
        # Условие окончания: |x^(k+1) - x^(k)| < eps.
        if abs(x_new - x) < eps:
            return x_new, k + 1, history
        x = x_new
    raise RuntimeError(f"Метод Ньютона не сошёлся за {MAX_ITER} итераций")


# Метод простых итерации
def simple_iteration(x0, q, eps):
    """
    Метод простой итерации: x^(k+1) = phi(x^(k)).

    Оценка погрешности:
        |x* - x^(k+1)| <= q / (1 - q) * |x^(k+1) - x^(k)|.
    Останавливаемся, когда правая часть стала <= eps.

    Возвращает (x, k, history), history - строки (k, x^(k), phi(x^(k)), оценка).
    """
    coef = q / (1 - q)       # множитель в оценке, считаем один раз
    x = x0
    history = []
    for k in range(MAX_ITER):
        x_new = phi(x)
        err = coef * abs(x_new - x)
        history.append((k, x, x_new, err))
        if err <= eps:
            return x_new, k + 1, history
        x = x_new
    raise RuntimeError(f"Метод простой итерации не сошёлся за {MAX_ITER} итераций")



BLUE = "#2a78d6"       # 1-я линия
ORANGE = "#eb6834"     # 2-я линия
INK = "#0b0b0b"        # основной цвет текста
INK2 = "#52514e"       # подписи осей
GRID = "#e4e3df"       # сетка - светлая, чтобы не спорила с линиями
SURFACE = "#fcfcfb"    # фон


def _style_axes(fig, ax, zero_line=True):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    if zero_line:
        ax.axhline(0, color=INK2, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_errors(filename, newton_errors, iter_errors):
    """
    График |x^(k) - x*| от номера итерации k для обоих методов.
    Ось y логарифмическая: так линейное убывание ошибки
    (простая итерация) выглядит прямой, а квадратичное (Ньютон) -
    кривой, круто уходящей вниз.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(fig, ax, zero_line=False)
    ax.semilogy(range(len(newton_errors)), newton_errors, "-o", color=BLUE,
                linewidth=2, markersize=6, label="Метод Ньютона")
    ax.semilogy(range(len(iter_errors)), iter_errors, "-o", color=ORANGE,
                linewidth=2, markersize=6, label="Метод простой итерации")
    # Номер итерации - целое число, поэтому дробные метки на оси x не нужны.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel("погрешность |x^(k) − x*|")
    ax.set_title("Зависимость погрешности от числа итераций", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График погрешностей сохранён: {filename}\n")


def main():
    if len(sys.argv) > 1:
        eps = float(sys.argv[1])
    else:
        eps = DEFAULT_EPS

    here = os.path.dirname(os.path.abspath(__file__))

    print("=" * 64)
    print("Задание 2.1. Нелинейное уравнение (вариант 26)")
    print("    f(x) = lg(x + 1) - x + 0.5 = 0")
    print("=" * 64)
    print(f"Точность eps = {eps}\n")

    # 1. Отделение корня
    # Таблица значений на [0, 2] с шагом 0.1: ищем смену знака f.
    print("1. Отделение корня. Таблица значений f(x):")
    intervals = tabulate(0.0, 2.0, 0.1)
    print(f"Смена знака на отрезках: "
          + ", ".join(f"[{l:.1f}, {r:.1f}]" for l, r in intervals) + "\n")

    # По графику (см. lab2_1_graph.png) прямая y = x - 0.5 пересекает
    # кривую y = lg(x + 1) при x между 0.5 и 1. Берём этот отрезок:
    # на нём выполняются условия обоих методов (проверяем ниже).
    a, b = 0.5, 1.0
    print(f"Отрезок, содержащий положительный корень: [a, b] = [{a}, {b}]\n")

    # 2. Метод Ньютона
    print("-" * 64)
    print("2. Метод Ньютона")
    print("-" * 64)
    x0_newton = check_newton_conditions(a, b)
    x_n, k_n, hist_n = newton(x0_newton, eps)

    print("  k |    x^(k)    |   f(x^(k))   |  f'(x^(k))  | -f/f'")
    print("----+-------------+--------------+-------------+-------------")
    for k, x, fx, dfx, step in hist_n:
        print(f"{k:3d} | {x:11.7f} | {fx:12.4e} | {dfx:11.7f} | {step:12.4e}")
    print(f"{k_n:3d} | {x_n:11.7f} |")
    print(f"\nКорень (Ньютон): x* ≈ {x_n:.7f},  итераций: {k_n},"
          f"  невязка f(x*) = {f(x_n):.2e}\n")

    # 3. Метод простой итерации
    print("-" * 64)
    print("3. Метод простой итерации,  x = phi(x) = lg(x + 1) + 0.5")
    print("-" * 64)
    q = check_iteration_conditions(a, b)
    x0_iter = (a + b) / 2          # начальное приближение - середина отрезка
    print(f"Начальное приближение x0 = (a + b) / 2 = {x0_iter}")
    print(f"Критерий остановки: q/(1-q) * |x^(k+1) - x^(k)| <= eps,"
          f"  q/(1-q) = {q / (1 - q):.4f}\n")
    x_i, k_i, hist_i = simple_iteration(x0_iter, q, eps)

    print("  k |    x^(k)    | phi(x^(k))  | оценка погрешности")
    print("----+-------------+-------------+-------------------")
    for k, x, px, err in hist_i:
        print(f"{k:3d} | {x:11.7f} | {px:11.7f} | {err:12.4e}")
    print(f"{k_i:3d} | {x_i:11.7f} |")
    print(f"\nКорень (простая итерация): x* ≈ {x_i:.7f},  итераций: {k_i},"
          f"  невязка f(x*) = {f(x_i):.2e}\n")

    # 4. Анализ погрешности
    print("-" * 64)
    print("4. Зависимость погрешности от числа итераций")
    print("-" * 64)
    # "Точный" корень - Ньютоном с очень высокой точностью.
    x_star, _, _ = newton(x0_newton, 1e-15)
    print(f"Точное значение (для сравнения): x* = {x_star:.12f}\n")

    # Прогоняем оба метода с маленьким eps, чтобы увидеть, как убывает
    # реальная погрешность |x^(k) - x*| по итерациям.
    _, _, full_n = newton(x0_newton, 1e-13)
    _, _, full_i = simple_iteration(x0_iter, q, 1e-13)
    err_n = [abs(row[1] - x_star) for row in full_n]   # row[1] = x^(k)
    err_i = [abs(row[1] - x_star) for row in full_i]

    print("  k | Ньютон |x^(k)-x*| | e_k/e_(k-1)^2 || Итерации |x^(k)-x*| | e_k/e_(k-1)")
    print("----+------------------+---------------++--------------------+------------")
    for k in range(max(len(err_n), len(err_i))):
        # Для Ньютона показываем e_k / e_(k-1)^2 (постоянство -> квадратичная
        # сходимость), для простой итерации e_k / e_(k-1) (постоянство ->
        # линейная сходимость, отношение ≈ |phi'(x*)|).
        if k < len(err_n):
            sn = f"{err_n[k]:16.4e}"
            if k > 0 and err_n[k - 1] > 0 and err_n[k] > 1e-15:
                rn = f"{err_n[k] / err_n[k - 1] ** 2:13.4f}"
            else:
                rn = " " * 13
        else:
            sn, rn = " " * 16, " " * 13
        if k < len(err_i):
            si = f"{err_i[k]:18.4e}"
            ri = f"{err_i[k] / err_i[k - 1]:10.4f}" if k > 0 and err_i[k - 1] > 0 else ""
        else:
            si, ri = "", ""
        print(f"{k:3d} | {sn} | {rn} || {si} | {ri}")
    print(f"\n|phi'(x*)| = {dphi(x_star):.4f}  - во столько раз за итерацию "
          "уменьшается ошибка простой итерации")
    print(f"|f''(x*) / (2 f'(x*))| = {abs(d2f(x_star) / (2 * df(x_star))):.4f}"
          "  - константа квадратичной сходимости Ньютона\n")

    # Число итераций для разных точностей.
    print("Число итераций в зависимости от точности:")
    print("    eps    | Ньютон | простая итерация")
    print("-----------+--------+-----------------")
    for p in range(1, 11):
        e = 10.0 ** (-p)
        _, kn, _ = newton(x0_newton, e)
        _, ki, _ = simple_iteration(x0_iter, q, e)
        print(f"  1e-{p:<2d}    | {kn:6d} | {ki:6d}")
    print()

    # Графики
    plot_localization(os.path.join(here, "lab2_1_graph.png"), root=x_star)
    plot_errors(os.path.join(here, "lab2_1_errors.png"),
                [e for e in err_n if e > 0], [e for e in err_i if e > 0])


if __name__ == "__main__":
    main()
