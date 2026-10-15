"""
Решение системы нелинейных уравнений методом простой итерации
и методом Ньютона.

(параметр a = 2):
    f1(x1, x2) = a*x1^2 - x2 + x2^2 - a     = 0,
    f2(x1, x2) = x1 - sqrt(x2 + a) + 1      = 0.
У системы два решения; нужно найти то, где x1 > 0 и x2 > 0.

Этапы решения:
    1. Начальное приближение - графически: на плоскости (x1, x2) строим
       кривые f1 = 0 и f2 = 0; решения - точки их пересечения.
         f1 = 0  <=>  2*x1^2 + (x2 - 0.5)^2 = 2.25   - эллипс,
         f2 = 0  <=>  x1 = sqrt(x2 + 2) - 1          - "ветка параболы".
       По графику выбираем область G, где лежит положительное решение,
       и начальное приближение x^(0) - её центр.

    2. Метод Ньютона (формулы (2.13)-(2.20)):
         x^(k+1) = x^(k) + dx^(k),  где dx^(k) - решение линейной системы
             J(x^(k)) * dx^(k) = -f(x^(k)),
         J - матрица Якоби (матрица частных производных).
       Для системы 2x2 решение линейной системы записываем через
       определители (формулы (2.20)):
             x1^(k+1) = x1^(k) - det A1 / det J,
             x2^(k+1) = x2^(k) - det A2 / det J.
       Стоп: max |x_i^(k+1) - x_i^(k)| <= eps   (2.18).

    3. Метод простой итерации (формулы (2.21)-(2.26)):
         систему переписываем в виде x = phi(x) и считаем
             x^(k+1) = phi(x^(k)).
       Условие сходимости (теорема 2.4): в области G
             max ||phi'(x)|| <= q < 1,
       где phi'(x) - матрица Якоби функции phi, а норма - максимум по
       строкам суммы модулей (2.27). Стоп:
             q / (1 - q) * max |x_i^(k+1) - x_i^(k)| <= eps   (2.26).

       Как выбрать phi. "Наивные" варианты - выразить x1 из f2, а x2 из f1:
             x1 = sqrt(x2 + 2) - 1,   x2 = 0.5 + sqrt(2.25 - 2*x1^2)
       - не годятся: |d phi2 / d x1| около решения ≈ 1.9 > 1, условие
       (2.25) не выполняется (программа это показывает).
       Поэтому используем приём из замечания методички (там он для одного
       уравнения: x = x - lambda*f(x)), обобщённый на систему:
             x = x - L * f(x)  =  phi(x),
       где L - постоянная матрица 2x2, обратная к матрице Якоби в начальной
       точке: L = J(x^(0))^(-1). Тогда phi'(x) = E - L*J(x), и возле x^(0)
       эта матрица близка к нулевой - условие сходимости выполняется.

    4. Анализ: как погрешность зависит от номера итерации.

Запуск:
    python3 lab2_2_system.py            - точность по умолчанию 0.001
    python3 lab2_2_system.py 1e-6       - точность задаётся аргументом
Графики сохраняются в PNG рядом с программой (нужен matplotlib).
"""

import math
import os
import sys

# Параметр a из таблицы вариантов (вариант 26 -> a = 2).
A_PARAM = 2.0

# Точность по умолчанию и защита от бесконечного цикла.
DEFAULT_EPS = 1e-3
MAX_ITER = 1000

# Область G, в которой по графику лежит положительное решение:
#     X1_MIN <= x1 <= X1_MAX,   X2_MIN <= x2 <= X2_MAX.
X1_MIN, X1_MAX = 0.6, 1.0
X2_MIN, X2_MAX = 1.2, 1.6


# Функции системы и их произведение
def f1(x1, x2):
    """f1 = a*x1^2 - x2 + x2^2 - a."""
    a = A_PARAM
    return a * x1 ** 2 - x2 + x2 ** 2 - a


def f2(x1, x2):
    """f2 = x1 - sqrt(x2 + a) + 1. Определена при x2 >= -a."""
    a = A_PARAM
    return x1 - math.sqrt(x2 + a) + 1


def jacobian(x1, x2):
    """
    Матрица Якоби J = | df1/dx1  df1/dx2 |
                      | df2/dx1  df2/dx2 |.
    Производные считаем вручную:
        df1/dx1 = 2*a*x1,        df1/dx2 = 2*x2 - 1,
        df2/dx1 = 1,             df2/dx2 = -1 / (2*sqrt(x2 + a)).
    Возвращаем матрицу как список строк [[..., ...], [..., ...]].
    """
    a = A_PARAM
    return [
        [2 * a * x1, 2 * x2 - 1],
        [1.0, -1 / (2 * math.sqrt(x2 + a))],
    ]


def det2(M):
    """Определитель матрицы 2x2: |a b; c d| = a*d - b*c."""
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def inverse2(M):
    """
    Обратная матрица 2x2:
        | a b |^(-1)       1     |  d  -b |
        | c d |      = -------- * | -c   a |.
                       a*d - b*c
    """
    d = det2(M)
    if d == 0:
        raise ZeroDivisionError("Матрица вырождена, обратной нет")
    return [
        [M[1][1] / d, -M[0][1] / d],
        [-M[1][0] / d, M[0][0] / d],
    ]


def row_norm(M):
    """Норма матрицы ||M|| = max по строкам суммы модулей"""
    result = 0.0
    for row in M:
        s = 0.0
        for v in row:
            s += abs(v)
        result = max(result, s)
    return result


# метод ньютона
def newton(x0, eps):
    """
    Метод Ньютона для системы 2x2:
        x1^(k+1) = x1^(k) - det A1 / det J,
        x2^(k+1) = x2^(k) - det A2 / det J,
    где
        J  = | df1/dx1  df1/dx2 |,
             | df2/dx1  df2/dx2 |
        A1 = | f1  df1/dx2 |     (в J первый столбец заменён на f),
             | f2  df2/dx2 |
        A2 = | df1/dx1  f1 |     (в J второй столбец заменён на f).
             | df2/dx1  f2 |

    """
    x1, x2 = x0
    history = []
    for k in range(MAX_ITER):
        # значения функций и матрица Якоби в текущей точке
        F1 = f1(x1, x2)
        F2 = f2(x1, x2)
        J = jacobian(x1, x2)

        # определители
        det_J = det2(J)
        if det_J == 0:
            raise ZeroDivisionError("det J = 0: метод Ньютона неприменим")
        # A1: первый столбец J заменяем на столбец (f1, f2).
        det_A1 = det2([[F1, J[0][1]],
                       [F2, J[1][1]]])
        # A2: второй столбец J заменяем на столбец (f1, f2).
        det_A2 = det2([[J[0][0], F1],
                       [J[1][0], F2]])

        history.append((k, x1, x2, F1, F2, J, det_A1, det_A2, det_J))

        # новое приближение
        x1_new = x1 - det_A1 / det_J
        x2_new = x2 - det_A2 / det_J

        # критерий окончания: максимум модуля приращения
        step = max(abs(x1_new - x1), abs(x2_new - x2))
        x1, x2 = x1_new, x2_new
        if step <= eps:
            return [x1, x2], k + 1, history
    raise RuntimeError(f"Метод Ньютона не сошёлся за {MAX_ITER} итераций")


# Метод простой итерации
def make_phi(x0):
    """
    Строит функцию phi(x) = x - L * f(x), где L = J(x^(0))^(-1).


    Возвращает пару функций (phi, dphi):
        phi(x1, x2)  -> [phi1, phi2],
        dphi(x1, x2) -> матрица Якоби функции phi:  E - L * J(x).
    """
    L = inverse2(jacobian(x0[0], x0[1]))

    def phi(x1, x2):
        F1 = f1(x1, x2)
        F2 = f2(x1, x2)
        # (L * f)_i = L[i][0]*f1 + L[i][1]*f2
        p1 = x1 - (L[0][0] * F1 + L[0][1] * F2)
        p2 = x2 - (L[1][0] * F1 + L[1][1] * F2)
        return [p1, p2]

    def dphi(x1, x2):
        J = jacobian(x1, x2)
        M = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                # (L * J)_ij = L[i][0]*J[0][j] + L[i][1]*J[1][j]
                LJ = L[i][0] * J[0][j] + L[i][1] * J[1][j]
                # E - L*J: на диагонали единица минус, вне диагонали минус.
                M[i][j] = (1.0 if i == j else 0.0) - LJ
        return M

    return phi, dphi, L


def naive_q():
    """
    q = max ‖φ′(x)‖ ≤ q < 1
    """
    a = A_PARAM
    q = 0.0
    n = 40
    for i in range(n + 1):
        for j in range(n + 1):
            x1 = X1_MIN + (X1_MAX - X1_MIN) * i / n
            x2 = X2_MIN + (X2_MAX - X2_MIN) * j / n
            under_root = a + 0.25 - a * x1 ** 2
            if under_root <= 0:
                return float("inf")
            M = [[0.0, 1 / (2 * math.sqrt(x2 + a))],
                 [-a * x1 / math.sqrt(under_root), 0.0]]
            q = max(q, row_norm(M))
    return q


def check_iteration_conditions(phi, dphi, n=40):
    """
        1) q = max ||phi'(x)|| < 1
        2) phi(x) не выходит из G
    Возвращает q.
    """
    q = 0.0
    inside = True
    for i in range(n + 1):
        for j in range(n + 1):
            x1 = X1_MIN + (X1_MAX - X1_MIN) * i / n
            x2 = X2_MIN + (X2_MAX - X2_MIN) * j / n
            q = max(q, row_norm(dphi(x1, x2)))
            p1, p2 = phi(x1, x2)
            if not (X1_MIN <= p1 <= X1_MAX and X2_MIN <= p2 <= X2_MAX):
                inside = False
    print("Проверка условий теоремы 2.4 в области G:")
    print(f"  q = max ||phi'(x)|| = {q:.4f} < 1 : {q < 1}")
    print(f"  phi(x) из G снова попадает в G  : {inside}\n")
    return q


def simple_iteration(phi, x0, q, eps):
    """
    Метод простой итерации: x^(k+1) = phi(x^(k)).
    Стоп: q / (1 - q) * max |x_i^(k+1) - x_i^(k)| <= eps.
    Дополнительно следим, чтобы приближения не покинули G

    """
    coef = q / (1 - q)
    x1, x2 = x0
    history = []
    for k in range(MAX_ITER):
        p1, p2 = phi(x1, x2)
        err = coef * max(abs(p1 - x1), abs(p2 - x2))
        history.append((k, x1, x2, p1, p2, err))
        if not (X1_MIN <= p1 <= X1_MAX and X2_MIN <= p2 <= X2_MAX):
            print("приближение вышло из области G")
        x1, x2 = p1, p2
        if err <= eps:
            return [x1, x2], k + 1, history
    raise RuntimeError(f"Метод простой итерации не сошёлся за {MAX_ITER} итераций")



BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axes(fig, ax):
    """Светлый фон, тонкая сетка, без рамки сверху/справа."""
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_curves(filename, solution):
    """
    Графическое определение начального приближения: кривые f1 = 0 и f2 = 0
    на плоскости (x1, x2). Точки пересечения - решения системы.
    Пунктиром показана область G.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return

    a = A_PARAM
    # f1 = 0:  a*x1^2 + (x2 - 0.5)^2 = a + 0.25  - эллипс.
    # Параметризуем: x1 = sqrt((a+0.25)/a) * cos t,  x2 = 0.5 + sqrt(a+0.25) * sin t.
    n = 400
    e_x1, e_x2 = [], []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        e_x1.append(math.sqrt((a + 0.25) / a) * math.cos(t))
        e_x2.append(0.5 + math.sqrt(a + 0.25) * math.sin(t))

    # f2 = 0:  x1 = sqrt(x2 + a) - 1, x2 от -a до 2.5.
    c_x1, c_x2 = [], []
    for i in range(n + 1):
        x2 = -a + (2.5 + a) * i / n
        c_x2.append(x2)
        c_x1.append(math.sqrt(x2 + a) - 1)

    fig, ax = plt.subplots(figsize=(7, 7.5))
    _style_axes(fig, ax)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.plot(e_x1, e_x2, color=BLUE, linewidth=2, solid_capstyle="round", label="f1 = 0:  2x1² − x2 + x2² − 2 = 0")
    ax.plot(c_x1, c_x2, color=ORANGE, linewidth=2, label="f2 = 0:  x1 − √(x2 + 2) + 1 = 0")
    # Область G - пунктирный прямоугольник.
    ax.add_patch(Rectangle((X1_MIN, X2_MIN), X1_MAX - X1_MIN, X2_MAX - X2_MIN,
                           fill=False, linestyle="--", edgecolor=INK2, linewidth=1))
    ax.annotate("G", (X1_MAX, X2_MAX), textcoords="offset points",
                xytext=(4, 2), color=INK2)
    # Найденные решения.
    ax.plot([solution[0]], [solution[1]], "o", color=INK, markersize=7, zorder=5)
    # Подпись справа-снизу от точки: там нет линий графика.
    ax.annotate(f"({solution[0]:.4f}; {solution[1]:.4f})", solution,
                textcoords="offset points", xytext=(28, -22), color=INK)
    ax.plot([0], [-1], "o", color=INK2, markersize=6, zorder=5)
    ax.annotate("(0; −1)", (0, -1), textcoords="offset points",
                xytext=(8, -14), color=INK2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # Одинаковый масштаб по осям, чтобы эллипс не искажался.
    ax.set_xlim(-1.4, 2.1)
    ax.set_ylim(-2.2, 2.7)
    ax.set_aspect("equal")
    ax.set_title("Кривые f1 = 0 и f2 = 0", color=INK, loc="left")
    # Легенда под графиком, чтобы не закрывать кривые.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График кривых сохранён: {filename}\n")


def plot_errors(filename, newton_errors, iter_errors):
    """График погрешности от номера итерации (ось y логарифмическая)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(fig, ax)
    ax.semilogy(range(len(newton_errors)), newton_errors, "-o", color=BLUE,
                linewidth=2, markersize=6, label="Метод Ньютона")
    ax.semilogy(range(len(iter_errors)), iter_errors, "-o", color=ORANGE,
                linewidth=2, markersize=6, label="Метод простой итерации")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel("погрешность max|x_i^(k) − x_i*|")
    ax.set_title("Зависимость погрешности от числа итераций", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График погрешностей сохранён: {filename}\n")


def main():
    # Точность - входные данные программы
    if len(sys.argv) > 1:
        eps = float(sys.argv[1])
    else:
        eps = DEFAULT_EPS
    here = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Задание 2.2. Система нелинейных уравнений (вариант 26, a = 2)")
    print("    f1 = a*x1^2 - x2 + x2^2 - a = 0")
    print("    f2 = x1 - sqrt(x2 + a) + 1  = 0")
    print("=" * 70)
    print(f"Точность eps = {eps}\n")


    x0 = [(X1_MIN + X1_MAX) / 2, (X2_MIN + X2_MAX) / 2]
    print("1. По графику положительное решение лежит в области G:")
    print(f"   {X1_MIN} <= x1 <= {X1_MAX},  {X2_MIN} <= x2 <= {X2_MAX}")
    print(f"   Начальное приближение - центр G: x^(0) = ({x0[0]}, {x0[1]})\n")

    print("-" * 70)
    print("2. Метод Ньютона")
    print("-" * 70)
    xn, kn, hist_n = newton(x0, eps)
    print("  k |   x1^(k)   |   x2^(k)   |   f1       |   f2       |"
          "  det A1    |  det A2    |  det J")
    print("----+------------+------------+------------+------------+"
          "------------+------------+----------")
    for k, x1, x2, F1, F2, J, dA1, dA2, dJ in hist_n:
        print(f"{k:3d} | {x1:10.6f} | {x2:10.6f} | {F1:10.2e} | {F2:10.2e} |"
              f" {dA1:10.2e} | {dA2:10.2e} | {dJ:8.4f}")
    print(f"{kn:3d} | {xn[0]:10.6f} | {xn[1]:10.6f} |")
    print(f"\nРешение (Ньютон): x1 ≈ {xn[0]:.6f}, x2 ≈ {xn[1]:.6f},"
          f"  итераций: {kn}")
    print(f"Невязки: f1 = {f1(*xn):.2e}, f2 = {f2(*xn):.2e}\n")

    print("-" * 70)
    print("3. Метод простой итерации")
    print("-" * 70)
    print("Наивный выбор phi (x1 из f2, x2 из f1):")
    print("    x1 = sqrt(x2 + 2) - 1,   x2 = 0.5 + sqrt(2.25 - 2*x1^2)")
    print(f"    q = max ||phi'(x)|| в G = {naive_q():.4f} >= 1"
          " -> условие сходимости НЕ выполняется.\n")

    phi, dphi, L = make_phi(x0)
    print("Поэтому используем phi(x) = x - L * f(x),  L = J(x^(0))^(-1):")
    print(f"    L = | {L[0][0]:9.5f}  {L[0][1]:9.5f} |")
    print(f"        | {L[1][0]:9.5f}  {L[1][1]:9.5f} |\n")
    q = check_iteration_conditions(phi, dphi)
    print(f"Критерий остановки: q/(1-q) * max|x_i^(k+1) - x_i^(k)| <= eps,"
          f"  q/(1-q) = {q / (1 - q):.4f}\n")

    xi, ki, hist_i = simple_iteration(phi, x0, q, eps)
    print("  k |   x1^(k)   |   x2^(k)   |  phi1      |  phi2      | оценка")
    print("----+------------+------------+------------+------------+----------")
    for k, x1, x2, p1, p2, err in hist_i:
        print(f"{k:3d} | {x1:10.6f} | {x2:10.6f} | {p1:10.6f} | {p2:10.6f} | {err:9.2e}")
    print(f"{ki:3d} | {xi[0]:10.6f} | {xi[1]:10.6f} |")
    print(f"\nРешение (простая итерация): x1 ≈ {xi[0]:.6f}, x2 ≈ {xi[1]:.6f},"
          f"  итераций: {ki}")
    print(f"Невязки: f1 = {f1(*xi):.2e}, f2 = {f2(*xi):.2e}\n")

    print("-" * 70)
    print("4. Зависимость погрешности от числа итераций")
    print("-" * 70)
    x_star, _, _ = newton(x0, 1e-15)
    print(f"Точное решение (для сравнения): x1* = {x_star[0]:.12f},"
          f" x2* = {x_star[1]:.12f}\n")

    _, _, full_n = newton(x0, 1e-13)
    _, _, full_i = simple_iteration(phi, x0, q, 1e-13)
    # Погрешность приближения - максимум по компонентам |x_i^(k) - x_i*|.
    err_n = [max(abs(r[1] - x_star[0]), abs(r[2] - x_star[1])) for r in full_n]
    err_i = [max(abs(r[1] - x_star[0]), abs(r[2] - x_star[1])) for r in full_i]

    print("  k | Ньютон: погрешность | e_k/e_(k-1)^2 || Итерации: погрешность | e_k/e_(k-1)")
    print("----+---------------------+---------------++-----------------------+------------")
    for k in range(max(len(err_n), len(err_i))):
        if k < len(err_n):
            sn = f"{err_n[k]:19.4e}"
            if k > 0 and err_n[k - 1] > 0 and err_n[k] > 1e-15:
                rn = f"{err_n[k] / err_n[k - 1] ** 2:13.4f}"
            else:
                rn = " " * 13
        else:
            sn, rn = " " * 19, " " * 13
        if k < len(err_i):
            si = f"{err_i[k]:21.4e}"
            ri = f"{err_i[k] / err_i[k - 1]:10.4f}" if k > 0 and err_i[k - 1] > 0 else ""
        else:
            si, ri = "", ""
        print(f"{k:3d} | {sn} | {rn} || {si} | {ri}")
    print()

    print("Число итераций в зависимости от точности:")
    print("    eps    | Ньютон | простая итерация")
    print("-----------+--------+-----------------")
    for p in range(1, 11):
        e = 10.0 ** (-p)
        _, k1, _ = newton(x0, e)
        _, k2, _ = simple_iteration(phi, x0, q, e)
        print(f"  1e-{p:<2d}    | {k1:6d} | {k2:6d}")
    print()

    rate = err_i[4] / err_i[3]
    print(f"Измеренный множитель e_k/e_(k-1) = {rate:.4f};"
          f"  ||phi'(x*)|| = {row_norm(dphi(x_star[0], x_star[1])):.4f};"
          f"  q по области G = {q:.4f}\n")

    # Графики
    plot_curves(os.path.join(here, "lab2_2_graph.png"), x_star)
    plot_errors(os.path.join(here, "lab2_2_errors.png"),
                [e for e in err_n if e > 0], [e for e in err_i if e > 0])


if __name__ == "__main__":
    main()
