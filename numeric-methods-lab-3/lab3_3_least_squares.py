"""

Чем МНК отличается от интерполяции (3.1, 3.2):
    интерполяционный многочлен обязан ПРОЙТИ через все точки.
    В МНК многочлен невысокой степени F(x) = a0 + a1 x + ... + an x^n
    проходит "как можно ближе" к точкам: коэффициенты подбираются так,
    чтобы сумма квадратов отклонений была минимальной (3.15):
        Phi = sum_j (F(x_j) - y_j)^2  ->  min.

Как найти коэффициенты (формулы (3.16)-(3.17)):
    в точке минимума все частные производные dPhi/da_k равны нулю.
    Это даёт систему линейных уравнений - НОРМАЛЬНУЮ СИСТЕМУ МНК:
        sum_{i=0..n} a_i * (sum_j x_j^(k+i))  =  sum_j y_j * x_j^k,   k = 0..n.
    Для n = 1 (прямая a0 + a1 x):
        a0 * (N+1)   + a1 * sum x_j   = sum y_j
        a0 * sum x_j + a1 * sum x_j^2 = sum y_j x_j
    Для n = 2 (парабола) - аналогично, матрица 3x3 из сумм x^0 .. x^4.
    Систему решаем методом Гаусса с выбором главного элемента (лаба 1.1).

Запуск:  python3 lab3_3_least_squares.py
Графики сохраняются в PNG рядом с программой (нужен matplotlib).
"""

import os


# РЕШЕНИЕ ЛИНЕЙНОЙ СИСТЕМЫ
def gauss_solve(M, rhs):
    """
    Решение системы M a = rhs методом Гаусса с выбором главного элемента
    по столбцу.
    """
    n = len(M)
    # Расширенная матрица [M | rhs] - копия, чтобы не портить исходные данные.
    A = [list(map(float, M[i])) + [float(rhs[i])] for i in range(n)]

    # Прямой ход: обнуляем элементы под диагональю.
    for k in range(n):
        # Главный элемент - максимальный по модулю в столбце k.
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i
        A[k], A[p] = A[p], A[k]
        for i in range(k + 1, n):
            mu = A[i][k] / A[k][k]
            for j in range(k, n + 1):
                A[i][j] -= mu * A[k][j]

    # Обратный ход: снизу вверх.
    a = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * a[j]
        a[i] = (A[i][n] - s) / A[i][i]
    return a


# МНК
def normal_system(xs, ys, degree):
    """
    Строит нормальную систему МНК для многочлена степени degree.
    Элемент матрицы в строке k, столбце i:  sum_j x_j^(k+i).
    Правая часть в строке k:                sum_j y_j * x_j^k.
    Σx⁰ = N+1,  Σx,  Σx²,  Σx³,  Σx⁴         Σy,  Σxy,  Σx²y
    | N+1   Σx  | |a₀|   | Σy  |
    |           | |  | = |     | 
    | Σx    Σx² | |a₁|   | Σxy |   
    """
    n = degree
    M = [[0.0] * (n + 1) for _ in range(n + 1)]
    rhs = [0.0] * (n + 1)
    for k in range(n + 1):
        for i in range(n + 1):
            s = 0.0
            for x in xs:
                s += x ** (k + i)
            M[k][i] = s
        s = 0.0
        for x, y in zip(xs, ys):
            s += y * x ** k
        rhs[k] = s
    return M, rhs


def poly_value(a, x):
    """Значение многочлена F(x) = a0 + a1 x + a2 x^2 + ..."""
    s = 0.0
    for i in range(len(a)):
        s += a[i] * x ** i
    return s


def sum_squared_errors(a, xs, ys):
    """Сумма квадратов ошибок Phi = sum_j (F(x_j) - y_j)^2   (3.15).
    Φ = Σⱼ (F(xⱼ) − yⱼ)²"""
    s = 0.0
    for x, y in zip(xs, ys):
        s += (poly_value(a, x) - y) ** 2
    return s


def poly_text(a):
    """Текстовая запись многочлена: a0 + a1 x + a2 x^2."""
    text = f"{a[0]:.4f}"
    for i in range(1, len(a)):
        sign = "-" if a[i] < 0 else "+"
        power = "x" if i == 1 else f"x^{i}"
        text += f" {sign} {abs(a[i]):.4f}{power}"
    return text


def least_squares(xs, ys, degree, verbose=True):
    """Находит многочлен МНК степени degree; печатает систему и таблицу."""
    M, rhs = normal_system(xs, ys, degree)
    a = gauss_solve(M, rhs)
    phi = sum_squared_errors(a, xs, ys)
    if verbose:
        names = [f"a{i}" for i in range(degree + 1)]
        print(f"Нормальная система МНК для многочлена {degree}-й степени:")
        for k in range(degree + 1):
            left = " + ".join(f"{M[k][i]:.4f}*{names[i]}" for i in range(degree + 1))
            print(f"   {left} = {rhs[k]:.4f}")
        print("Решение: " + ", ".join(f"{names[i]} = {a[i]:.4f}" for i in range(degree + 1)))
        print(f"F{degree}(x) = {poly_text(a)}\n")
        print("  j |   x_j  |     y_j    |  F(x_j)    | F(x_j) - y_j")
        print("----+--------+------------+------------+-------------")
        for j, (x, y) in enumerate(zip(xs, ys)):
            Fx = poly_value(a, x)
            print(f"{j:3d} | {x:6.2f} | {y:10.4f} | {Fx:10.4f} | {Fx - y:11.4f}")
        print(f"\nСумма квадратов ошибок Phi = {phi:.4f}\n")
    return a, phi


# ГРАФИК
def plot_fit(filename, xs, ys, a1, a2):
    """Точки таблицы и два приближающих многочлена на одном графике."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен.\n")
        return
    px = [xs[0] + (xs[-1] - xs[0]) * k / 300 for k in range(301)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(px, [poly_value(a1, x) for x in px], color="#2a78d6", linewidth=2,
            label=f"F1(x) = {poly_text(a1)}")
    ax.plot(px, [poly_value(a2, x) for x in px], color="#eb6834", linewidth=2,
            linestyle="--", label=f"F2(x) = {poly_text(a2)}")
    ax.plot(xs, ys, "o", color="#0b0b0b", markersize=6, zorder=5,
            label="табличные точки (x_j, y_j)")
    ax.set_xlabel("x", color="#52514e")
    ax.set_ylabel("y", color="#52514e")
    ax.set_title("МНК: приближающие многочлены 1-й и 2-й степени", loc="left",
                 color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График сохранён: {filename}\n")


# ПРОВЕРКА НА ПРИМЕРЕ
def check_textbook():
    """
    Пример 1 из п. 3.2: x = 0, 1.7, ..., 8.5;
    F1 = 0.4713 + 0.3177x, Phi = 0.4872;
    F2 = 0.1295 + 0.6193x - 0.0355x^2, Phi = 0.0946.
    """
    xs = [0.0, 1.7, 3.4, 5.1, 6.8, 8.5]
    ys = [0.0, 1.3038, 1.8439, 2.2583, 2.6077, 2.9155]
    a1, p1 = least_squares(xs, ys, 1, verbose=False)
    a2, p2 = least_squares(xs, ys, 2, verbose=False)
    print("Проверка на примере методички:")
    print(f"  F1(x) = {poly_text(a1)},  Phi = {p1:.4f}"
          "   (в методичке 0.4713 + 0.3177x, Phi = 0.4872)")
    print(f"  F2(x) = {poly_text(a2)},  Phi = {p2:.4f}"
          "   (в методичке 0.1295 + 0.6193x - 0.0355x^2, Phi = 0.0946)\n")


def main():
    xs = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    ys = [100.01, 4.250, 2.0446, 2.2817, 3.236, 4.6368]

    print("=" * 70)
    print("Задание 3.3. Метод наименьших квадратов (вариант 26)")
    print("=" * 70)
    print("  x_j: " + "  ".join(f"{x:8.4f}" for x in xs))
    print("  y_j: " + "  ".join(f"{y:8.4f}" for y in ys))
    print()

    print("-" * 70)
    print("а) Многочлен 1-й степени  F1(x) = a0 + a1 x")
    print("-" * 70)
    a1, phi1 = least_squares(xs, ys, 1)

    print("-" * 70)
    print("б) Многочлен 2-й степени  F2(x) = a0 + a1 x + a2 x^2")
    print("-" * 70)
    a2, phi2 = least_squares(xs, ys, 2)

    print("Вывод:")
    print(f"  Phi для 1-й степени = {phi1:.4f},  для 2-й степени = {phi2:.4f}.")

    plot_fit(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lab3_3_least_squares.png"), xs, ys, a1, a2)

    check_textbook()


if __name__ == "__main__":
    main()
