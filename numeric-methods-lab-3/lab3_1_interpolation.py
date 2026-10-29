"""
    y = 1/x^2 + x^2,
    а) X_i = 0.1, 0.5, 0.9, 1.3;
    б) X_i = 0.1, 0.5, 1.1, 1.3;
    X* = 0.8.

"""

import math



def f(x):
    """y = 1/x^2 + x^2."""
    return 1 / x ** 2 + x ** 2


def f4(x):
    """
    Четвёртая производная f, нужна для аоценки погрешности при n = 3.
    (x^2)'''' = 0,  (x^(-2))'''' = (-2)(-3)(-4)(-5) x^(-6) = 120 / x^6.
    """
    return 120 / x ** 6


# подсчет коэффициентов
def lagrange_coefficients(xs, ys):
    """
        L(x) = sum_i  [f_i / omega'(x_i)] * prod_{j != i} (x - x_j),
    где omega'(x_i) = prod_{j != i} (x_i - x_j).


             (x − x₁)(x − x₂)(x − x₃)
    l₀(x) = ─────────────────────────── 
        (x₀ − x₁)(x₀ − x₂)(x₀ − x₃)

    omega'(x_i) = (x_i - x_0)...(x_i - x_{i-1})(x_i - x_{i+1})...(x_i - x_n)
    """
    n = len(xs)
    coefs = []
    w_primes = []
    for i in range(n):
        # omega'(x_i) = (x_i - x_0)...(x_i - x_{i-1})(x_i - x_{i+1})...(x_i - x_n)
        w = 1.0
        for j in range(n):
            if j != i:
                w *= xs[i] - xs[j]
        w_primes.append(w)
        coefs.append(ys[i] / w)
    return coefs, w_primes


def lagrange_value(xs, coefs, x):
    """
    Значение многочлена Лагранжа в точке x:
        L(x) = sum_i c_i * prod_{j != i} (x - x_j).
        L(x) = f₀·l₀(x) + f₁·l₁(x) + f₂·l₂(x) + f₃·l₃(x)
    L(x) = c₀·(x − 0.5)(x − 0.9)(x − 1.3)
     + c₁·(x − 0.1)(x − 0.9)(x − 1.3)
     + c₂·(x − 0.1)(x − 0.5)(x − 1.3) 
     + c₃·(x − 0.1)(x − 0.5)(x − 0.9) 
    """
    n = len(xs)
    total = 0.0
    for i in range(n):
        term = coefs[i]
        for j in range(n):
            if j != i:
                term *= x - xs[j]
        total += term
    return total


def lagrange_formula(xs, coefs):
    """Текстовая запись многочлена Лагранжа (для печати)."""
    parts = []
    for i in range(len(xs)):
        factors = "".join(f"(x - {xs[j]})" for j in range(len(xs)) if j != i)
        sign = "-" if coefs[i] < 0 else "+"
        parts.append(f" {sign} {abs(coefs[i]):.5f}{factors}")
    text = "".join(parts).strip()
    return text[2:] if text.startswith("+ ") else text



def divided_differences(xs, ys):
    """
    Таблица разделённых разностей.
    table[k][i] - разность порядка k, построенная по узлам x_i .. x_{i+k}:
        table[0][i] = f_i,
        table[k][i] = (table[k-1][i] - table[k-1][i+1]) / (x_i - x_{i+k}).
    Для многочлена Ньютона нужны только верхние элементы каждого столбца:
        table[0][0] = f(x0), table[1][0] = f(x0,x1), table[2][0] = f(x0,x1,x2),
         
    f(xᵢ, xᵢ₊₁) = (fᵢ − fᵢ₊₁) / (xᵢ − xᵢ₊₁)
    f(xᵢ, xᵢ₊₁, xᵢ₊₂) = (f(xᵢ,xᵢ₊₁) − f(xᵢ₊₁,xᵢ₊₂)) / (xᵢ − xᵢ₊₂)

    """
    n = len(xs)
    table = [list(map(float, ys))]
    for k in range(1, n):
        prev = table[k - 1]
        col = []
        for i in range(n - k):
            col.append((prev[i] - prev[i + 1]) / (xs[i] - xs[i + k]))
        table.append(col)
    return table


def newton_value(xs, table, x):
    """
    Значение многочлена Ньютона в точке x:
        P(x) = d0 + (x - x0) d1 + (x - x0)(x - x1) d2 + ...,
    где d_k = table[k][0] - верхние разделённые разности.
    Произведение (x - x0)(x - x1)... накапливаем в переменной prod.
    """
    total = 0.0
    prod = 1.0
    for k in range(len(xs)):
        total += table[k][0] * prod
        prod *= x - xs[k]
    return total


def newton_formula(xs, table):
    """Текстовая запись многочлена Ньютона (для печати)."""
    text = f"{table[0][0]:.5f}"
    factors = ""
    for k in range(1, len(xs)):
        factors += f"(x - {xs[k - 1]})"
        d = table[k][0]
        text += f" {'-' if d < 0 else '+'} {abs(d):.5f}{factors}"
    return text


def print_divided_differences(xs, table):
    """Печать таблицы разделённых разностей."""
    n = len(xs)
    header = "  i |   x_i  |     f_i      " + "".join(
        f"| разность {k}-го пор. " for k in range(1, n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{i:3d} | {xs[i]:6.3f} | {table[0][i]:12.5f} "
        for k in range(1, n):
            row += f"| {table[k][i]:19.5f} " if i < n - k else "|" + " " * 21
        print(row)
    print()


# ЕШЕНИЕ ДЛЯ ОДНОГО НАБОРА УЗЛОВ
def solve_case(title, xs, x_star):
    """Строит оба многочлена для узлов xs и считает погрешность в x_star."""
    n = len(xs) - 1                        # степень многочлена (у нас 3)
    ys = [f(x) for x in xs]                # Y_i = f(X_i)

    print("=" * 70)
    print(title)
    print("=" * 70)

    # Многочлен Лагранжа
    coefs, w_primes = lagrange_coefficients(xs, ys)
    print("Многочлен Лагранжа. Таблица (как в примере 3.1):")
    print("  i |   x_i  |     f_i      |  omega'(x_i)  | f_i/omega'(x_i) |  X* - x_i")
    print("----+--------+--------------+---------------+-----------------+----------")
    for i in range(n + 1):
        print(f"{i:3d} | {xs[i]:6.3f} | {ys[i]:12.5f} | {w_primes[i]:13.5f} |"
              f" {coefs[i]:15.5f} | {x_star - xs[i]:8.3f}")
    print(f"\nL(x) = {lagrange_formula(xs, coefs)}\n")
    L_star = lagrange_value(xs, coefs, x_star)

    # Многочлен Ньютона
    table = divided_differences(xs, ys)
    print("Многочлен Ньютона. Таблица разделённых разностей:")
    print_divided_differences(xs, table)
    print(f"P(x) = {newton_formula(xs, table)}\n")
    P_star = newton_value(xs, table, x_star)

    # Проверка: многочлены проходят через узлы
    max_node_err = 0.0
    for i in range(n + 1):
        max_node_err = max(max_node_err,
                           abs(lagrange_value(xs, coefs, xs[i]) - ys[i]),
                           abs(newton_value(xs, table, xs[i]) - ys[i]))
    print(f"Проверка в узлах: max |P(x_i) - f_i| = {max_node_err:.2e}\n")

    # Погрешность в X*
    y_star = f(x_star)
    print(f"Значения в точке X* = {x_star}:")
    print(f"  Лагранж  L(X*) = {L_star:.6f}")
    print(f"  Ньютон   P(X*) = {P_star:.6f}   (совпадает с Лагранжем:"
          f" разница {abs(L_star - P_star):.1e})")
    print(f"  Точно    f(X*) = {y_star:.6f}")
    print(f"  Фактическая погрешность |f(X*) - L(X*)| = {abs(y_star - L_star):.6f}")

    # Априорная оценка (3.9): M4 = max |f''''| на [x0, x3].
    # f''''(x) = 120/x^6 убывает при x > 0, поэтому максимум - в левом узле.
    omega = 1.0
    for x in xs:
        omega *= x_star - x
    M = f4(min(xs))
    bound = M / math.factorial(n + 1) * abs(omega)
    print(f"  Априорная оценка (3.9): M4 / 4! * |omega4(X*)| ="
          f" {M:.3e} / 24 * {abs(omega):.5f} = {bound:.3e}\n")
    return abs(y_star - L_star)



def check_textbook():
    """
    Пример 3.1: y = ln x, X_i = 0.1, 0.5, 0.9, 1.3, X* = 0.8 -> L(0.8) = -0.20036.
    Пример 3.2: y = sin(pi x / 6), X_i = 0, 1, 2, 3, X* = 1.5 -> P(1.5) = 0.70589.
    """
    xs = [0.1, 0.5, 0.9, 1.3]
    coefs, _ = lagrange_coefficients(xs, [math.log(x) for x in xs])
    L = lagrange_value(xs, coefs, 0.8)
    xs2 = [0.0, 1.0, 2.0, 3.0]
    table = divided_differences(xs2, [math.sin(math.pi * x / 6) for x in xs2])
    P = newton_value(xs2, table, 1.5)
    print("Проверка на примерах методички:")
    print(f"  пример 3.1: L(0.8) = {L:.5f}   (в методичке -0.20036)")
    print(f"  пример 3.2: P(1.5) = {P:.5f}   (в методичке  0.70589)\n")


# ОСНОВНАЯ ПРОГРАММА
def main():
    x_star = 0.8
    print(" y = 1/x^2 + x^2,  X* = 0.8\n")

    err_a = solve_case("а) X_i = 0.1, 0.5, 0.9, 1.3", [0.1, 0.5, 0.9, 1.3], x_star)
    err_b = solve_case("б) X_i = 0.1, 0.5, 1.1, 1.3", [0.1, 0.5, 1.1, 1.3], x_star)

    print("=" * 70)
    print("Вывод")
    print("=" * 70)
    print(f"Погрешность в X* = 0.8:  а) {err_a:.4f},  б) {err_b:.4f}.")
    print("Лагранж и Ньютон дают одно и то же значение: это один и тот же\n"
          "многочлен, записанный двумя способами. Погрешность большая, потому\n"
          "что функция 1/x^2 + x^2 очень резко растёт около x = 0.1\n"
          "(f(0.1) = 100, а f(0.5) уже 4.25), и многочлен 3-й степени не может\n"
          "одновременно пройти через такую точку и точно повторить форму\n"
          "функции между узлами. Априорная оценка (3.9) это подтверждает:\n"
          "f'''' = 120/x^6 в точке 0.1 огромна (1.2e8). В случае б) узлы\n"
          "дальше от X* = 0.8 (ближайшие 0.5 и 1.1), поэтому |omega4(X*)| и\n"
          "погрешность больше, чем в случае а).\n")

    check_textbook()


if __name__ == "__main__":
    main()
