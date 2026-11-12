"""
Лабораторная работа 3, задание 3.5. Вариант 26.

Задача. Вычислить определённый интеграл

        F = ∫ y dx  на [X0, Xk],      y = x^2 * sqrt(36 - x^2),
        X0 = 1,  Xk = 5,  h1 = 1.0,  h2 = 0.5

методами прямоугольников, трапеций и Симпсона с шагами h1 и h2.
Оценить погрешность методом Рунге-Ромберга(-Ричардсона) и уточнить результат.

Идея всех трёх методов (п. 3.4 методички): разбиваем [X0, Xk] на отрезки длины h
и на каждом отрезке заменяем y(x) простым многочленом, интеграл от которого
считается точно:
    прямоугольники - константой (значение в середине отрезка),     порядок h^2;
    трапеции       - прямой через концы отрезка,                   порядок h^2;
    Симпсон        - параболой через три соседних узла,            порядок h^4.

Точное значение интеграла здесь тоже можно найти (табличный интеграл), поэтому
сравниваем численные результаты с ним - так видно, насколько хорошо работает
оценка Рунге-Ромберга.
"""

import math


def f(x):
    """y = x^2 * sqrt(36 - x^2)."""
    return x ** 2 * math.sqrt(36 - x ** 2)


def antiderivative(x):
    """
    Первообразная для x^2 * sqrt(a^2 - x^2) при a = 6:

        F(x) = x (2x^2 - a^2) sqrt(a^2 - x^2) / 8  +  a^4 / 8 * arcsin(x / a).

    """
    a = 6.0
    return (x * (2 * x ** 2 - a ** 2) * math.sqrt(a ** 2 - x ** 2) / 8
            + a ** 4 / 8 * math.asin(x / a))


def exact_integral(a, b):
    """Точное значение интеграла по формуле Ньютона-Лейбница: F(b) - F(a)."""
    return antiderivative(b) - antiderivative(a)


# СЕТКА
def make_grid(a, b, h):
    """
    Узлы x_i = a + i*h, i = 0..N, где N = (b - a) / h.

    N округляем до целого и проверяем, что h действительно укладывается
    в отрезок целое число раз
    """
    n = round((b - a) / h)
    if abs(a + n * h - b) > 1e-9:
        raise ValueError(f"Шаг h = {h} не делит отрезок [{a}, {b}] нацело")
    # Считаем узлы через i*h, а не накапливая x += h, чтобы не копить
    # ошибку округления.
    return [a + i * h for i in range(n + 1)]


#     I_i = ∫ y dx  от x_0 до x_i,   i = 0..N,

def rectangles(func, xs):
    """
    Формула прямоугольников:

        F ≈ h * [ y((x_0+x_1)/2) + y((x_1+x_2)/2) + ... + y((x_{N-1}+x_N)/2) ].

    На каждом отрезке [x_{i-1}, x_i] функция заменяется константой - своим
    значением в середине отрезка 
    Оценка погрешности: |R| <= h^2 * M2 * (b - a) / 24.
    """
    cum = [0.0]
    for i in range(1, len(xs)):
        h = xs[i] - xs[i - 1] # длина текущего отрезка
        mid = (xs[i - 1] + xs[i]) / 2
        cum.append(cum[-1] + h * func(mid))
    return cum


def trapezoids(func, xs):
    """
    Формула трапеций:

        F ≈ h * [ y_0/2 + y_1 + y_2 + ... + y_{N-1} + y_N/2 ].

    На каждом отрезке функция заменяется прямой через (x_{i-1}, y_{i-1})
    и (x_i, y_i); площадь под ней - площадь трапеции h * (y_{i-1} + y_i) / 2.
    
    """
    cum = [0.0]
    for i in range(1, len(xs)):
        h = xs[i] - xs[i - 1]
        cum.append(cum[-1] + h * (func(xs[i - 1]) + func(xs[i])) / 2)
    return cum


def simpson(func, xs):
    """
    Формула парабол:

        F ≈ h/3 * [ y_0 + 4y_1 + 2y_2 + 4y_3 + ... + 2y_{N-2} + 4y_{N-1} + y_N ].

    Берём 2 соседних отрезков [x_{i-2}, x_i] и проводим параболу через три
    узла x_{i-2}, x_{i-1}, x_i. Интеграл от неё равен
        h/3 * (y_{i-2} + 4 y_{i-1} + y_i).
    
    Оценка погрешности : |R| <= (b - a) * h^4 * M4 / 180.
    """
    n = len(xs) - 1
    if n % 2 != 0:
        raise ValueError("Для формулы Симпсона нужно чётное число отрезков")
    cum = [0.0] + [None] * n
    for i in range(2, n + 1, 2):
        h = xs[i] - xs[i - 1]
        parabola = h / 3 * (func(xs[i - 2]) + 4 * func(xs[i - 1]) + func(xs[i]))
        cum[i] = cum[i - 2] + parabola
    return cum


METHODS = [
    ("прямоугольников", rectangles, 2),
    ("трапеций", trapezoids, 2),
    ("Симпсона", simpson, 4),
]


def runge_romberg(f_fine, f_coarse, k, p):
    """
    Пусть F_h - результат с мелким шагом h, F_kh - с крупным
    шагом k*h, а метод имеет порядок p, т.е. F = F_h + C*h^p + ... .
    Тогда
    Вычитая одно из другого, исключаем неизвестную C:
        C h^p ≈ (F_h - F_kh) / (k^p - 1).

    """
    correction = (f_fine - f_coarse) / (k ** p - 1)
    return f_fine + correction, abs(correction)


def fmt(v, width=12):
    """Число в 5 знаков после запятой или пустая клетка для None."""
    return " " * width if v is None else f"{v:{width}.5f}"


def integrate_with_step(func, a, b, h, verbose=True):
    """
    Считает интеграл тремя методами с шагом h.
    """
    xs = make_grid(a, b, h)
    tables = {name: method(func, xs) for name, method, _ in METHODS}

    if verbose:
        print(f"Шаг h = {h}  (N = {len(xs) - 1} отрезков)")
        print("   i      x_i         y_i      " +
              "".join(f"{name:>16s}" for name, _, _ in METHODS))
        for i, x in enumerate(xs):
            row = "".join("    " + fmt(tables[name][i]) for name, _, _ in METHODS)
            print(f"  {i:2d}  {x:7.3f}  {func(x):11.5f}  {row}")
        print()

    return {name: tables[name][-1] for name, _, _ in METHODS}


# ПОЛНЫЙ РАСЧЁТ
def solve(func, a, b, h1, h2, exact=None, verbose=True):
    """
    Считает интеграл с шагами h1 и h2, затем для каждого метода применяет
    Рунге-Ромберга (k = h1 / h2) и, если задано exact, сравнивает с точным
    значением. Возвращает словарь с результатами по методам.
    """
    # Рунге-Ромберг: "мелкий" шаг - меньший из двух.
    h_fine, h_coarse = min(h1, h2), max(h1, h2)
    k = h_coarse / h_fine                      # во сколько раз шаги отличаются

    res_coarse = integrate_with_step(func, a, b, h_coarse, verbose)
    res_fine = integrate_with_step(func, a, b, h_fine, verbose)

    results = {}
    for name, _, p in METHODS:
        refined, rr_err = runge_romberg(res_fine[name], res_coarse[name], k, p)
        results[name] = {
            "coarse": res_coarse[name], "fine": res_fine[name],
            "p": p, "refined": refined, "rr_err": rr_err,
        }
        if exact is not None:
            results[name]["true_err_fine"] = abs(exact - res_fine[name])
            results[name]["true_err_refined"] = abs(exact - refined)
    return results, k


def main():
    a, b = 1.0, 5.0
    h1, h2 = 1.0, 0.5

    print(f"  F = ∫ x^2 sqrt(36 - x^2) dx  на [{a}, {b}],  h1 = {h1}, h2 = {h2}\n")

    exact = exact_integral(a, b)
    results, k = solve(f, a, b, h1, h2, exact=exact)

    # Сводная таблица
    print(f"Точное значение: F = {exact:.6f}\n")
    print(f"{'метод':>16s}  {'F(h1=' + str(h1) + ')':>12s}  {'F(h2=' + str(h2) + ')':>12s}"
          f"  {'|F - F(h2)|':>12s}")
    for name, r in results.items():
        print(f"{name:>16s}  {r['coarse']:12.6f}  {r['fine']:12.6f}  {r['true_err_fine']:12.2e}")
    print()

    print(f"Рунге-Ромберг (k = h1/h2 = {k:g}):  F ≈ F(h2) + (F(h2) - F(h1)) / (k^p - 1)")
    print(f"{'метод':>16s}  {'p':>2s}  {'оценка погр.':>12s}  {'истинная погр.':>14s}"
          f"  {'уточнённое F':>12s}  {'его погр.':>10s}")
    for name, r in results.items():
        print(f"{name:>16s}  {r['p']:2d}  {r['rr_err']:12.2e}  {r['true_err_fine']:14.2e}"
              f"  {r['refined']:12.6f}  {r['true_err_refined']:10.2e}")
    print()
    print("  'оценка погр.' - погрешность F(h2) по Рунге-Ромбергу,")
    print("  'истинная погр.' - |F_точн - F(h2)|; они должны быть одного порядка.\n")

    check_textbook()


# ПРОВЕРКА НА ПРИМЕРЕ ИЗ МЕТОДИЧКИ
def check_textbook():
    """
    Пример п. 3.4: y = x / (3x + 4)^2, X0 = -1, Xk = 1, h1 = 0.5, h2 = 0.25.
    В методичке: h = 0.5:  -0.11914 (прям.), -0.27663 (трап.), -0.20558 (Симпс.);
                 h = 0.25: -0.14914,         -0.19789,          -0.17164.

    Замечание: для прямоугольников с h = 0.25 у нас -0.14931. Все промежуточные
    суммы таблицы методички (-0.11570 ... -0.15430) совпадают, а на последнем
    шаге там прибавлено 0.00516 вместо верного h * y(0.875) =
    0.25 * 0.875 / 6.625^2 = 0.00498 - опечатка в методичке.
    """
    g = lambda x: x / (3 * x + 4) ** 2
    print("Проверка на примере методички (y = x / (3x+4)^2, [-1, 1]):")
    for h, expected in ((0.5, (-0.11914, -0.27663, -0.20558)),
                        (0.25, (-0.14914, -0.19789, -0.17164))):
        r = integrate_with_step(g, -1.0, 1.0, h, verbose=False)
        got = "  ".join(f"{r[name]:9.5f}" for name, _, _ in METHODS)
        want = "  ".join(f"{v:9.5f}" for v in expected)
        print(f"  h = {h:<5}  получено: {got}   в методичке: {want}")
    print("  (расхождение -0.14931 / -0.14914 - арифметическая опечатка в методичке\n"
          "   на последнем шаге таблицы; все промежуточные суммы совпадают)\n")


if __name__ == "__main__":
    main()
