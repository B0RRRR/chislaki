'''Задача. Функция задана только таблицей из пяти точек, x = 0, 1, 2, 3, 4. 
Нужно найти её первую и вторую производные в точке X* = 2.0.'''

# ФОРМУЛЫ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ
def slope(xs, ys, i):
    """
    Наклон прямой через точки i и i+1 (разделённая разность 1-го порядка):
        (y_{i+1} - y_i) / (x_{i+1} - x_i).
    """
    return (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])


def first_derivative_2nd_order(xs, ys, i, x):
    """
    производная параболы, проведённой через
    точки i, i+1, i+2, в точке x.
    Первое слагаемое - наклон на [x_i, x_{i+1}], второе - поправка
    на "изгиб" (разность наклонов соседних отрезков).
    y'(x) ≈ s₁ + (s₂ − s₁) / (xᵢ₊₂ − xᵢ) · (2x − xᵢ − xᵢ₊₁)

    """
    s1 = slope(xs, ys, i)              # наклон на [x_i, x_{i+1}]
    s2 = slope(xs, ys, i + 1)          # наклон на [x_{i+1}, x_{i+2}]
    curvature = (s2 - s1) / (xs[i + 2] - xs[i])
    return s1 + curvature * (2 * x - xs[i] - xs[i + 1])


def second_derivative(xs, ys, i):
    """
    Формула (3.21): вторая производная параболы через точки i, i+1, i+2:
        2 * (s2 - s1) / (x_{i+2} - x_i).
    """
    s1 = slope(xs, ys, i)
    s2 = slope(xs, ys, i + 1)
    return 2 * (s2 - s1) / (xs[i + 2] - xs[i])


def node_index(xs, x):
    """Номер узла, совпадающего с x (с допуском на округление), или None."""
    for k in range(len(xs)):
        if abs(xs[k] - x) < 1e-12:
            return k
    return None


#  РАСЧЁТ ДЛЯ ОДНОЙ ТАБЛИЦЫ
def derivatives_at(xs, ys, x_star, verbose=True):
    """
    Считает в точке x_star:
        левостороннюю и правостороннюю производные,
        первую производную 2-го порядка точности,
        вторую производную.
    Возвращает словарь с результатами.
    """

    # Находим узел, совпадающий с X*
    k = node_index(xs, x_star)
    if k is None or k == 0 or k == len(xs) - 1:
        raise ValueError("X* должна совпадать с внутренним узлом таблицы")

    # Левый отрезок [x_{k-1}, x_k] и правый [x_k, x_{k+1}].
    # наклон левосторонний и правосторонний (разделённые разности 1-го порядка):
    left = slope(xs, ys, k - 1)
    right = slope(xs, ys, k)
    # Парабола через x_{k-1}, x_k, x_{k+1}: в формулах (3.20), (3.21) i = k - 1.
    i = k - 1
    d1 = first_derivative_2nd_order(xs, ys, i, x_star)

    d2 = second_derivative(xs, ys, i)

    if verbose:
        xl, xc, xr = xs[k - 1], xs[k], xs[k + 1]
        yl, yc, yr = ys[k - 1], ys[k], ys[k + 1]
        print(f"X* = {x_star} совпадает с узлом x_{k}. Используем точки"
              f" x_{k - 1} = {xl}, x_{k} = {xc}, x_{k + 1} = {xr}.\n")

        print("1) Левосторонняя производная,  на [x_{k-1}, x_k]:")
        print(f"   y'({x_star}) ≈ (y_{k} - y_{k - 1}) / (x_{k} - x_{k - 1})"
              f" = ({yc} - {yl}) / ({xc} - {xl}) = {left:.5f}")
        print("   Правосторонняя производная, на [x_k, x_{k+1}]:")
        print(f"   y'({x_star}) ≈ (y_{k + 1} - y_{k}) / (x_{k + 1} - x_{k})"
              f" = ({yr} - {yc}) / ({xr} - {xc}) = {right:.5f}")
        print("   (обе - первого порядка точности)\n")

        print("2) Первая производная со вторым порядком точности, :")
        print(f"   y'({x_star}) ≈ {left:.5f} + ({right:.5f} - ({left:.5f})) / ({xr} - {xl})"
              f" * (2*{x_star} - {xl} - {xc}) = {d1:.5f}")
        print(f"   (на равномерной сетке это полусумма левой и правой:"
              f" ({left:.4f} + ({right:.4f})) / 2 = {(left + right) / 2:.4f})\n")

        print("3) Вторая производная, формула (3.21):")
        print(f"   y''({x_star}) ≈ 2 * ({right:.5f} - ({left:.5f})) / ({xr} - {xl})"
              f" = {d2:.5f}\n")

    return {"left": left, "right": right, "d1": d1, "d2": d2}



# ПРОВЕРКА НА ПРИМЕРЕ ИЗ МЕТОДИЧКИ
def check_textbook():
    """
    Пример п. 3.3: x = 0, 0.1, ..., 0.4; y = e^x; X* = 0.2 ->
    1.1623 (левая), 1.2846 (правая), 1.2234 (3.20), 1.2224 (3.21).
    """
    import math
    xs = [0.0, 0.1, 0.2, 0.3, 0.4]
    ys = [math.exp(x) for x in xs]
    r = derivatives_at(xs, ys, 0.2, verbose=False)
    print("Проверка на примере методички (y = e^x, X* = 0.2):")
    print(f"  левая  {r['left']:.4f}  (1.1623),  правая {r['right']:.4f}  (1.2846)")
    print(f"  (3.20) {r['d1']:.4f}  (1.2234),  (3.21) {r['d2']:.4f}  (1.2224)\n")



def main():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.0, 0.86603, 1.0, 0.0, -2.0]
    x_star = 2.0

    print("  x_i: " + "  ".join(f"{x:8.4f}" for x in xs))
    print("  y_i: " + "  ".join(f"{y:8.5f}" for y in ys))
    print()

    r = derivatives_at(xs, ys, x_star)

    print("Итог:")
    print(f"  y'({x_star})  ≈ {r['left']:.5f} (левая),  {r['right']:.5f} (правая),"
          f"  {r['d1']:.5f} (2-й порядок точности)")
    print(f"  y''({x_star}) ≈ {r['d2']:.5f}\n")


    check_textbook()


if __name__ == "__main__":
    main()
