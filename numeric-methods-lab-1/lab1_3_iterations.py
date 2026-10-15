
import math

# Максимально допустимое число итераций.
MAX_ITER = 10000



def print_matrix(name, M, fmt="{:10.4f}"):
    """
    Печатает матрицу M построчно.
    """
    print(f"{name} =")
    for row in M:
        # Превращаем каждое число строки в текст нужного формата
        # и склеиваем их через пробел.
        numbers = [fmt.format(v) for v in row]
        print("  [" + " ".join(numbers) + " ]")
    print()


def print_vector(name, v, fmt="{:10.4f}"):
    """Печатает вектор v в одну строку."""
    numbers = [fmt.format(x) for x in v]
    print(f"{name} = [" + " ".join(numbers) + " ]")


def vec_norm_c(v):
    """
    Норма вектора ||v||_c = max |v_i|
    самая большая по модулю компонента вектора
    """
    result = 0.0
    for x in v:
        if abs(x) > result:
            result = abs(x)
    return result


def mat_norm_c(M):
    """
    Норма матрицы ||M||_c = max_i sum_j |m_ij|.
    Для каждой строки считаем сумму модулей её элементов,
    из этих сумм берём наибольшую.
    """
    result = 0.0
    for row in M:
        row_sum = 0.0                 # сумма модулей элементов этой строки
        for x in row:
            row_sum += abs(x)
        if row_sum > result:          # запоминаем максимальную сумму
            result = row_sum
    return result


def mat_vec(A, x):
    """
    Умножение матрицы A на вектор x.
    i-я компонента результата = sum_j A[i][j] * x[j].
    Нужна только для вычисления невязки A x - b.
    """
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(len(x)):
            s += A[i][j] * x[j]
        result[i] = s
    return result


def is_diagonally_dominant(A):
    """
    Проверка диагонального преобладания:
        |a_ii| > sum_{j != i} |a_ij|   для каждой строки i.
    Если оно есть, метод Якоби (и Зейделя) гарантированно сходится.
    """
    n = len(A)
    for i in range(n):
        # Сумма модулей всех элементов строки, кроме диагонального.
        off_diag_sum = 0.0
        for j in range(n):
            if j != i:
                off_diag_sum += abs(A[i][j])
        # Достаточно одной плохой строки, чтобы преобладания не было.
        if abs(A[i][i]) <= off_diag_sum:
            return False
    return True


# приведение системы к виду  x = beta + alpha * x
def to_iterative_form(A, b):
    """
    Строит beta и alpha по формулам
    Требуется a_ii != 0 для всех i.

        beta_i    = b_i / a_ii                 - свободный член;
        alpha_ij  = -a_ij / a_ii   (j != i)    - коэффициент при x_j;
        alpha_ii  = 0                          - x_i не выражается через себя.
    """
    n = len(A)

    # матрица alpha n x n из нулей и вектор beta из нулей.
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n

    for i in range(n):
        # На диагональный элемент делим, поэтому он не должен быть нулём.
        if A[i][i] == 0:
            raise ValueError(f"Нулевой диагональный элемент a_{i + 1}{i + 1}")

        beta[i] = b[i] / A[i][i]

        for j in range(n): # для каждого коэффициента строки
            if j != i:
                alpha[i][j] = -A[i][j] / A[i][i] # alpha_ij = -a_ij / a_ii
            # при j == i alpha[i][i] так и остаётся равным 0

    return alpha, beta


#  Метод Якоби
def simple_iterations(A, b, eps):
    """
    Метод простых итераций:
        x^(0) = beta,
        x^(k) = beta + alpha * x^(k-1),   k = 1, 2, ...

    Возвращает три значения:
        x       - найденное решение;
        k       - сколько итераций понадобилось;
        history - список оценок ошибки eps^(k) на каждой итерации
                  (нужен, чтобы потом напечатать таблицу).
    """
    # Подготовка
    alpha, beta = to_iterative_form(A, b)
    n = len(beta)                          # число неизвестных
    norm_alpha = mat_norm_c(alpha)         # ||alpha|| = 0.7083

    # Оценка ошибки по формуле:
    #  eps^(k) = ||alpha|| / (1 - ||alpha||) * ||x^(k) - x^(k-1)||
    # Множитель перед ||x^(k) - x^(k-1)|| не меняется от итерации к итерации,
    # поэтому считаем его один раз заранее
    if norm_alpha < 1:
        coef = norm_alpha / (1 - norm_alpha)
    else:
        # При ||alpha|| >= 1 формула не работает
        # Тогда считаем просто ||x^(k) - x^(k-1)||.
        coef = 1.0

    # Начальное приближение x^(0) = beta.
    # beta[:] - это копия списка. Если написать x = beta, то x и beta
    # станут одним и тем же списком, и изменение x испортит beta.
    x = beta[:]

    history = []  # сюда складываем eps^(k)

    # Итерации
    # k принимает значения 1, 2, 3, ..., MAX_ITER.
    for k in range(1, MAX_ITER + 1):

        # 1. Вычисляем новое приближение x^(k) = beta + alpha * x^(k-1)
        # Новые значения складываем в отдельный список x_new.
        # Старый список x в это время не меняется, поэтому все компоненты
        # считаются по старой догадке
        x_new = [0.0] * n
        for i in range(n):
            # s = alpha_i1 * x_1 + alpha_i2 * x_2 + ... + alpha_in * x_n
            # (x - старое приближение x^(k-1))
            s = 0.0
            for j in range(n):
                s += alpha[i][j] * x[j]
            x_new[i] = beta[i] + s

        # 2. Оценка ошибки на этой итерации
        # Сначала находим, насколько изменилась догадка:
        # diff = x^(k) - x^(k-1)  (покомпонентно).
        diff = [0.0] * n
        for i in range(n):
            diff[i] = x_new[i] - x[i]
        # eps^(k) = coef * ||x^(k) - x^(k-1)||
        err = coef * vec_norm_c(diff)
        history.append(err)                # запоминаем для таблицы

        # 3. Новое приближение становится текущим
        x = x_new

        # 4. Проверка окончания
        if err <= eps:
            return x, k, history           # точность достигнута - выходим

    # если ошибка
    raise RuntimeError("Метод простых итераций не сошёлся "
                       f"за {MAX_ITER} итераций")


# Метод Зейделя
def seidel(A, b, eps):
    """
    Метод Зейделя. Отличается от метода Якоби одним:
    при вычислении x_i на k-й итерации используются
        x_1 .. x_{i-1} - уже новые (посчитаны на этой же итерации),
        x_{i+1} .. x_n - ещё старые (с прошлой итерации).

    Возвращает (x, k, history) - так же, как simple_iterations.
    """
    # Подготовка
    alpha, beta = to_iterative_form(A, b)
    n = len(beta)
    norm_alpha = mat_norm_c(alpha)

    # Для оценки ошибки Зейделя нужна матрица C - верхняя треугольная
    # часть alpha (диагональ и всё, что правее неё). Именно эти
    # коэффициенты в методе Зейделя умножаются на старые значения x.
    # Элементы левее диагонали заменяем нулями.
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j >= i:                     # диагональ и правее
                C[i][j] = alpha[i][j]
    norm_C = mat_norm_c(C)

    # Оценка ошибки: eps^(k) = ||C|| / (1 - ||alpha||) * ||x^(k) - x^(k-1)||.
    # Множитель считаем заранее (у нас 0.5 / 0.2917 = 1.7143).
    if norm_alpha < 1:
        coef = norm_C / (1 - norm_alpha)
    else:
        coef = 1.0

    # Начальное приближение x^(0) = beta.
    # ниже мы записываем новые значения прямо в x.
    x = beta[:]

    history = []

    # Итерации
    for k in range(1, MAX_ITER + 1):

        # Запоминаем приближение до начала итерации, чтобы потом
        # посчитать, насколько оно изменилось.
        # если написать x_old = x, то x_old будет тем же
        # списком, что и x, и после обновления x разница всегда будет 0.
        x_old = x[:]

        # 1. Обновляем компоненты по очереди
        for i in range(n):
            s = 0.0
            for j in range(n):
                # Здесь используется список x, который мы же меняем в этом цикле:
                #   при j < i в x[j] уже лежит НОВОЕ значение (посчитано выше
                #   на этой итерации);
                #   при j > i в x[j] ещё СТАРОЕ значение.
                #   При j == i alpha[i][i] = 0, так что старое x[i] не влияет.
                s += alpha[i][j] * x[j]
            # Сразу записываем новое значение в x - следующие компоненты
            # на этой же итерации будут его использовать.
            x[i] = beta[i] + s

        # 2. Оценка ошибки
        diff = [0.0] * n
        for i in range(n):
            diff[i] = x[i] - x_old[i]      # насколько изменилась каждая компонента
        err = coef * vec_norm_c(diff)
        history.append(err)

        # 3. Проверка окончания
        if err <= eps:
            return x, k, history

    raise RuntimeError(f"Метод Зейделя не сошёлся за {MAX_ITER} итераций")


# Оценка числа итераций

def a_priori_iterations(A, b, eps):
    """
    Оценка числа итераций для метода простых итераций.
    Формула из методички:
        k + 1 >= (lg eps - lg||beta|| + lg(1 - ||alpha||)) / lg||alpha||.
    Имеет смысл только при ||alpha|| < 1.
    """
    alpha, beta = to_iterative_form(A, b)
    na = mat_norm_c(alpha)
    if na >= 1:
        return None                        # формула неприменима

    # math.log10 - десятичный логарифм (lg).
    numerator = math.log10(eps) - math.log10(vec_norm_c(beta)) + math.log10(1 - na)
    k_plus_1 = numerator / math.log10(na)  # получили оценку для k + 1

    # Число итераций - целое, поэтому округляем ВВЕРХ (math.ceil),
    # чтобы оценка точно гарантировала нужную точность.
    # max(0, ...) - на случай, если при очень грубой eps получилось < 0.
    return max(0, math.ceil(k_plus_1 - 1))



def main():
    # Исходные данные
    A = [
        [18,  0, -2,  7],
        [-1, 14, -3,  2],
        [ 5,  5, 26,  7],
        [-2, -6,  9, 24],
    ]
    b = [50, 2, 273, 111]
    eps = 0.01                             # заданная точность

    print("=" * 60)
    print("Задание 1.3. Метод простых итераций и метод Зейделя")
    print("=" * 60)
    print_matrix("A", A, fmt="{:6.0f}")
    print_vector("b", b, fmt="{:6.0f}")
    print(f"Точность eps = {eps}\n")

    # Проверка условий сходимости
    alpha, beta = to_iterative_form(A, b)
    print_matrix("alpha", alpha)
    print_vector("beta", beta)
    print()

    norm_alpha = mat_norm_c(alpha)
    if is_diagonally_dominant(A):
        print("Диагональное преобладание: есть")
    else:
        print("Диагональное преобладание: нет")

    if norm_alpha < 1:
        print(f"||alpha||_c = {norm_alpha:.4f} < 1 -> "
              "достаточное условие сходимости выполнено")
    else:
        print(f"||alpha||_c = {norm_alpha:.4f} >= 1 -> "
              "сходимость не гарантирована")
    print()

    # Метод простых итераций 
    # Функция возвращает три значения - сразу раскладываем их в три переменные.
    x_si, k_si, hist_si = simple_iterations(A, b, eps)
    print("Метод простых итераций:")
    for i in range(len(x_si)):
        print(f"  x{i + 1} = {x_si[i]:.6f}")
    print(f"  Число итераций: {k_si}")
    print(f"  Априорная оценка числа итераций: {a_priori_iterations(A, b, eps)}")
    # Невязка r = A x - b: подставляем ответ в систему.
    # Для приближённого решения (eps = 0.01) она тоже порядка 0.01.
    Ax = mat_vec(A, x_si)
    r = [Ax[i] - b[i] for i in range(len(b))]
    print_vector("  Невязка A x - b", r, fmt="{:10.2e}")
    print()

    # Метод Зейделя
    x_s, k_s, hist_s = seidel(A, b, eps)
    print("Метод Зейделя:")
    for i in range(len(x_s)):
        print(f"  x{i + 1} = {x_s[i]:.6f}")
    print(f"  Число итераций: {k_s}")
    Ax = mat_vec(A, x_s)
    r = [Ax[i] - b[i] for i in range(len(b))]
    print_vector("  Невязка A x - b", r, fmt="{:10.2e}")
    print()

    # Таблица: оценка ошибки на каждой итерации
    # Методы сделали разное число итераций, поэтому таблица длиной
    # в большее из них, а у "короткого" метода лишние ячейки пустые.
    print("Оценка погрешности eps^(k) по итерациям (eps = 0.01):")
    print("   k | простые итерации |     Зейдель")
    print("-----+------------------+-----------------")
    for k in range(max(k_si, k_s)):
        if k < k_si:
            si = f"{hist_si[k]:16.6e}"
        else:
            si = " " * 16
        if k < k_s:
            sd = f"{hist_s[k]:16.6e}"
        else:
            sd = ""
        print(f"{k + 1:4d} | {si} | {sd}")
    print()

    # Анализ: сколько итераций нужно для разной точности
    # Запускаем оба метода для eps = 0.1, 0.01, ..., 1e-10 и сравниваем.
    print("Зависимость числа итераций от заданной точности:")
    print("    eps    | простые итерации | Зейдель | априорная оценка")
    print("-----------+------------------+---------+-----------------")
    for p in range(1, 11):
        e = 10.0 ** (-p)                   # e = 10^(-p)
        # Нужно только число итераций (второе значение),
        # поэтому x и history пишем в "_" - так в Python обозначают
        # значения, которые не будут использоваться.
        _, k1, _ = simple_iterations(A, b, e)
        _, k2, _ = seidel(A, b, e)
        k_apr = a_priori_iterations(A, b, e)
        print(f"  1e-{p:<2d}    | {k1:16d} | {k2:7d} | {k_apr:16d}")
    print()
    print("Вывод: число итераций растёт примерно линейно с ростом -lg(eps),\n"
          "т.е. каждая итерация уменьшает погрешность в постоянное число раз.\n"
          "Метод Зейделя сходится быстрее метода простых итераций, так как\n"
          "сразу использует уточнённые значения компонент. Априорная оценка\n"
          "даёт завышенное число итераций.")



if __name__ == "__main__":
    main()
