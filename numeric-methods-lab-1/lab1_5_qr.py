"""
Лабораторная работа 1, задание 1.5. Вариант 26.

QR-разложение матрицы (преобразованием Хаусхолдера) и QR-алгоритм
нахождения всех собственных значений произвольной (несимметричной)
матрицы. Входные данные: матрица и точность.

Матрица варианта 26:
        | -9  -9  -3 |
    A = | -9   0  -2 |
        | -5  -1  -4 |

Теория (методичка, п. 1.2.4).

1) QR-разложение: A = Q R, Q - ортогональная (Q^T Q = E),
   R - верхняя треугольная.
   Строится с помощью матриц Хаусхолдера (1.29):
        H = E - 2 v v^T / (v^T v).
   На k-м шаге вектор v выбирается так, чтобы H обнулила
   поддиагональные элементы k-го столбца (1.30):
        v_i = 0                                   (i < k),
        v_k = a_kk + sign(a_kk) * sqrt(sum_{j>=k} a_jk^2),
        v_i = a_ik                                (i > k).
   После n-1 шагов:  R = H_{n-1} ... H_1 A,   Q = H_1 H_2 ... H_{n-1}.

2) QR-алгоритм:
        A^(0) = A,
        A^(k) = Q^(k) R^(k)  (разложение),
        A^(k+1) = R^(k) Q^(k) (перемножение в обратном порядке).
   Все A^(k) подобны A, т.е. имеют те же собственные значения.
   A^(k) сходится к верхней треугольной (если все собственные значения
   вещественные) или к квазитреугольной матрице (блоки 2x2 на диагонали
   соответствуют комплексно-сопряжённым парам).

   Критерии окончания:
   * вещественное собственное значение в столбце j:
        sqrt( sum_{l>j} (a_lj^(k))^2 ) <= eps,  lambda = a_jj^(k);
   * комплексная пара из блока (j, j+1): собственные значения блока
        (a_jj - lambda)(a_{j+1,j+1} - lambda) = a_{j,j+1} a_{j+1,j}
     почти не меняются от итерации к итерации:
        |lambda^(k) - lambda^(k-1)| <= eps.

Программа написана на "чистом" Python без сторонних библиотек
(комплексные числа встроены в Python: модуль cmath).
"""

import cmath
import math

MAX_ITER = 1000  # защита от зацикливания


# ----------------------------------------------------------------------
#                    Вспомогательные функции
# ----------------------------------------------------------------------

def print_matrix(name, M, fmt="{:10.4f}"):
    """Печатает матрицу M с заголовком name (значения ~1e-12 печатаются как 0)."""
    print(f"{name} =")
    for row in M:
        print("  [" + " ".join(fmt.format(v if abs(v) > 1e-12 else 0.0)
                               for v in row) + " ]")
    print()


def identity(n):
    """Единичная матрица порядка n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(M):
    """Транспонирование."""
    return [list(col) for col in zip(*M)]


def mat_mul(A, B):
    """Произведение матриц."""
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def sign(x):
    """Знак числа; для нуля считаем sign(0) = 1, чтобы v_k не обнулился."""
    return 1.0 if x >= 0 else -1.0


def determinant(M):
    """
    Определитель методом Гаусса с выбором главного элемента
    (нужен для проверки: произведение собственных значений = det A).
    """
    n = len(M)
    M = [row[:] for row in M]
    det = 1
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) == 0:
            return 0
        if p != k:
            M[k], M[p] = M[p], M[k]
            det = -det
        det *= M[k][k]
        for i in range(k + 1, n):
            mu = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= mu * M[k][j]
    return det


# ----------------------------------------------------------------------
#                 QR-разложение методом Хаусхолдера
# ----------------------------------------------------------------------

def householder_matrix(v):
    """Матрица Хаусхолдера H = E - 2 v v^T / (v^T v)   (формула (1.29))."""
    n = len(v)
    vtv = sum(x * x for x in v)
    return [[(1.0 if i == j else 0.0) - 2.0 * v[i] * v[j] / vtv
             for j in range(n)] for i in range(n)]


def qr_decompose(A):
    """
    QR-разложение A = Q R с помощью преобразований Хаусхолдера.
    Возвращает (Q, R).
    """
    n = len(A)
    R = [list(map(float, row)) for row in A]   # сюда применяются H_k
    Q = identity(n)                             # накапливаем H_1 H_2 ...

    for k in range(n - 1):
        # Евклидова норма части k-го столбца, начиная с диагонали.
        norm = math.sqrt(sum(R[i][k] ** 2 for i in range(k, n)))
        if norm == 0:
            continue  # столбец уже нулевой - преобразование не нужно

        # Вектор v по формуле (1.30): первые k компонент нулевые,
        # поэтому H_k не затрагивает уже обработанные строки/столбцы.
        v = [0.0] * n
        v[k] = R[k][k] + sign(R[k][k]) * norm
        for i in range(k + 1, n):
            v[i] = R[i][k]

        H = householder_matrix(v)
        R = mat_mul(H, R)   # A_k = H_k A_{k-1}: обнуляем низ k-го столбца
        Q = mat_mul(Q, H)   # Q = H_1 H_2 ... H_k  (H симметрична и ортогональна)

    return Q, R


# ----------------------------------------------------------------------
#                            QR-алгоритм
# ----------------------------------------------------------------------

def block_eigenvalues(A, j):
    """
    Собственные значения блока 2x2, стоящего в строках/столбцах j, j+1.
    Решаем квадратное уравнение
        (a_jj - l)(a_{j+1,j+1} - l) - a_{j,j+1} a_{j+1,j} = 0,
    т.е.  l^2 - (a_jj + a_{j+1,j+1}) l + (a_jj a_{j+1,j+1} - a_{j,j+1} a_{j+1,j}) = 0.
    cmath.sqrt корректно даёт комплексный корень при отрицательном
    дискриминанте.
    """
    a11, a12 = A[j][j], A[j][j + 1]
    a21, a22 = A[j + 1][j], A[j + 1][j + 1]
    tr = a11 + a22                  # след блока
    det = a11 * a22 - a12 * a21     # определитель блока
    disc = cmath.sqrt(tr * tr - 4 * det)
    return (tr + disc) / 2, (tr - disc) / 2


def qr_algorithm(A, eps):
    """
    QR-алгоритм поиска всех собственных значений.

    Возвращает (eigenvalues, Ak, k, history):
        eigenvalues - список собственных значений (вещественных или комплексных);
        Ak          - итоговая (квази)треугольная матрица;
        k           - число итераций;
        history     - для каждой итерации список норм поддиагональных
                      частей столбцов 1..n-1 (для анализа сходимости).
    """
    n = len(A)
    Ak = [list(map(float, row)) for row in A]
    prev_block = {}   # собственные значения 2x2-блоков с прошлой итерации
    history = []

    for k in range(1, MAX_ITER + 1):
        # Одна итерация QR-алгоритма: A^(k) = Q R  ->  A^(k+1) = R Q.
        Q, R = qr_decompose(Ak)
        Ak = mat_mul(R, Q)

        # Нормы поддиагональных частей всех столбцов:
        # subs[j] = sqrt( sum_{l>j} a_lj^2 ). Для последнего столбца = 0.
        subs = [math.sqrt(sum(Ak[l][j] ** 2 for l in range(j + 1, n)))
                for j in range(n)]
        history.append(subs[:-1])

        # --- Проверка сходимости: идём по столбцам слева направо ---
        eigenvalues = []
        converged = True
        j = 0
        while j < n:
            if subs[j] <= eps:
                # Вещественное собственное значение - диагональный элемент.
                eigenvalues.append(Ak[j][j])
                j += 1
                continue

            # Столбец не "обнулился". Возможно, это комплексная пара:
            # тогда большим должен быть только a_{j+1,j}, а всё, что ниже
            # блока (j, j+1), - маленьким.
            if j + 1 < n:
                below = math.sqrt(sum(Ak[l][j] ** 2 + Ak[l][j + 1] ** 2
                                      for l in range(j + 2, n)))
                l1, l2 = block_eigenvalues(Ak, j)
                if below <= eps and abs(l1.imag) > 0:
                    # Сравниваем с собственными значениями того же блока
                    # на предыдущей итерации: |lambda^(k) - lambda^(k-1)| <= eps.
                    old = prev_block.get(j)
                    prev_block[j] = (l1, l2)
                    if old is not None and abs(l1 - old[0]) <= eps \
                            and abs(l2 - old[1]) <= eps:
                        eigenvalues.extend([l1, l2])
                        j += 2
                        continue
                else:
                    prev_block.pop(j, None)

            # Ни один из критериев не выполнен - нужны ещё итерации.
            converged = False
            break

        if converged:
            return eigenvalues, Ak, k, history

    raise RuntimeError(f"QR-алгоритм не сошёлся за {MAX_ITER} итераций")


def format_complex(z):
    """Печать числа: вещественное - как обычно, комплексное - как a +/- bi."""
    if isinstance(z, complex):
        if abs(z.imag) < 1e-12:
            return f"{z.real:.6f}"
        return f"{z.real:.6f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.6f}i"
    return f"{z:.6f}"


# ----------------------------------------------------------------------
#                            Основная программа
# ----------------------------------------------------------------------

def main():
    # Данные варианта 26.
    A = [
        [-9, -9, -3],
        [-9,  0, -2],
        [-5, -1, -4],
    ]
    eps = 0.01
    n = len(A)

    print("=" * 60)
    print("Задание 1.5. QR-разложение и QR-алгоритм (вариант 26)")
    print("=" * 60)
    print_matrix("A", A, fmt="{:6.0f}")
    print(f"Точность eps = {eps}\n")

    # --- QR-разложение исходной матрицы ---
    Q, R = qr_decompose(A)
    print("QR-разложение исходной матрицы:")
    print_matrix("Q", Q)
    print_matrix("R", R)

    # Проверки: Q R = A и Q^T Q = E.
    QR = mat_mul(Q, R)
    err_qr = max(abs(QR[i][j] - A[i][j]) for i in range(n) for j in range(n))
    QtQ = mat_mul(transpose(Q), Q)
    err_q = max(abs(QtQ[i][j] - (1.0 if i == j else 0.0))
                for i in range(n) for j in range(n))
    print(f"Проверка: max|QR - A|    = {err_qr:.2e}")
    print(f"Проверка: max|Q^T Q - E| = {err_q:.2e}\n")

    # --- QR-алгоритм ---
    eigenvalues, Ak, k, history = qr_algorithm(A, eps)
    print(f"QR-алгоритм сошёлся за {k} итераций.")
    print_matrix(f"A^({k})", Ak)

    print("Собственные значения:")
    for m, lam in enumerate(eigenvalues, 1):
        print(f"  lambda{m} = {format_complex(lam)}")
    print()

    # --- Проверка ---
    # Сумма собственных значений равна следу матрицы, а произведение -
    # её определителю (при точности eps совпадение приближённое).
    print("Проверка:")
    trace = sum(A[i][i] for i in range(n))
    s = sum(eigenvalues)
    p = 1
    for lam in eigenvalues:
        p *= lam
    print(f"  Сумма собственных значений = {format_complex(s)},"
          f" след A = {trace}")
    print(f"  Произведение собств. значений = {format_complex(p)},"
          f" det A = {determinant(A):.6f}")
    print()

    # --- Как убывают поддиагональные элементы ---
    print("Нормы поддиагональных частей столбцов по итерациям:")
    print("   k | " + " | ".join(f"{'столбец ' + str(j + 1):>14}"
                                 for j in range(n - 1)))
    for kk, subs in enumerate(history, 1):
        print(f"{kk:4d} | " + " | ".join(f"{h:14.6e}" for h in subs))


if __name__ == "__main__":
    main()
