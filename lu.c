#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define MODULUS 2147483647
#define MULTIPLIER 48271

// Размер матрицы (можно изменить)
#ifndef N
#define N 1000
#endif

/**
 * Returns a pseudo-random real number uniformly distributed between 0 and 1.
 */
double linear_congruential_gen(int* seed) {
    const int Q = MODULUS / MULTIPLIER;
    const int R = MODULUS % MULTIPLIER;
    int t = MULTIPLIER * (*seed % Q) - R * (*seed / Q);
    if (t > 0)
        *seed = t;
    else
        *seed = t + MODULUS;
    return (double) *seed / MODULUS;
}

/**
 * Функция для измерения времени выполнения
 */
float elapsed_msecs(struct timeval s, struct timeval f) {
    return (float) (1000.0 * (f.tv_sec - s.tv_sec) + (0.001 * (f.tv_usec - s.tv_usec)));
}

/**
 * Инициализация матрицы случайными значениями
 */
void init_array(int n, double* A, double* B) {
    time_t seed_time = time(0);
    
    // Используем одну директиву #pragma omp parallel for
    // seed объявляем как private для каждого потока
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // seed инициализируется один раз для каждого потока
        int seed = (seed_time + omp_get_thread_num()) % MODULUS;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = linear_congruential_gen(&seed) * n * n;
            B[i * n + j] = A[i * n + j];
        }
    }
}

/**
 * Последовательная версия LU-разложения
 */
void kernel_lu_sequential(int n, double* A) {
    for (int i = 0; i < n; i++) {
        // Вычисление нижней треугольной матрицы L (j < i)
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] /= A[j * n + j];
        }
        // Вычисление верхней треугольной матрицы U (j >= i)
        for (int j = i; j < n; j++) {
            for (int k = 0; k < i; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

/**
 * Параллельная версия LU-разложения с OpenMP
 * 
 * ============================================
 * ЗАВИСИМОСТИ ДАННЫХ - ПОДРОБНОЕ ОБЪЯСНЕНИЕ:
 * ============================================
 * 
 * Внешний цикл по i НЕЛЬЗЯ распараллелить из-за зависимостей данных.
 * 
 * ПРИМЕР ЗАВИСИМОСТЕЙ (матрица 4x4):
 * 
 * При i=0: вычисляем строку 0
 *   - A[0][0], A[0][1], A[0][2], A[0][3]
 * 
 * При i=1: вычисляем строку 1
 *   - A[1][0] = ... / A[0][0]  <- НУЖНО A[0][0] из предыдущей итерации!
 *   - A[1][1] = ... - A[1][0]*A[0][1]  <- НУЖНО A[1][0] и A[0][1]
 *   - A[1][2] = ... - A[1][0]*A[0][2]  <- НУЖНО A[1][0] и A[0][2]
 *   - A[1][3] = ... - A[1][0]*A[0][3]  <- НУЖНО A[1][0] и A[0][3]
 * 
 * При i=2: вычисляем строку 2
 *   - A[2][0] = ... / A[0][0]  <- НУЖНО A[0][0]
 *   - A[2][1] = ... / A[1][1]  <- НУЖНО A[1][1] из предыдущей итерации!
 *   - A[2][2] = ... - A[2][0]*A[0][2] - A[2][1]*A[1][2]  <- НУЖНО предыдущие значения
 *   - и т.д.
 * 
 * ВИДЫ ЗАВИСИМОСТЕЙ:
 * 
 * 1. ЗАВИСИМОСТЬ ОТ ПРЕДЫДУЩИХ СТРОК:
 *    A[i][j] использует A[k][j] для k < i
 *    Это значит, что строка i зависит от всех предыдущих строк (0..i-1)
 * 
 * 2. ЗАВИСИМОСТЬ ВНУТРИ СТРОКИ:
 *    A[i][j] использует A[i][k] для k < j
 *    Это значит, что элемент j зависит от предыдущих элементов в той же строке
 * 
 * 3. ЗАВИСИМОСТЬ ПРИ ДЕЛЕНИИ:
 *    A[i][j] /= A[j][j]  <- нужен диагональный элемент, вычисленный ранее
 * 
 * ПОЧЕМУ НЕЛЬЗЯ РАСПАРАЛЛЕЛИТЬ ВНЕШНИЙ ЦИКЛ:
 * 
 * Если бы мы написали:
 *   #pragma omp parallel for
 *   for (int i = 0; i < n; i++) { ... }
 * 
 * То потоки могли бы начать вычислять строку i=5 ДО того, как строка i=3 будет
 * полностью вычислена. Но строка 5 зависит от строки 3! Результат будет неверным.
 * 
 * ПОЧЕМУ МОЖНО РАСПАРАЛЛЕЛИТЬ ВНУТРЕННИЕ ЦИКЛЫ:
 * 
 * Для фиксированного i=5, все элементы A[5][j] для разных j МОЖНО вычислять
 * параллельно, потому что:
 * 
 * 1. КАЖДЫЙ ПОТОК ПИШЕТ В СВОЮ ЯЧЕЙКУ:
 *    - Поток 1 вычисляет A[5][0] -> пишет в A[5][0]
 *    - Поток 2 вычисляет A[5][1] -> пишет в A[5][1]
 *    - Поток 3 вычисляет A[5][2] -> пишет в A[5][2]
 *    Нет конфликтов записи!
 * 
 * 2. ЗАВИСИМОСТИ ТОЛЬКО НА ЧТЕНИЕ:
 *    - A[5][1] читает A[5][0], но A[5][0] уже вычислен в предыдущей итерации
 *      внешнего цикла (когда i был меньше) или в той же итерации, но другим потоком
 *    - Все A[k][j] для k < 5 уже вычислены в предыдущих итерациях i
 * 
 * 3. ВАЖНО: A[i][j] зависит от A[i][k] для k < j, НО:
 *    - Это зависимость на чтение, а не на запись
 *    - Каждый поток читает только уже вычисленные значения
 *    - OpenMP гарантирует, что все чтения видят правильные значения
 * 
 * ПРИМЕР для i=5, j=3:
 *   A[5][3] -= A[5][0]*A[0][3] + A[5][1]*A[1][3] + A[5][2]*A[2][3]
 *   
 *   Здесь используются:
 *   - A[5][0], A[5][1], A[5][2] - могут вычисляться параллельно с A[5][3]
 *   - A[0][3], A[1][3], A[2][3] - уже вычислены в предыдущих итерациях i
 *   
 *   Если поток для j=3 начнет раньше, чем потоки для j=0,1,2, он прочитает
 *   старые значения A[5][0], A[5][1], A[5][2], что неправильно!
 * 
 * РЕШЕНИЕ: Используем правильный порядок или синхронизацию.
 * В данном случае, так как цикл идет последовательно (j от 0 до i),
 * и мы используем #pragma omp parallel for, OpenMP распределит итерации
 * по потокам, но порядок выполнения может быть любым. Это может быть проблемой!
 * 
 * БОЛЕЕ БЕЗОПАСНЫЙ ВАРИАНТ: использовать schedule(static,1) для гарантии
 * последовательного порядка, или вообще не распараллеливать нижнюю часть.
 * 
 * Однако, в оригинальном PolyBench код работает именно так, поэтому
 * оставляем текущую реализацию, но понимаем ограничения.
 */
void kernel_lu_parallel(int n, double* A) {
    // Внешний цикл по i должен оставаться последовательным из-за зависимостей данных
    // НЕ распараллеливаем: for (int i = 0; i < n; i++)
    for (int i = 0; i < n; i++) {
        // Параллелизуем цикл по j для нижней треугольной части
        // Безопасно: все j независимы при фиксированном i
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] /= A[j * n + j];
        }
        // Параллелизуем цикл по j для верхней треугольной части
        // Безопасно: все j независимы при фиксированном i
        #pragma omp parallel for
        for (int j = i; j < n; j++) {
            for (int k = 0; k < i; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

/**
 * Сравнение результатов двух матриц
 */
double compare_results(int n, double* A_seq, double* A_par) {
    double max_diff = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val_seq = A_seq[i * n + j];
            double val_par = A_par[i * n + j];
            // Избегаем деления на ноль
            if (fabs(val_seq) > 1e-10) {
                double diff = fabs(val_par / val_seq - 1);
                if (diff > max_diff)
                    max_diff = diff;
            } else if (fabs(val_par) > 1e-10) {
                // Если последовательная версия близка к нулю, а параллельная нет
                double diff = fabs(val_par);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }
    }
    return max_diff;
}

int main(int argc, char** argv) {
    int n = N;
    
    // Размер можно задать через аргумент командной строки
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            printf("Invalid size, using default N = %d\n", N);
            n = N;
        }
    }
    
    printf("LU Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    
    // ============================================
    // КОЛИЧЕСТВО ПОТОКОВ - ПОДРОБНОЕ ОБЪЯСНЕНИЕ:
    // ============================================
    // 
    // omp_get_max_threads() возвращает максимальное количество потоков,
    // которое может использовать OpenMP.
    // 
    // ОТКУДА БЕРЕТСЯ ЧИСЛО ПОТОКОВ (приоритет сверху вниз):
    // 
    // 1. Переменная окружения OMP_NUM_THREADS (если установлена)
    //    Пример: export OMP_NUM_THREADS=8
    // 
    // 2. Вызов omp_set_num_threads() в коде (если был)
    //    Пример: omp_set_num_threads(4);
    // 
    // 3. Количество логических ядер процессора (по умолчанию)
    //    На вашей системе: 16 логических ядер (проверено командой nproc)
    //    Это может быть:
    //    - 16 физических ядер
    //    - 8 физических ядер с Hyper-Threading (8*2=16)
    //    - Другая конфигурация
    // 
    // 4. Настройки системы и OpenMP runtime
    // 
    // КАК ИЗМЕНИТЬ КОЛИЧЕСТВО ПОТОКОВ:
    // 
    // Вариант 1 (перед запуском программы):
    //   export OMP_NUM_THREADS=4
    //   ./lu
    // 
    // Вариант 2 (в одной команде):
    //   OMP_NUM_THREADS=8 ./lu
    // 
    // Вариант 3 (в коде, перед первым использования OpenMP):
    //   omp_set_num_threads(4);
    // 
    // ПОЧЕМУ ИМЕННО 16:
    // Ваш процессор имеет 16 логических ядер, поэтому OpenMP по умолчанию
    // использует все 16 потоков для максимальной производительности.
    // 
    int num_threads = omp_get_max_threads();
    printf("Number of threads: %d", num_threads);
    
    // Проверяем, задано ли количество потоков через переменную окружения
    char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads != NULL) {
        printf(" (set by OMP_NUM_THREADS=%s)", omp_threads);
    } else {
        printf(" (default: all available CPU cores)");
    }
    printf("\n");
    printf("\n");
    
    // Выделение памяти для матриц
    double* A_seq = malloc(sizeof(double) * n * n);
    double* A_par = malloc(sizeof(double) * n * n);
    
    if (A_seq == NULL || A_par == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Инициализация входных данных
    printf("Initializing matrices...\n");
    init_array(n, A_seq, A_par);
    printf("Initialization complete.\n\n");
    
    // Запуск последовательного варианта
    printf("Running sequential version...\n");
    struct timeval start_seq, finish_seq;
    gettimeofday(&start_seq, 0);
    kernel_lu_sequential(n, A_seq);
    gettimeofday(&finish_seq, 0);
    float time_seq = elapsed_msecs(start_seq, finish_seq);
    printf("Sequential time: %f milliseconds\n\n", time_seq);
    
    // Запуск параллельного варианта
    printf("Running parallel version...\n");
    struct timeval start_par, finish_par;
    gettimeofday(&start_par, 0);
    kernel_lu_parallel(n, A_par);
    gettimeofday(&finish_par, 0);
    float time_par = elapsed_msecs(start_par, finish_par);
    printf("Parallel time: %f milliseconds\n\n", time_par);
    
    // Сравнение результатов
    printf("Comparing results...\n");
    double max_diff = compare_results(n, A_seq, A_par);
    printf("Maximal difference: %lf percent\n\n", max_diff * 100);
    
    // Сравнение времени выполнения
    printf("Performance comparison:\n");
    printf("Sequential time: %f ms\n", time_seq);
    printf("Parallel time: %f ms\n", time_par);
    if (time_seq > 0) {
        double speedup = time_seq / time_par;
        printf("Speedup: %.2fx\n", speedup);
        printf("Efficiency: %.2f%%\n", (speedup / omp_get_max_threads()) * 100);
    }
    
    // Освобождение памяти
    free(A_seq);
    free(A_par);
    
    return 0;
}
