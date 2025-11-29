#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define MODULUS 2147483647
#define MULTIPLIER 48271

// Переменная для размера матрицы
#ifndef N
// По умолчанию размер матрицы 1000
#define N 1000
#endif

// Возвращает число с равномерным распределением от 0 до 1.
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

// Измерение времени выполнения
float elapsed_msecs(struct timeval s, struct timeval f) {
    return (float) (1000.0 * (f.tv_sec - s.tv_sec) + (0.001 * (f.tv_usec - s.tv_usec)));
}

// Инициализация матрицы случайными числами
void init_array(int n, double* A, double* B) {
    time_t seed_time = time(0);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int seed = (seed_time + omp_get_thread_num()) % MODULUS;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = linear_congruential_gen(&seed) * n * n;
            B[i * n + j] = A[i * n + j];
        }
    }
}

// Последровательное LU-разложение
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

// Параллельное LU-разложение
void kernel_lu_parallel(int n, double* A) {
    // Внешний цикл по i должен оставаться последовательным из-за зависимостей данных
    for (int i = 0; i < n; i++) {
        // Вычисление нижней треугольной матрицы L (j < i)
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] /= A[j * n + j];
        }
        // Вычисление верхней треугольной матрицы U (j >= i)
        #pragma omp parallel for
        for (int j = i; j < n; j++) {
            for (int k = 0; k < i; k++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

// Сравнение результатов двух матриц
double compare_results(int n, double* A_seq, double* A_par) {
    double max_diff = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val_seq = A_seq[i * n + j];
            double val_par = A_par[i * n + j];
            if (fabs(val_seq) > 1e-10) {
                double diff = fabs(val_par / val_seq - 1);
                if (diff > max_diff)
                    max_diff = diff;
            } else if (fabs(val_par) > 1e-10) {
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
    
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            printf("Нельзя использовать размер матрицы меньше 1, используем значение по умолчанию N = %d\n", N);
            n = N;
        }
    }
    
    printf("LU-разложение - последовательная и параллельная версии\n");
    printf("Размер матрицы: %d x %d\n", n, n);
    
    int num_threads = omp_get_max_threads();
    printf("Количество потоков: %d", num_threads);
    
    // Проверяем, задано ли количество потоков через переменную окружения
    char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads != NULL) {
        printf(" (задано через OMP_NUM_THREADS=%s)", omp_threads);
    } else {
        printf(" (все доступные ядра процессора)");
    }
    printf("\n");
    printf("\n");
    
    // Выделение памяти для матриц
    double* A_seq = malloc(sizeof(double) * n * n);
    double* A_par = malloc(sizeof(double) * n * n);
    
    if (A_seq == NULL || A_par == NULL) {
        printf("Ошибка выделения памяти!\n");
        return 1;
    }
    
    // Инициализация входных данных
    printf("Инициализация матриц...\n");
    init_array(n, A_seq, A_par);
    printf("Инициализация завершена.\n\n");
    
    // Запуск последовательного варианта
    printf("Запуск последовательного варианта...\n");
    struct timeval start_seq, finish_seq;
    gettimeofday(&start_seq, 0);
    kernel_lu_sequential(n, A_seq);
    gettimeofday(&finish_seq, 0);
    float time_seq = elapsed_msecs(start_seq, finish_seq);
    printf("Время выполнения последовательного варианта: %f миллисекунд\n\n", time_seq);
    
    // Запуск параллельного варианта
    printf("Запуск параллельного варианта...\n");
    struct timeval start_par, finish_par;
    gettimeofday(&start_par, 0);
    kernel_lu_parallel(n, A_par);
    gettimeofday(&finish_par, 0);
    float time_par = elapsed_msecs(start_par, finish_par);
    printf("Время выполнения параллельного варианта: %f миллисекунд\n\n", time_par);
    
    // Сравнение результатов
    printf("Сравнение результатов...\n");
    double max_diff = compare_results(n, A_seq, A_par);
    printf("Максимальная разница: %lf процентов\n\n", max_diff * 100);
    
    // Сравнение времени выполнения
    printf("Сравнение производительности:\n");
    printf("Время выполнения последовательного варианта: %f миллисекунд\n", time_seq);
    printf("Время выполнения параллельного варианта: %f миллисекунд\n", time_par);
    if (time_seq > 0) {
        double speedup = time_seq / time_par;
        printf("Ускорение: %.2fx\n", speedup);
        printf("Эффективность: %.2f%%\n", (speedup / omp_get_max_threads()) * 100);
    }
    
    // Освобождение памяти
    free(A_seq);
    free(A_par);
    
    return 0;
}
