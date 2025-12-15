#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

// "Стол пуст?" (1 = пуст, 0 = на столе лежит пара ингредиентов)
sem_t table_semaphore;

// Семафоры курильщиков: бармен будит конкретного курильщика
sem_t smoker_semaphores[3];

// Мьютекс для rand() и общих счетчиков
pthread_mutex_t mutex;

struct smoker
{
    int id;             
    char* name;         
    char* component;    
    int smoke_times;   
};

struct smoker smokers[3];

pthread_t smoker_threads[3];
pthread_t barman_thread;

// Курильщик ждёт, пока бармен положит на стол пару ИМЕННО для него
void smoker_wait(struct smoker *s)
{
    fprintf(stdout, "%s (%s) ждёт недостающие компоненты\n", s->name, s->component); fflush(stdout);
    sem_wait(&smoker_semaphores[s->id]);
}

// Курильщик "курит"
void smoker_smoke(struct smoker *s)
{
    pthread_mutex_lock(&mutex);
    int t = rand() % 2 + 1; 
    pthread_mutex_unlock(&mutex);

    fprintf(stdout, "%s курит %d сек\n", s->name, t); fflush(stdout);
    sleep(t);
}

// Жизненный цикл курильщика
void *smoker_lifecycle(void *arg)
{
    struct smoker *s = (struct smoker *)arg;

    while (s->smoke_times < 3)
    {
        smoker_wait(s);

        // Курильщик забрал два ингредиента со стола -> стол снова пуст
        fprintf(stdout, "%s забрал ингредиенты со стола и скрутил сигарету\n", s->name); fflush(stdout);
        sem_post(&table_semaphore);

        s->smoke_times++;
        smoker_smoke(s);
    }

    fprintf(stdout, "%s вышел, покурив %d раз\n", s->name, s->smoke_times); fflush(stdout);

    return NULL;
}

// Жизненный цикл бармена
void *barman_lifecycle(void *arg)
{
    (void)arg;

    // каждый курит 3 раза
    for (int i = 0; i < 9; i++)
    {
        sem_wait(&table_semaphore);

        int a, b, c;
        while (1)
        {
            pthread_mutex_lock(&mutex);
            a = rand() % 3;
            b = rand() % 3;
            while (b == a) { 
                b = rand() % 3;
            }
            c = 3 - a - b; // третий курильщик
            int c_done = (smokers[c].smoke_times >= 3);
            pthread_mutex_unlock(&mutex);
            
            if (c_done == 0) {
                break;
            }
        }

        fprintf(stdout, "Бармен положил на стол: %s, %s -> %s (%s) курит\n",
                smokers[a].component, smokers[b].component,
                smokers[c].name, smokers[c].component);
        fflush(stdout);

        // Будим нужного курильщика. Стол остаётся "занятым" пока он не заберёт.
        sem_post(&smoker_semaphores[c]);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    int res = 0;

    res = pthread_mutex_init(&mutex, NULL);
    if (res != 0)
    {
        fprintf(stderr, "pthread_mutex_init for mutex\n");
        exit(EXIT_FAILURE);
    }

    // Стол изначально пуст
    res = sem_init(&table_semaphore, 0, 1);
    if (res != 0)
    {
        fprintf(stderr, "sem_init for table_semaphore\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 3; i++)
    {
        res = sem_init(&smoker_semaphores[i], 0, 0);
        if (res != 0)
        {
            fprintf(stderr, "sem_init for smoker %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Инициализация курильщиков
    smokers[0].id = 0;
    smokers[0].name = "Курильщик 1";
    smokers[0].component = "табак";
    smokers[0].smoke_times = 0;

    smokers[1].id = 1;
    smokers[1].name = "Курильщик 2";
    smokers[1].component = "бумага";
    smokers[1].smoke_times = 0;

    smokers[2].id = 2;
    smokers[2].name = "Курильщик 3";
    smokers[2].component = "спички";
    smokers[2].smoke_times = 0;
    // Потоки курильщиков
    for (int i = 0; i < 3; i++)
    {
        res = pthread_create(&smoker_threads[i], NULL, smoker_lifecycle, &smokers[i]);
        if (res != 0)
        {
            fprintf(stderr, "pthread_create for smoker %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Поток бармена
    res = pthread_create(&barman_thread, NULL, barman_lifecycle, NULL);
    if (res != 0)
    {
        fprintf(stderr, "pthread_create for barman\n");
        exit(EXIT_FAILURE);
    }

    // Ждём завершения
    pthread_join(barman_thread, NULL);
    for (int i = 0; i < 3; i++)
        pthread_join(smoker_threads[i], NULL);

    // Очистка
    pthread_mutex_destroy(&mutex);
    for (int i = 0; i < 3; i++)
        sem_destroy(&smoker_semaphores[i]);
    sem_destroy(&table_semaphore);

    return EXIT_SUCCESS;
}
