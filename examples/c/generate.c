/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include "qwen_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <errno.h>

typedef struct {
    sigset_t signals;
    pthread_mutex_t lock;
    qwi_engine *engine;
    atomic_int first_signal, done;
} stop_control;

static void *receive_stop(void *arg) {
    stop_control *s = arg;
    while (!atomic_load(&s->done)) {
        struct timespec timeout = {0, 100000000};
        int sig = sigtimedwait(&s->signals, NULL, &timeout);
        if (sig == SIGTERM || sig == SIGINT) {
            int expected = 0;
            atomic_compare_exchange_strong(&s->first_signal, &expected, sig);
            pthread_mutex_lock(&s->lock);
            if (s->engine) qwi_cancel(s->engine);
            pthread_mutex_unlock(&s->lock);
        }
    }
    return NULL;
}
static void progress(qwi_phase phase, int step, int total, double elapsed, void *arg) {
    (void)phase; (void)step; (void)total; (void)elapsed;
    stop_control *s = arg;
    if (atomic_load(&s->first_signal)) qwi_cancel(s->engine);
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL_ROOT OUTPUT.ppm\n", argv[0]); return 2; }
    stop_control stop = {0};
    atomic_init(&stop.first_signal, 0); atomic_init(&stop.done, 0);
    if (pthread_mutex_init(&stop.lock, NULL)) return 1;
    sigemptyset(&stop.signals); sigaddset(&stop.signals, SIGTERM); sigaddset(&stop.signals, SIGINT);
    sigset_t previous;
    if (pthread_sigmask(SIG_BLOCK, &stop.signals, &previous)) { pthread_mutex_destroy(&stop.lock); return 1; }
    pthread_t receiver;
    if (pthread_create(&receiver, NULL, receive_stop, &stop)) {
        pthread_sigmask(SIG_SETMASK, &previous, NULL); pthread_mutex_destroy(&stop.lock); return 1;
    }
    int status = 1;
    FILE *file = NULL;
    unsigned char *rgb = NULL;
    qwi_engine *engine = qwi_create(argv[1], NULL);
    if (!engine) { fprintf(stderr, "%s\n", qwi_create_error()); goto done; }
    pthread_mutex_lock(&stop.lock); stop.engine = engine; pthread_mutex_unlock(&stop.lock);
    if (atomic_load(&stop.first_signal)) goto done;
    const int width = 512, height = 512;
    const size_t size = (size_t)width * height * 3;
    qwi_request request = {"A red fox in the snow, watercolor painting.", width, height, 42};
    rgb = malloc(size);
    if (!rgb) { fprintf(stderr, "output allocation failed\n"); goto done; }
    if (qwi_generate(engine, &request, progress, &stop, rgb, size) < 0) {
        if (!atomic_load(&stop.first_signal)) fprintf(stderr, "%s\n", qwi_last_error(engine));
        goto done;
    }
    if (atomic_load(&stop.first_signal)) goto done;
    file = fopen(argv[2], "wb");
    if (!file) { perror(argv[2]); goto done; }
    if (fprintf(file, "P6\n%d %d\n255\n", width, height) < 0 || fwrite(rgb, 1, size, file) != size) {
        fprintf(stderr, "output write failed\n"); goto done;
    }
    if (fclose(file) != 0) { file = NULL; perror("close output"); goto done; }
    file = NULL; status = 0;
done:
    if (file) fclose(file);
    free(rgb);
    /* No generation/callback remains. Retire the cancel caller before destroy. */
    pthread_mutex_lock(&stop.lock); stop.engine = NULL; pthread_mutex_unlock(&stop.lock);
    atomic_store(&stop.done, 1); pthread_join(receiver, NULL);
    qwi_destroy(engine);
    int received = atomic_load(&stop.first_signal);
    pthread_mutex_destroy(&stop.lock);
    pthread_sigmask(SIG_SETMASK, &previous, NULL);
    return received ? 128 + received : status;
}
