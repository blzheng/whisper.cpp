#include "ggml-trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define EVENT_NAME_MAX_LEN 64
#define INITIAL_CAPACITY 1024
#define TLS_INITIAL_CAPACITY 256

typedef struct {
    char ph; // 'B' or 'E'
    char name[EVENT_NAME_MAX_LEN];
    long long ts;
} TraceEvent;

typedef struct {
    TraceEvent* events;
    size_t size;
    size_t capacity;
    pthread_mutex_t lock;
} EventBuffer;

typedef struct {
    TraceEvent* events;
    size_t size;
    size_t capacity;
} ThreadLocalBuffer;

static FILE* trace_file = NULL;
static int trace_initialized = 0;

static EventBuffer buffer = {NULL, 0, 0, PTHREAD_MUTEX_INITIALIZER};
static pthread_key_t tls_key;
static pthread_once_t tls_once = PTHREAD_ONCE_INIT;

// === Buffer Management ===
static void buffer_init(EventBuffer* buf) {
    buf->capacity = INITIAL_CAPACITY;
    buf->size = 0;
    buf->events = (TraceEvent*)malloc(buf->capacity * sizeof(TraceEvent));
    if (!buf->events) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    pthread_mutex_init(&buf->lock, NULL);
}

static void buffer_append(EventBuffer* buf, const char* name, char ph, long long ts) {
    if (buf->size >= buf->capacity) {
        size_t new_capacity = buf->capacity + buf->capacity / 2; // 1.5x
        TraceEvent* new_events = (TraceEvent*)realloc(buf->events, new_capacity * sizeof(TraceEvent));
        if (!new_events) {
            fprintf(stderr, "realloc failed\n");
            exit(1);
        }
        buf->events = new_events;
        buf->capacity = new_capacity;
    }

    TraceEvent* ev = &buf->events[buf->size++];
    ev->ph = ph;
    size_t len = strnlen(name, EVENT_NAME_MAX_LEN - 1);
    memcpy(ev->name, name, len);
    ev->name[len] = '\0';
    ev->ts = ts;
}

static void buffer_free(EventBuffer* buf) {
    if (!buf || !buf->events) return;
    free(buf->events);
    buf->events = NULL;
    buf->size = 0;
    buf->capacity = 0;
    pthread_mutex_destroy(&buf->lock);
}

// === Thread-local Buffer ===
static void tls_destructor(void* ptr) {
    ThreadLocalBuffer* tls = (ThreadLocalBuffer*)ptr;
    if (!tls || !tls->events || tls->size == 0) {
        free(tls);
        return;
    }

    pthread_mutex_lock(&buffer.lock);
    for (size_t i = 0; i < tls->size; i++) {
        TraceEvent* ev = &tls->events[i];
        buffer_append(&buffer, ev->name, ev->ph, ev->ts);
    }
    pthread_mutex_unlock(&buffer.lock);

    free(tls->events);
    free(tls);
}

static void make_tls_key() {
    pthread_key_create(&tls_key, tls_destructor);
}

static ThreadLocalBuffer* get_tls_buffer() {
    pthread_once(&tls_once, make_tls_key);

    ThreadLocalBuffer* tls = (ThreadLocalBuffer*)pthread_getspecific(tls_key);
    if (!tls) {
        tls = (ThreadLocalBuffer*)malloc(sizeof(ThreadLocalBuffer));
        if (!tls) {
            fprintf(stderr, "malloc tls buffer failed\n");
            exit(1);
        }
        tls->capacity = TLS_INITIAL_CAPACITY;
        tls->size = 0;
        tls->events = (TraceEvent*)malloc(tls->capacity * sizeof(TraceEvent));
        if (!tls->events) {
            fprintf(stderr, "malloc tls events failed\n");
            free(tls);
            exit(1);
        }
        pthread_setspecific(tls_key, tls);
    }
    return tls;
}

static void buffer_append_tls(const char* name, char ph, long long ts) {
    ThreadLocalBuffer* tls = get_tls_buffer();

    if (tls->size >= tls->capacity) {
        size_t new_capacity = tls->capacity * 2;
        TraceEvent* new_events = (TraceEvent*)realloc(tls->events, new_capacity * sizeof(TraceEvent));
        if (!new_events) {
            fprintf(stderr, "tls realloc failed\n");
            exit(1);
        }
        tls->events = new_events;
        tls->capacity = new_capacity;
    }

    TraceEvent* ev = &tls->events[tls->size++];
    ev->ph = ph;
    size_t len = strnlen(name, EVENT_NAME_MAX_LEN - 1);
    memcpy(ev->name, name, len);
    ev->name[len] = '\0';
    ev->ts = ts;
}

// === Public API ===
void trace_init(const char* fname) {
    if (trace_initialized) return;

    trace_file = fopen(fname, "w");
    if (!trace_file) {
        fprintf(stderr, "Failed to open trace file %s\n", fname);
        exit(1);
    }

    fputs("[\n", trace_file);
    buffer_init(&buffer);
    pthread_once(&tls_once, make_tls_key);
    trace_initialized = 1;
}

void trace_event(const char* name, int begin) {
    if (!trace_initialized || !trace_file) {
        fprintf(stderr, "trace_event called before trace_init\n");
        return;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long us = ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;

    buffer_append_tls(name, begin ? 'B' : 'E', us);
}

void trace_flush_thread() {
    ThreadLocalBuffer* tls = (ThreadLocalBuffer*)pthread_getspecific(tls_key);
    if (!tls || !tls->events || tls->size == 0) return;

    pthread_mutex_lock(&buffer.lock);
    for (size_t i = 0; i < tls->size; i++) {
        TraceEvent* ev = &tls->events[i];
        buffer_append(&buffer, ev->name, ev->ph, ev->ts);
    }
    pthread_mutex_unlock(&buffer.lock);

    tls->size = 0;
}

void trace_finalize() {
    if (!trace_initialized) return;

    trace_flush_thread(); // flush main thread

    pthread_mutex_lock(&buffer.lock);
    for (size_t i = 0; i < buffer.size; i++) {
        TraceEvent* ev = &buffer.events[i];
        fprintf(trace_file,
                "{\"cat\":\"trace\",\"ph\":\"%c\",\"name\":\"%s\",\"ts\":%lld,\"pid\":0,\"tid\":0}",
                ev->ph, ev->name, ev->ts);
        if (i + 1 < buffer.size)
            fputs(",\n", trace_file);
        else
            fputc('\n', trace_file);
    }
    pthread_mutex_unlock(&buffer.lock);

    fputs("]\n", trace_file);
    fclose(trace_file);
    trace_file = NULL;

    buffer_free(&buffer);
    trace_initialized = 0;
}
