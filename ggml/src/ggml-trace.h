#ifndef GGML_TRACE_H
#define GGML_TRACE_H

#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void trace_init(const char* fname);
void trace_event(const char* name, int begin);
void trace_finalize(void);
void trace_flush_thread(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_TRACE_H
