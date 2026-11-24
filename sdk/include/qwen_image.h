/*
 * qwen_image.h - C API of libqwen_image.so: Qwen-Image-2.1-Turbo text-to-image on the Hexagon NPU.
 *
 * Call sequence
 *   qwi_options o = qwi_default_options();
 *   qwi_engine *e = qwi_create("models", &o);          // loads the model files, opens the NPU sessions
 *   int w, h;  qwi_image_size(1024, QWI_ASPECT_16_9, &w, &h);
 *   qwi_request r = { "a red fox in the snow", w, h, 42 };
 *   uint8_t *rgb = malloc((size_t) w * h * 3);
 *   qwi_generate(e, &r, on_progress, user, rgb, (size_t) w * h * 3);
 *   qwi_destroy(e);
 *
 * What the library does for the caller: the Turbo 8-step schedule with CFG 1 (the checkpoint's own sampling, not
 * configurable), the placement of the weights on the NPU, text encoding, denoising and VAE decoding.
 * Left to the caller: where prompts come from, random seeds, queueing of requests, image file formats.
 *
 * Threading: one engine runs one generation at a time; calls on the same engine must not overlap, except
 * qwi_cancel(), which may be called from any thread while qwi_generate() runs. The progress callback runs on the
 * thread that called qwi_generate(). Token counting may also overlap generation. Destruction must wait for all
 * callers and callbacks to finish. The C API is not async-signal-safe: coordinate stop/cancel/join outside a signal
 * handler, then destroy. Only one engine per process (the NPU memory is sized for one).
 *
 * Determinism: the same request gives the same image up to small run-to-run differences of the NPU kernels
 * (mean absolute pixel difference within 2/255).
 *
 * Memory layout: the weights do not all fit the NPU at once; the text encoder and the diffusion model are loaded
 * from disk when their stage runs (part of the encode and the first denoising step). The layout depends on the size
 * class of the image (512, 1024 or 1536 tier); a change of size class rebuilds it (under a second).
 *
 * The library is NPU-architecture neutral: the Hexagon version is detected at run time and the matching
 * libggml-htp-v<NN>.so is loaded through FastRPC. Configure ADSP_LIBRARY_PATH to use a custom search path.
 * By default the SDK uses the matching DSP library next to libqwen_image.so, or the system DSP search path.
 *
 * Output: the engine logs to stderr. Vendor runtime messages that would go to stdout are routed to stderr during
 * qwi_create, qwi_generate and qwi_destroy, so the caller's stdout carries only its own output (set
 * QWI_KEEP_STDOUT=1 to disable).
 */
#ifndef QWEN_IMAGE_H
#define QWEN_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define QWI_API __declspec(dllexport)
#else
#define QWI_API __attribute__((visibility("default")))
#endif

#define QWI_VERSION "1.0.0"

/* Return codes: >= 0 success, < 0 error. */
#define QWI_OK          0
#define QWI_ERR_ARG    -1 /* invalid argument (NULL pointer, unsupported size, buffer too small, empty prompt) */
#define QWI_ERR_MODEL  -2 /* model file missing or unreadable */
#define QWI_ERR_NPU    -3 /* NPU / FastRPC failure */
#define QWI_ERR_STATE  -4 /* call not allowed now (e.g. qwi_generate while another generation runs) */
#define QWI_ERR_CANCEL -5 /* the generation was cancelled with qwi_cancel() */
#define QWI_ERR_UNSUPPORTED -6 /* valid size that this device cannot generate (e.g. the 2048 tier on QCS8550) */
#define QWI_ERR_INTERNAL -7

typedef struct qwi_engine qwi_engine;

typedef struct {
    int n_threads; /* CPU threads for the work that stays on the CPU; 0 = default (4) */
    int verbose;   /* 0 = warnings and errors on stderr, 1 = engine info log on stderr */
} qwi_options;

/* Aspect ratios of the model card, as in the official Hugging Face demo. */
typedef enum {
    QWI_ASPECT_1_1 = 0,
    QWI_ASPECT_4_3,
    QWI_ASPECT_3_4,
    QWI_ASPECT_3_2,
    QWI_ASPECT_2_3,
    QWI_ASPECT_16_9,
    QWI_ASPECT_9_16,
} qwi_aspect;

typedef struct {
    const char *prompt; /* UTF-8, required */
    int width;          /* pixels, multiple of 32; use qwi_image_size() */
    int height;
    int64_t seed;       /* >= 0; the caller picks a random one if it wants randomness */
} qwi_request;

/* Generation phases reported to the progress callback:
 *   QWI_PHASE_ENCODE   once at the start: step 0, n_steps 1 (text encoder; loads its weights)
 *   QWI_PHASE_DENOISE  after each finished step: step 1..8, n_steps 8 (step 0 when the phase starts)
 *   QWI_PHASE_DECODE   step 0 when the VAE starts; with a tiled decode (above the 512 tier) also after each tile:
 *                      step = tiles done, n_steps = number of tiles */
typedef enum {
    QWI_PHASE_ENCODE = 0,
    QWI_PHASE_DENOISE,
    QWI_PHASE_DECODE,
} qwi_phase;

typedef void (*qwi_progress_cb)(qwi_phase phase, int step, int n_steps, double elapsed_s, void *user);

typedef struct {
    const char *version;      /* QWI_VERSION */
    int n_steps;              /* 8 */
    int n_resolutions;        /* entries in resolutions[] */
    int resolutions[4];       /* resolution tiers this device can generate (QCS8550: 512, 1024, 1536) */
    int hexagon_arch;         /* detected HTP version, e.g. 73 */
    int max_prompt_tokens[4]; /* longest prompt per tier in resolutions[], in tokens (see qwi_count_tokens) */
} qwi_info;

typedef struct {
    double encode_ms;  /* last generation: text encoder */
    double denoise_ms; /* all denoising steps */
    double decode_ms;  /* VAE */
    double total_ms;
} qwi_stats;

QWI_API qwi_options qwi_default_options(void);

/* model_dir holds models.json, which lists the supported device targets and model files.
 * The device needs a matching target. Returns NULL on failure, see qwi_create_error().
 * Concurrent callers compete for the single engine slot; the slot is released after destroy or a failed create. */
QWI_API qwi_engine *qwi_create(const char *model_dir, const qwi_options *opt);
QWI_API void qwi_destroy(qwi_engine *e);

/* Output size for a resolution tier (512, 1024, 1536, 2048) and aspect ratio: the model card's 2048-base presets
 * scaled to the tier and rounded to multiples of 32, as in the official demo (1024 + 16:9 -> 1376 x 768).
 * Pure arithmetic; whether the device can generate the size is listed in qwi_info.resolutions.
 * Returns QWI_ERR_ARG for an unknown tier. */
QWI_API int qwi_image_size(int resolution, qwi_aspect aspect, int *width, int *height);

/* Generate one image. out_rgb receives width * height * 3 bytes, RGB8, row-major, top row first.
 * Blocks until the image is done (from under a minute at 512 to several minutes at 1536). progress may be NULL.
 * Returns QWI_ERR_UNSUPPORTED for a size above the largest tier in qwi_info.resolutions. */
QWI_API int qwi_generate(qwi_engine *e, const qwi_request *req, qwi_progress_cb progress, void *user,
                         uint8_t *out_rgb, size_t out_size);

/* Ask a running qwi_generate() to stop at the next step; it then returns QWI_ERR_CANCEL. Thread-safe. */
QWI_API void qwi_cancel(qwi_engine *e);

QWI_API int qwi_get_info(const qwi_engine *e, qwi_info *info);

/* Number of tokens of a prompt, as the text encoder sees it. Longer prompts take longer and need more NPU memory;
 * qwi_generate refuses a prompt above qwi_info.max_prompt_tokens of the requested size's tier with QWI_ERR_ARG.
 * May be called from any thread, also while qwi_generate runs. Returns the count (>= 0),
 * QWI_ERR_ARG for an invalid argument, or QWI_ERR_INTERNAL if tokenization fails. */
QWI_API int qwi_count_tokens(qwi_engine *e, const char *prompt);
QWI_API int qwi_get_stats(const qwi_engine *e, qwi_stats *stats);

/* Error message of the last failed call on e (valid until the next call on e). */
QWI_API const char *qwi_last_error(const qwi_engine *e);
/* Error message of the last failed qwi_create() on this thread. */
QWI_API const char *qwi_create_error(void);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_IMAGE_H */
