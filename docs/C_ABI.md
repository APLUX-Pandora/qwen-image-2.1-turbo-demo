# C ABI

`sdk/include/qwen_image.h` · `sdk/lib/libqwen_image.so` · Generate one RGB image from a prompt.

## Build and link

From the Demo root, compile against the shipped header and native library, which includes the GGML runtime:

```bash
cc -std=c99 -I sdk/include your_app.c -L sdk/lib -lqwen_image -Wl,-rpath,'$ORIGIN/sdk/lib' -o your_app
```

Before running the program, see [Model setup](MODEL_SETUP.md#targets) for the required bundled Hexagon DSP library (v73 or v75) and FastRPC search path.

## Lifecycle

```text
qwi_create → qwi_generate → qwi_destroy
```

If creation fails, read the same-thread create diagnostic before retrying. For a generation error, discard the output; correct input errors and recreate the engine after a backend or internal failure. On shutdown, stop accepting work, request cancellation, wait for callers and callbacks, then destroy the engine once.

## Functions

| Function | Purpose | Returns |
| --- | --- | --- |
| `qwi_options qwi_default_options(void)` | returns a value initialized to threads4/verbose0 | Options struct |
| `qwi_engine* qwi_create(const char* model_dir,const qwi_options* opt)` | `model_dir` is the directory containing `models.json` and the files for this device; opt may be NULL; returns owned handle or NULL; NULL failure diagnostic through same-thread create_error. It initializes native tokenizer/context. | Owned handle or `NULL` |
| `void qwi_destroy(qwi_engine* e)` | NULL allowed; destroys a live owned handle, model/context and cached FastRPC sessions; wait for every caller/callback first. Destroying a previously freed pointer is invalid. | Void |
| `int qwi_image_size(int resolution,qwi_aspect aspect,int* width,int* height)` | output pointers non-NULL; tiers512/1024/1536/2048; valid aspect0..6; pure preset arithmetic, returns0 or ARG. Device support still comes from info. | Status code |
| `int qwi_generate(qwi_engine* e,const qwi_request* req,qwi_progress_cb progress,void* user,uint8_t* out_rgb,size_t out_size)` | valid handle/request/output, capacity at least width*height*3; progress may be NULL. Blocks; returns0 on a complete image or negative status. One generation per handle. | Status code |
| `void qwi_cancel(qwi_engine* e)` | NULL/idle safe no-op; may overlap generation from an ordinary control thread; request only, not a join. | Void |
| `int qwi_get_info(const qwi_engine* e,qwi_info* info)` | non-NULL writable output; returns0 or ARG. NULL engine returns static profile with arch0. Do not query a live engine concurrently with generation unless its contract allows it; only cancel/count_tokens are the supported overlap operations. | Status code |
| `int qwi_count_tokens(qwi_engine* e,const char* prompt)` | non-NULL handle and UTF8 C string; count>=0, or negative; allowed concurrently with generation and from callback. | Token count or error code |
| `int qwi_get_stats(const qwi_engine* e,qwi_stats* stats)` | valid non-NULL handle/output, called after generation;0 or ARG | Status code |
| `const char* qwi_last_error(const qwi_engine* e)` | borrowed error text, valid until next call changes it or destroy; NULL handle returns empty static text; do not free | Borrowed string |
| `const char* qwi_create_error(void)` | borrowed thread-local last-create diagnostic, copy before next create/thread exit; do not free | Borrowed string |

## Error codes

| Code | Meaning | What the caller should do |
| --- | --- | --- |
| 0 | Success | Use the complete output. |
| −1 | Invalid argument | Correct the request or buffer. |
| −2 | Model error | Check the model folder and resources. Recreate the engine after a native model failure. |
| −3 | NPU error | Check the device runtime and recreate the engine. |
| −4 | Invalid state | Check ownership and the call sequence. |
| −5 | Cancelled | Discard the output. |
| −6 | Unsupported | Use a supported request. |
| −7 | Internal error | Read the diagnostic and recreate the engine. |

## Threads and ownership

During generation only `qwi_cancel` and `qwi_count_tokens` may run concurrently or reenter from a callback. Serialize other operations. Use an ordinary control thread for signal coordination, wait for native calls to finish and retire borrowed handles before destroy.

`qwi_engine` is opaque and owned by a successful create caller. Only one engine per process. If callers create engines concurrently, one reserves the slot and the others receive a same-thread diagnostic. `qwi_options` has C-int n_threads (default4; <=0 falls back to4) and verbose (default0). Initialize it with qwi_default_options or pass NULL for defaults.

`qwi_request` contains borrowed `const char* prompt` (nonempty NUL-terminated UTF8), int width/height in pixels, and int64_t seed >=0. The request/prompt must remain valid through synchronous generate/callbacks. Use recommended32-aligned image presets. Raw native validation accepts positive16 alignment, but only supported size-class pixel/token budgets are accepted; that does not certify arbitrary aspect ratios on every target. SDK input weights remain the separate three-file manifest package.

Output RGB is caller-allocated uint8 bytes, width*height*3, row-major, top row first, no header/stride padding. Keep it alive through generate. On success it contains a complete image. On error/cancel discard it. Native library never frees caller output. C frees with its allocator; C++ can use vector<uint8_t>. No STL object, exception or allocator ownership crosses the C ABI.

`qwi_info` fields: borrowed const char* version, int n_steps, int n_resolutions, int resolutions[4], int hexagon_arch, int max_prompt_tokens[4]. Only the first n_resolutions entries apply; token limits correspond to each resolutions entry. Copy borrowed strings if retaining beyond library lifetime; never free them.

`qwi_stats`: double encode_ms, denoise_ms, decode_ms, total_ms for the last generation. All units are milliseconds (Python converts to seconds). They may include stage loading/rebuilding, not HTTP/browser time.

`qwi_aspect` values0..6: QWI_ASPECT_1_1,4_3,3_4,3_2,2_3,16_9,9_16. `qwi_phase` values0..2: ENCODE,DENOISE,DECODE. The callback `void(qwi_phase,int step,int n_steps,double elapsed_s,void* user)` runs synchronously on the generating thread; user is a borrowed pointer valid throughout the call. Encode starts0/1; denoise starts0/8 then reports steps1..8; decode starts0 and may report tile completion. elapsed_s is seconds, unlike stats. Do not throw C++ exceptions across a C callback; capture an error in application state, cancel and report it after generate returns.

## Example

A complete program is in [examples/c/generate.c](../examples/c/generate.c).
