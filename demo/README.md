# Web demo

Generate an image from a text prompt using the same installed SDK as the examples.

## Start

Complete [installation](../README.md#1-install) and [model setup](../docs/MODEL_SETUP.md), then run:

```bash
bash demo/run.sh --port 8810
```

Open `http://<device-ip>:8810/`. The service listens on all IPv4 interfaces and has no login. Use it on a trusted network. Generated images go to `out/web/`. Set `QWEN_IMAGE_MODELS` to use a different folder containing the model manifest.

## The page

Enter a prompt, select a resolution and aspect ratio, then select **Generate**. The token counter shows the prompt limit at the chosen size. Set a seed to reuse it, or select **New seed each run**. Use **Cancel** to stop or **Download PNG** to save a completed image. Concurrent browser requests queue for one pipeline.

## Connection and model recovery

Use **Retry connection** after a network error. When an open page reconnects, it queries the current job without submitting it again. After a native failure, use **Reload model** once active and queued work has finished. Stop the service with `Ctrl+C`.
