"""Generate one image from the Demo root with the installed SDK."""

from pandora_qwen_image_2_1_turbo import GenerationCancelledError, ImageGenerationPipeline

from _lifecycle import StopControl

PROMPT = (
    "A small orange cat sitting on a wooden windowsill beside a white ceramic "
    "cup and a leafy green plant. Warm afternoon sunlight streams through the "
    "window, casting soft shadows across the wood. Outside, a quiet garden is "
    "gently blurred in the background. Watercolor illustration with delicate "
    "brushwork, subtle paper texture, warm natural colors, and a peaceful "
    "atmosphere."
)


def main():
    with StopControl() as control:
        with ImageGenerationPipeline.from_pretrained("models") as pipe:
            control.publish(pipe)
            if control.stopped.is_set():
                return control.exit_code

            def on_progress(_event):
                if control.stopped.is_set():
                    pipe.cancel()

            try:
                result = pipe(PROMPT, width=512, height=512, seed=42, on_progress=on_progress)
            except GenerationCancelledError:
                if control.stopped.is_set():
                    return control.exit_code
                raise
            if control.stopped.is_set():
                return control.exit_code

        result.images[0].save("out.png")
        print(f"Saved out.png ({result.width} x {result.height}, seed {result.seed})")
    return control.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
