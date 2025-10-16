import os
import time

from fastvideo import VideoGenerator

def main():
    # set the attention backend 
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    start_time = time.perf_counter()
    gen = VideoGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
    )
    load_time = time.perf_counter() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")

    gen_start_time = time.perf_counter()

    gen.generate_video(
        prompt=
        "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
        seed=1024,
        output_path="example_outputs/")
    
    generation_time = time.perf_counter() - gen_start_time
    print(f"Video generation time: {generation_time:.2f} seconds")

    total_time = time.perf_counter() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
