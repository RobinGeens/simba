cfg = dict(
    model="simba_l_bf16",
    drop_path=0.3,
    clip_grad=1.0,  # None Disables the loss scaling # 1.0,
    # Output dir corresponds to run name
    output_dir="checkpoints/simba_l_bf16_TL",
)
