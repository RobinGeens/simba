cfg = dict(
    model="simba_b_fp8",
    drop_path=0.1,  # same as simba_b_bf16
    clip_grad=1.0,  # None Disables the loss scaling # 1.0,
)
