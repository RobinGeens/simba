cfg = dict(
    model="simba_xs_fp8",
    drop_path=0.05,  # same as simba_xs_bf16
    clip_grad=1.0,  # None Disables the loss scaling # 1.0,
)
