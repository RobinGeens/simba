cfg = dict(
    model="simba_xs_bf16",
    drop_path=0.05,  # smaller model -> less regularization than Simba-S (0.1)
    clip_grad=1.0,
)
