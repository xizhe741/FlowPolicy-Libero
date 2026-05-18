"""诊断脚本: 强制 eval-time obs 也走 augment=True, 测 train/eval R3M feature 分布 mismatch.

不修改 src/. monkey-patch src.data.preprocess_image, 再调用 src.eval.main.
CLI 参数透传给 src.eval (--ckpt / --method / --N / --output_dir / --num_episodes / --max_steps).

BRIGHTNESS / CONTRAST 与 scripts/precompute_r3m.py 锁定值一致 (0.3 / 0.3),
使 eval-time R3M feature 落在与 train cache 同分布的 augmented manifold 上.

用法:
    python -m scripts.eval_force_augment \
        --ckpt runs/cfm_open_the_top_drawer_..._seed42/best.pt \
        --method cfm --N 16 \
        --output_dir runs/diag_aug_open_drawer_seed42 \
        --num_episodes 20 --max_steps 300
"""
import src.data as _data

_BRIGHTNESS = 0.3
_CONTRAST = 0.3
_orig_preprocess_image = _data.preprocess_image


def _force_augment_preprocess(image, augment=False, rng=None, brightness=None, contrast=None):
    if not augment:
        augment = True
        if brightness is None:
            brightness = _BRIGHTNESS
        if contrast is None:
            contrast = _CONTRAST
    return _orig_preprocess_image(
        image, augment=augment, rng=rng, brightness=brightness, contrast=contrast,
    )


_data.preprocess_image = _force_augment_preprocess


if __name__ == "__main__":
    from src.eval import main
    main()
