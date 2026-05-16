"""
LIBERO HDF5 agentview 与 env.reset() agentview 方向一致性 sanity check.

PNG 输出到 scripts/out_agentview_check/:
  - hdf5_frame0.png       训练 HDF5 中 demo_0 的首帧 agentview
  - env_reset.png         env.reset() 后 obs['agentview_image']
  - env_reset_vflip.png   上面 [::-1] 后

肉眼比对 hdf5_frame0 与 env_reset / env_reset_vflip 哪个上下方向一致。
diff 一致的那个即 eval-time 应用的方向规则。
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

DATASET_DIR = Path("/root/shared-nvme/data/libero_goal")
OUT_DIR = Path(__file__).resolve().parent / "out_agentview_check"


def load_hdf5_agentview():
    hdf5_paths = sorted(DATASET_DIR.glob("*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"no .hdf5 under {DATASET_DIR}")
    path = hdf5_paths[0]
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        demo = f["data"][demo_keys[0]]
        obs_keys = list(demo["obs"].keys())
        # LIBERO HDF5 schema 在不同版本下 agentview key 命名不同
        key = (
            "agentview_rgb"
            if "agentview_rgb" in obs_keys
            else next(k for k in obs_keys if "agentview" in k)
        )
        img = np.asarray(demo["obs"][key][0])
    return {
        "path": path.name,
        "demo": demo_keys[0],
        "key": key,
        "obs_keys": obs_keys,
        "img": img,
    }


def load_env_agentview(camera_h: int, camera_w: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
    bddl = task_suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=camera_h,
        camera_widths=camera_w,
    )
    env.seed(0)
    obs = env.reset()
    obs_keys = list(obs.keys())
    img = np.asarray(obs["agentview_image"])
    env.close()
    return {
        "task": task_suite.get_task(0).name,
        "obs_keys": obs_keys,
        "img": img,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)

    h = load_hdf5_agentview()
    H, W = h["img"].shape[:2]
    print(f"[hdf5] file={h['path']}  demo={h['demo']}  key={h['key']}")
    print(f"       img.shape={h['img'].shape}  dtype={h['img'].dtype}")
    print(f"       obs_keys={h['obs_keys']}")

    e = load_env_agentview(camera_h=H, camera_w=W)
    print(f"[env]  task={e['task']}")
    print(f"       img.shape={e['img'].shape}  dtype={e['img'].dtype}")
    print(f"       obs_keys={e['obs_keys']}")

    imageio.imwrite(OUT_DIR / "hdf5_frame0.png", h["img"])
    imageio.imwrite(OUT_DIR / "env_reset.png", e["img"])
    imageio.imwrite(OUT_DIR / "env_reset_vflip.png", e["img"][::-1])

    print(f"\nPNGs under {OUT_DIR}")
    print("compare hdf5_frame0 vs env_reset / env_reset_vflip — 上下方向一致的那个即 eval 规则")


if __name__ == "__main__":
    main()
