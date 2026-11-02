#!/usr/bin/env python3

import shutil
import sys
import time

import numpy as np

SHADES = " .:-=+*#%@"          
CAMERA = np.array([0.0, 0.0, -4.5])
LIGHT = np.array([-0.5, 0.8, -0.7])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
HALF = 1.0                      
VIEW = 0.55                    
CHAR_ASPECT = 0.5               


def rotation(ax, ay, az):
    """Combined rotation matrix from Euler angles (radians)."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return ry @ rx @ rz


def make_rays(width, height):
    
    
    xs = np.linspace(-1, 1, width) * VIEW * (width * CHAR_ASPECT / height)
    ys = np.linspace(1, -1, height) * VIEW
    px, py = np.meshgrid(xs, ys)
    dirs = np.stack([px, py, np.ones_like(px)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def render(dirs, rot):
    
    
    origin = rot.T @ CAMERA              
    d = dirs @ rot                       
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)

   
    t1 = (-HALF - origin) / d
    t2 = (HALF - origin) / d
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)

    t_near = t_min.max(axis=-1)
    t_far = t_max.min(axis=-1)
    hit = (t_near < t_far) & (t_far > 0)

    
    axis = t_min.argmax(axis=-1)
    idx = np.indices(axis.shape)
    sign = -np.sign(d[idx[0], idx[1], axis])

    local_normal = np.zeros_like(d)
    local_normal[idx[0], idx[1], axis] = sign
    world_normal = local_normal @ rot.T  

    # Diffuse lighting
    diffuse = np.clip(world_normal @ LIGHT, 0, 1)
    brightness = 0.15 + 0.85 * diffuse

    
    hit_pt = origin + d * t_near[..., None]
    near_edge = (np.abs(hit_pt) > HALF - 0.04).sum(axis=-1) >= 2
    brightness = np.where(near_edge, brightness * 0.45, brightness)

    levels = np.clip((brightness * (len(SHADES) - 1)).astype(int), 1, len(SHADES) - 1)
    chars = np.where(hit, levels, 0)
    lut = np.array(list(SHADES))
    return ["".join(row) for row in lut[chars]]


def main():
    cols, rows = shutil.get_terminal_size((80, 24))
    width, height = cols, rows - 1      
    dirs = make_rays(width, height)

    sys.stdout.write("\x1b[2J\x1b[?25l")  
    start = time.time()
    try:
        while True:
            t = time.time() - start
            rot = rotation(t * 0.9, t * 1.3, t * 0.5)
            frame = "\n".join(render(dirs, rot))
            sys.stdout.write("\x1b[H" + frame)
            sys.stdout.flush()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\x1b[0m\n")  


if __name__ == "__main__":
    main()
