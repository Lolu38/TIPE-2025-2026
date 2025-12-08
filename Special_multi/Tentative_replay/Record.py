import gymnasium as gym
import imageio
import numpy as np
import json
import os
import math
import csv

# -------------------------------------------------------
# PARAMÈTRES
# -------------------------------------------------------
VIDEO_PATH = "recordings/"
FILENAME = "run.mp4"
FPS = 30

ACTIONS = [
    np.array([0.0, 1.0, 0.0]),     # accélère
    np.array([0.0, 0.0, 0.8]),     # freine
    np.array([-1.0, 0.5, 0.0]),    # gauche
    np.array([1.0, 0.5, 0.0]),     # droite
]

DOWNSAMPLE = 2
STATE_PRECISION = 1

# -------------------------------------------------------
# FONCTIONS IDENTIQUES À L'ENTRAÎNEMENT
# -------------------------------------------------------

def _normalize_pixel(px):
    px = np.asarray(px)
    if np.issubdtype(px.dtype, np.floating) and px.max() <= 1.0:
        px = (px * 255.0).astype(np.int16)
    else:
        px = px.astype(np.int16)
    return px

def is_road_pixel(px):
    px = _normalize_pixel(px)
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    mean = (r + g + b) / 3.0
    return (abs(g - r) > 40) and (abs(g - b) > 40) and (mean < 180)

def get_ray_distances(obs, car_angle=0.0, num_rays=7, max_dist=60, cone=np.pi/2):
    h, w, _ = obs.shape
    cx, cy = w // 2, h // 2
    angles = np.linspace(-cone, cone, num_rays)
    distances = []

    for a in angles:
        angle = car_angle + a
        dx, dy = math.cos(angle), math.sin(angle)

        low, high = 0, max_dist
        while low < high:
            d = (low + high) // 2
            x = int(cx + d * dx)
            y = int(cy - d * dy)

            if x < 0 or x >= w or y < 0 or y >= h:
                high = d - 1
                continue

            if is_road_pixel(obs[y, x, :3]):
                low = d + 1
            else:
                high = d - 1

        distances.append(low / max_dist)

    return np.array(distances)

def discretize_speed(speed):
    if speed < 5: return 0
    elif speed < 10: return 1
    elif speed < 15: return 2
    elif speed < 20: return 3
    return 4

def discretize_angle(angle):
    norm = (angle + math.pi) / (2 * math.pi)
    idx = int(norm * 5)
    return max(0, min(4, idx))

def discretize_state(state, precision=STATE_PRECISION):
    return tuple(round(float(x), precision) for x in state)

def get_state(env, obs):
    """Recrée EXACTEMENT l'état utilisé pendant l'entraînement."""

    # Downsample identique
    if DOWNSAMPLE > 1:
        obs_proc = obs[::DOWNSAMPLE, ::DOWNSAMPLE, :]
    else:
        obs_proc = obs

    car = env.unwrapped.car
    car_angle = float(car.hull.angle)

    # Rayons
    ray_state = get_ray_distances(obs_proc, car_angle)

    # Vitesse + angle
    vx = float(car.hull.linearVelocity.x)
    vy = float(car.hull.linearVelocity.y)
    speed = math.sqrt(vx*vx + vy*vy)

    speed_bin = discretize_speed(speed)
    angle_bin = discretize_angle(car_angle)

    full_state = np.concatenate([ray_state, np.array([speed_bin, angle_bin])])

    return discretize_state(full_state, precision=STATE_PRECISION)

def choose_action(state, Q):
    if state not in Q:
        return 0
    return int(np.argmax(Q[state]))

def load_q_table(path):
    Q = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = tuple(json.loads(row["state"]))
            a = int(row["action"])
            v = float(row["value"])
            if s not in Q:
                Q[s] = np.zeros(4)
            Q[s][a] = v
    return Q

# -------------------------------------------------------
# RECORDER
# -------------------------------------------------------

def record_run(qtable_path, seed=0, log_states=False):
    os.makedirs(VIDEO_PATH, exist_ok=True)

    Q = load_q_table(qtable_path)

    # IMPORTANT : le recorder doit utiliser render_mode="rgb_array"
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    writer = imageio.get_writer(VIDEO_PATH + FILENAME, fps=FPS)

    total_reward = 0
    step = 0
    done = False

    while not done:
        # ---- 1) Lire l'état depuis OBS (identique entraînement) ----
        state = get_state(env, obs)

        if log_states:
            print(f"STATE {step} :", state)

        # ---- 2) Choisir action ----
        action_id = choose_action(state, Q)

        # ---- 3) APPLIQUER L’ACTION ----
        obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
        total_reward += reward

        # ---- 4) RENDRE/APPELER render() APRÈS step() ----
        frame = env.render()
        writer.append_data(frame)

        step += 1

        if terminated or truncated:
            done = True

    writer.close()
    env.close()

    print(f"🎬 Vidéo enregistrée → {VIDEO_PATH}{FILENAME}")
    print(f"⭐ Reward total : {total_reward}")

    with open(VIDEO_PATH + "meta.json", "w") as f:
        json.dump({"seed": seed, "reward": total_reward}, f, indent=2)

# -------------------------------------------------------

if __name__ == "__main__":
    record_run("Save_folder/worker1/q_table.csv", seed=1, log_states=False)
