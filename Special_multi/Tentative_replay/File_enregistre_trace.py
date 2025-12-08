import gymnasium as gym
import math
import csv, json, os
import numpy as np
import multiprocessing as mp
import random
import traceback

# ---------------- CONFIG ----------------
os.environ["SDL_VIDEODRIVER"] = "dummy"

N_WORKERS = 1           # mets 1 pour tester, puis augmente si tu veux
N_EPISODES = 1000
BASE_MAX_STEPS = 300
MAX_STEPS_LIMIT = 1000

ALPHA = 0.1
GAMMA = 0.95

# OPTION B - epsilon schedule (sigma-boost)
EPSILON_START = 0.7
EPSILON_END = 0.2
EPSILON_DECAY = 0.998   # multiplicative per episode
Seuil_score = 150

BASE_SEED = 1

Q_TABLE_FOLDER = "Save_folder/"
SCORES_FILENAME = "scores.csv"

# OPTION D - speed bonus + clipping
SPEED_REWARD_FACTOR = 0.01
REWARD_CLIP = 5.0

# Downsample factor (Option C)
DOWNSAMPLE = 2   # 1 = no downsample, 2 = light, 4 = heavy (éviter 4)
# State discretization precision (Option A)
STATE_PRECISION = 1  # round(..., 1)

# ---------------- UTIL ----------------
def get_worker_paths(worker_id):
    folder = os.path.join(Q_TABLE_FOLDER, f"worker{worker_id}")
    os.makedirs(folder, exist_ok=True)
    return {
        "q_table": os.path.join(folder, "q_table.csv"),
        "scores": os.path.join(folder, SCORES_FILENAME)
    }

def count_lines(filename):
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        line_count = sum(1 for _ in reader)
    return line_count

def save_q_table(Q, filename):
    # sauvegarde simple CSV (state JSON, action, value)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "value"])
        for state, actions in Q.items():
            state_list = [float(x) for x in state]
            for a, val in enumerate(actions):
                writer.writerow([json.dumps(state_list), a, float(val)])

def load_q_table(path):
    Q = {}
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    state = tuple(json.loads(row["state"]))
                    action = int(row["action"])
                    value = float(row["value"])
                    if state not in Q:
                        Q[state] = np.zeros(len(ACTIONS))
                    Q[state][action] = value
                except Exception:
                    # ignore malformed lines
                    continue
    return Q

def append_score(episode, score, q_size, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "total_score", "q_size"])
        writer.writerow([episode, score, q_size])

# ---------------- ACTIONS ----------------
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # accélérer
    np.array([0.0, 0.0, 0.8]),   # frein
    np.array([-1.0, 0.5, 0.0]),  # tourner gauche + accélérer
    np.array([1.0, 0.5, 0.0]),   # tourner droite + accélérer
]

# --------------- RAY CAST / PIXEL --------------
def _normalize_pixel(px):
    """Retourne px en int16 (0..255) pour éviter overflow uint8."""
    px = np.asarray(px)
    if np.issubdtype(px.dtype, np.floating) and px.max() <= 1.0:
        px = (px * 255.0).astype(np.int16)
    else:
        px = px.astype(np.int16)
    return px

def is_road_pixel(px):
    """Retourne True si pixel considéré comme route (heuristique simple)."""
    px = _normalize_pixel(px)
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    mean = (r + g + b) / 3.0
    return (abs(g - r) > 40) and (abs(g - b) > 40) and (mean < 180)

def get_ray_distances(obs, car_angle=0.0, num_rays=7, max_dist=60, cone=np.pi/2):
    """
    Calcule distances normalisées [0,1] le long de num_rays rayons.
    Binarisation par recherche dichotomique (rapide).
    """
    h, w, _ = obs.shape
    cx, cy = w // 2, h // 2
    ray_angles = np.linspace(-cone, cone, num_rays)
    distances = []

    for a in ray_angles:
        angle = car_angle + a
        dx, dy = math.cos(angle), math.sin(angle)
        low, high = 0, max_dist
        dist = max_dist
        while low < high:
            d = (low + high) // 2
            x = int(cx + d * dx)
            y = int(cy - d * dy)
            if x < 0 or x >= w or y < 0 or y >= h:
                dist = d
                high = d - 1
                continue
            px = obs[y, x, :3]
            if is_road_pixel(px):
                low = d + 1
            else:
                dist = d
                high = d - 1
        distances.append(low / max_dist)
    return np.array(distances)

# ------------- SPEED & ANGLE discretization -------------
def discretize_speed(speed):
    # ajustable si tu observes des vitesses différentes
    if speed < 5: return 0
    elif speed < 10: return 1
    elif speed < 15: return 2
    elif speed < 20: return 3
    return 4

def discretize_angle(angle):
    norm = (angle + math.pi) / (2 * math.pi)  # 0..1
    idx = int(norm * 5)
    if idx < 0: idx = 0
    if idx > 4: idx = 4
    return idx

def discretize_state(state, precision=STATE_PRECISION):
    # safe conversion to tuple key
    return tuple(round(float(x), precision) for x in state)

# -------------------- WORKER --------------------
def run_worker(worker_id, nbr_steps):
    try:
        paths = get_worker_paths(worker_id)
        Q = load_q_table(paths["q_table"])

        seed = BASE_SEED + worker_id
        random.seed(seed)
        np.random.seed(seed)

        env = gym.make("CarRacing-v3", render_mode=None)
        env.reset(seed=seed)

        for ep in range(N_EPISODES):
            # OPTION B: epsilon schedule (multiplicative)
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** count_lines(paths["scores"])))

            # progressive MAX_STEPS (int) - augmente doucement, palier entier
            nbr_steps= nbr_steps + (3 if count_lines(paths["scores"])%2 == 0 else 0)
            nbr_steps = min(nbr_steps, MAX_STEPS_LIMIT)

            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            total_actions = []

            for step in range(int(nbr_steps)):
                # optionally downsample gently (Option C)
                if DOWNSAMPLE > 1:
                    obs_proc = obs[::DOWNSAMPLE, ::DOWNSAMPLE, :]
                else:
                    obs_proc = obs

                car = env.unwrapped.car
                car_angle = car.hull.angle

                # ray distances
                ray_state = get_ray_distances(obs_proc, car_angle)

                # speed & angle (world)
                vx = float(car.hull.linearVelocity.x)
                vy = float(car.hull.linearVelocity.y)
                speed = math.sqrt(vx*vx + vy*vy)
                speed_bin = discretize_speed(speed)
                angle_bin = discretize_angle(car_angle)

                # full state: rays + speed_bin + angle_bin
                full_state = np.concatenate([ray_state, np.array([speed_bin, angle_bin])])
                state_key = discretize_state(full_state, precision=STATE_PRECISION)

                # epsilon-greedy
                if random.random() < epsilon:
                    action_id = np.random.randint(len(ACTIONS))
                else:
                    action_id = int(np.argmax(Q[state_key]))

                action = ACTIONS[action_id]
                total_actions.append(action_id)
                obs_new, env_reward, terminated, truncated, _ = env.step(action)

                # downsample the new observation similarly
                if DOWNSAMPLE > 1:
                    obs_new_proc = obs_new[::DOWNSAMPLE, ::DOWNSAMPLE, :]
                else:
                    obs_new_proc = obs_new

                # OPTION D: small speed-based reward to encourage forward movement
                speed_bonus = speed * SPEED_REWARD_FACTOR
                step_reward = env_reward + speed_bonus

                # clip to avoid extreme jumps
                step_reward = max(-REWARD_CLIP, min(REWARD_CLIP, step_reward))

                # compute next state AFTER the step (use new car velocities)
                # Note: env.step may update car.hull, so sample again
                car_after = env.unwrapped.car
                new_angle = float(car_after.hull.angle)
                new_vx = float(car_after.hull.linearVelocity.x)
                new_vy = float(car_after.hull.linearVelocity.y)
                new_speed = math.sqrt(new_vx*new_vx + new_vy*new_vy)
                new_speed_bin = discretize_speed(new_speed)
                new_angle_bin = discretize_angle(new_angle)
                new_ray = get_ray_distances(obs_new_proc, new_angle)

                new_state = np.concatenate([new_ray, np.array([new_speed_bin, new_angle_bin])])
                new_key = discretize_state(new_state, precision=STATE_PRECISION)

                if new_key not in Q:
                    Q[new_key] = np.zeros(len(ACTIONS))

                # Q-learning update
                Q[state_key][action_id] += ALPHA * (step_reward + GAMMA * np.max(Q[new_key]) - Q[state_key][action_id])

                total_reward += step_reward

                obs = obs_new  # update obs for next iteration

                if terminated or truncated:
                    break

            # append results & save periodically
            append_score(ep, total_reward, len(Q), paths["scores"])
            print(f"Ep {ep}/{N_EPISODES} | eps={epsilon:.4f} | steps={nbr_steps} | total={total_reward:.2f} | Q={len(Q)}")
            if total_reward >= Seuil_score:
                print(f"🎉 Worker {worker_id} a atteint le seuil de score {Seuil_score} à l'épisode {ep} ! 🎉")
                print("On va desomrmais enregsitrer les traces de cet agent.")
                trace_path = os.path.join("Save_folder/Enregistre_trace", f"trace_ep{ep}.csv")
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                with open(trace_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "action_id"])
                    for step_id, act_id in enumerate(total_actions):
                        writer.writerow([step_id, act_id])
            # sauvegarde Q-table tous les épisodes (ou tous les 10 pour moins d'E/S)
            if ep % 5 == 0:
                save_q_table(Q, paths["q_table"])

        # final save
        save_q_table(Q, paths["q_table"])
        env.close()
        print(f"✅ Worker {worker_id} terminé (Q states = {len(Q)})")

    except Exception as e:
        print(f"❌ Erreur dans worker {worker_id}: {e}")
        traceback.print_exc()

# --------------- MAIN ----------------
if __name__ == "__main__":
    procs = []
    for wid in range(1, N_WORKERS + 1):
        path_scores = f"Save_folder/worker{wid}/scores.csv"
        MAX_STEPS = 300 + (3*count_lines(path_scores)// 2 if count_lines(path_scores)%2 == 0 else 3*(count_lines(path_scores)-1)//2)
        MAX_STEPS = min(MAX_STEPS, MAX_STEPS_LIMIT)
        p = mp.Process(target=run_worker, args=(wid, MAX_STEPS))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

# -------------- FIN DU FICHIER --------------
