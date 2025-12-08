import gymnasium as gym
import math
import csv, json, os
import numpy as np
import multiprocessing as mp
import random
import traceback

# ---------------- CONFIG ----------------
os.environ["SDL_VIDEODRIVER"] = "dummy"

N_WORKERS = 1
N_EPISODES = 1000
BASE_MAX_STEPS = 300
MAX_STEPS_LIMIT = 10000

ALPHA = 0.1
GAMMA = 0.95

# OPTION B - epsilon schedule
EPSILON_START = 0.7
EPSILON_END = 0.2
EPSILON_DECAY = 0.998

BASE_SEED = 1

Q_TABLE_FOLDER = "Save_folder/"
SCORES_FILENAME = "scores.csv"

# NEW — speed reward reduced
SPEED_REWARD_FACTOR = 0.002   # essai 1 (au lieu de 0.002)
# ou si tu veux un essai plus agressif:
# SPEED_REWARD_FACTOR = 0.005

# et adoucir la pénalité hors piste:
OFFROAD_PENALTY = -0.1   # au lieu de -0.2
REWARD_CLIP = 5.0

# Downsample factor
DOWNSAMPLE = 2
STATE_PRECISION = 1

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
        next(reader)
        return sum(1 for _ in reader)

def load_q_table(path):
    Q = {}
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    state = tuple(json.loads(row["state"]))
                    action = int(row["action"])
                    value = float(row["value"])
                    if state not in Q:
                        Q[state] = np.zeros(len(ACTIONS))
                    Q[state][action] = value
                except:
                    continue
    return Q

# ---------------- ACTIONS ----------------
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.8]),
    np.array([-1.0, 0.5, 0.0]),
    np.array([1.0, 0.5, 0.0]),
]

# --------------- RAY CAST / PIXEL --------------
def _normalize_pixel(px):
    px = np.asarray(px)
    if px.max() <= 1.0:
        px = (px * 255.0).astype(np.int16)
    else:
        px = px.astype(np.int16)
    return px

def is_road_pixel(px):
    px = _normalize_pixel(px)
    r,g,b = px
    mean = (r+g+b)/3
    return (abs(g-r) > 40) and (abs(g-b) > 40) and (mean < 180)

def get_ray_distances(obs, car_angle, num_rays=7, max_dist=60, cone=np.pi/2):
    h, w, _ = obs.shape
    cx, cy = w//2, h//2
    rays = np.linspace(-cone, cone, num_rays)
    out = []
    for a in rays:
        ang = car_angle + a
        dx, dy = math.cos(ang), math.sin(ang)
        low, high = 0, max_dist
        while low < high:
            d = (low+high)//2
            x, y = int(cx+d*dx), int(cy-d*dy)
            if x<0 or x>=w or y<0 or y>=h:
                high = d-1
                continue
            if is_road_pixel(obs[y,x,:3]):
                low = d+1
            else:
                high = d-1
        out.append(low/max_dist)
    return np.array(out)

# ------------- SPEED & ANGLE discretization -------------
def discretize_speed(speed):
    if speed < 5: return 0
    if speed < 10: return 1
    if speed < 15: return 2
    if speed < 20: return 3
    return 4

def discretize_angle(angle):
    norm = (angle + math.pi)/(2*math.pi)
    idx = int(norm * 5)
    return min(4, max(0, idx))

def discretize_state(state, precision):
    return tuple(round(float(x), precision) for x in state)

# ---------------------------------------------------------
#  NEW METHOD : compute_reward = Option A + Option D
# ---------------------------------------------------------
def compute_reward(env_reward, obs_proc, speed):
    h,w,_ = obs_proc.shape
    cx,cy = w//2, h//2
    on_road = is_road_pixel(obs_proc[cy, cx])

    speed_bonus = speed * SPEED_REWARD_FACTOR if on_road else 0.0
    offroad_penalty = 0.0 if on_road else OFFROAD_PENALTY

    return env_reward + speed_bonus + offroad_penalty


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
            epsilon = max(EPSILON_END, EPSILON_START *
                          (EPSILON_DECAY ** count_lines(paths["scores"])))

            nbr_steps = nbr_steps + (3 if count_lines(paths["scores"])%2==0 else 0)
            nbr_steps = min(nbr_steps, MAX_STEPS_LIMIT)

            obs, _ = env.reset(seed=seed)
            total_reward = 0.0

            for step in range(int(nbr_steps)):

                env.render()

                # downsample
                obs_proc = obs[::DOWNSAMPLE, ::DOWNSAMPLE] if DOWNSAMPLE>1 else obs

                # car vars
                car = env.unwrapped.car
                car_angle = car.hull.angle

                # state
                rays = get_ray_distances(obs_proc, car_angle)
                vx, vy = car.hull.linearVelocity
                speed = math.sqrt(vx*vx + vy*vy)
                sp = discretize_speed(speed)
                ang = discretize_angle(car_angle)

                full = np.concatenate([rays, np.array([sp, ang])])
                state = discretize_state(full, STATE_PRECISION)

                # epsilon-greedy
                if random.random() < epsilon:
                    action_id = np.random.randint(len(ACTIONS))
                else:
                    action_id = int(np.argmax(Q[state]))

                obs_new, env_reward, terminated, truncated, _ = env.step(ACTIONS[action_id])

                obs_new_proc = obs_new[::DOWNSAMPLE, ::DOWNSAMPLE] if DOWNSAMPLE>1 else obs_new

                # NEW : compute reward
                step_reward = compute_reward(env_reward, obs_proc, speed)

                # next state
                car2 = env.unwrapped.car
                vx2, vy2 = car2.hull.linearVelocity
                speed2 = math.sqrt(vx2*vx2 + vy2*vy2)

                rays2 = get_ray_distances(obs_new_proc, car2.hull.angle)
                sp2 = discretize_speed(speed2)
                ang2 = discretize_angle(car2.hull.angle)
                full2 = np.concatenate([rays2, np.array([sp2, ang2])])
                next_state = discretize_state(full2, STATE_PRECISION)

                # Q-update
                Q[state][action_id] += ALPHA * (
                    step_reward + GAMMA * np.max(Q[next_state]) - Q[state][action_id]
                )

                total_reward += step_reward
                obs = obs_new

                if terminated or truncated:
                    break

            print(f"Ep {ep}/{N_EPISODES} | eps={epsilon:.4f} | steps={nbr_steps} | total={total_reward:.2f} | Q={len(Q)}")

        env.close()
        print(f"✅ Worker {worker_id} terminé")

    except Exception as e:
        print(f"❌ Erreur worker {worker_id}: {e}")
        traceback.print_exc()


# --------------- MAIN ----------------
if __name__ == "__main__":
    procs = []
    for wid in range(1, N_WORKERS+1):
        path_scores = f"Save_folder/worker{wid}/scores.csv"
        MAX_STEPS = 300 + (3*count_lines(path_scores)//2 if count_lines(path_scores)%2==0 else 3*(count_lines(path_scores)-1)//2)
        MAX_STEPS = min(MAX_STEPS, MAX_STEPS_LIMIT)
        p = mp.Process(target=run_worker, args=(wid, MAX_STEPS))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
