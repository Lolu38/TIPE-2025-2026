import gymnasium as gym
import numpy as np
import math
import csv, json, os
import random
import traceback

# ---------------- CONFIG ----------------
#os.environ["SDL_VIDEODRIVER"] = "dummy"

N_EPISODES = 10000
MAX_STEPS_LIMIT = 10000  # nombre maximum de steps par épisode
ALPHA = 0.1
GAMMA = 0.95

EPSILON_START = 0.7
EPSILON_END = 0.2
EPSILON_DECAY = 0.998

BASE_SEED = 2  # seed fixe pour phase 1

Q_TABLE_FOLDER = "Save_folder_phase1/"
SCORES_FILENAME = "scores.csv"

OFFROAD_PENALTY = -10.0  # punition si sortie totale de piste
STEP_REWARD = 0.01        # reward step si sur piste
#CENTERING_BONUS = 0.01    # petit bonus si centrage

NB_CHECKPOINTS = 20       # nombre de checkpoints
CHECKPOINT_REWARD = 10    # reward par checkpoint atteint (à la fin)

DOWNSAMPLE = 2
STATE_PRECISION = 1

# ---------------- ACTIONS ----------------
# Phase 1 : pas de marche arrière, pas d'accélérations extrêmes
ACTIONS = [
    np.array([0.0, 0.5, 0.0]),     # avancer léger
    np.array([-1.0, 0.3, 0.0]),    # tourner gauche + avancer léger
    np.array([1.0, 0.3, 0.0]),     # tourner droite + avancer léger
    np.array([0.0, 0.0, 0.2]),     # frein léger
]

# ---------------- UTIL ----------------
def count_lines(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        line_count = sum(1 for _ in reader)
    return line_count

def get_worker_paths():
    os.makedirs(Q_TABLE_FOLDER, exist_ok=True)
    return {
        "q_table": os.path.join(Q_TABLE_FOLDER, "q_table.csv"),
        "scores": os.path.join(Q_TABLE_FOLDER, SCORES_FILENAME)
    }

def save_q_table(Q, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "value"])
        for state, actions in Q.items():
            sl = [float(x) for x in state]
            for a, val in enumerate(actions):
                writer.writerow([json.dumps(sl), a, float(val)])

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

def append_score(episode, score, q_size, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["episode", "total_score", "q_size"])
        writer.writerow([episode, score, q_size])

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


def get_ray_distances(obs, car_angle, num_rays=7, max_dist=70, cone=np.pi/2):
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

# ---------------- REWARD PHASE 1 (modifiée) ----------------
def is_wheel_on_grass(obs, x, y):
    h, w, _ = obs.shape
    if 0 <= x < w and 0 <= y < h:
        r, g, b = _normalize_pixel(obs[y, x, :3])
        return (g > r) and (g > b)
    return False   # très important : hors image != herbe


def is_offroad(obs, threshold=0.2):
    h, w, _ = obs.shape
    cx = w // 2
    cy = int(h * 0.75)
    WHEEL_DX = 6
    WHEEL_DY = 10
    wheels = [
        (cx - WHEEL_DX, cy - WHEEL_DY),  # front-left
        (cx + WHEEL_DX, cy - WHEEL_DY),  # front-right
        (cx - WHEEL_DX, cy + WHEEL_DY),  # rear-left
        (cx + WHEEL_DX, cy + WHEEL_DY),  # rear-right
    ]
    hits = 0
    for x, y in wheels:
        if 0 <= x < w and 0 <= y < h:
            r, g, b = obs[y, x][:3] / 255
            if g > r and g > b:
                hits += 1
    return (hits / 4) >= threshold

def compute_reward(obs_proc, step_id, speed):
    h, w, _ = obs_proc.shape
    cx, cy = w//2, h//2

    if step_id > 15:
        offroad = is_offroad(obs_proc, threshold=1)
        on_road = not offroad
    else:
        on_road = True

    reward = STEP_REWARD if (on_road and speed > 1.0) else 0.0

    # centering
    #road_check = sum(is_road_pixel(obs_proc[cy, cx+dx]) for dx in [-2,-1,0,1,2])
    #centering_bonus = CENTERING_BONUS if road_check >= 4 else 0.0
    #reward += centering_bonus

    return reward, on_road

def speed_penalty(speed):
    if speed < 0.5:
        return -0.1
    if speed < 5.0:
        return -0.03
    if speed < 10.0:
        return -0.01
    return 0.0

# ---------------- WORKER ----------------
def run_worker(worker_id, nbr_steps, epsilon):

    paths = get_worker_paths()
    Q = load_q_table(paths["q_table"])

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    env = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=MAX_STEPS_LIMIT)

    for ep in range(N_EPISODES):
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        obs, _ = env.reset(seed=BASE_SEED + ep)
        total_reward = 0.0
        checkpoints_passed = 0
        nbr_steps = nbr_steps + (3 if count_lines(paths["scores"])%5 == 0 else 0)
        nbr_steps = min(nbr_steps, MAX_STEPS_LIMIT)
        count_steps = 0
        outside_frame = 0

        for step in range(nbr_steps):
            env.render()
            obs_proc = obs[::DOWNSAMPLE, ::DOWNSAMPLE] if DOWNSAMPLE>1 else obs
            car = env.unwrapped.car
            car_angle = car.hull.angle
            vx, vy = car.hull.linearVelocity
            speed = math.sqrt(vx*vx + vy*vy)
            if count_lines(paths["scores"]) > 200:
                total_reward += speed_penalty(speed)

            rays = get_ray_distances(obs_proc, car_angle)
            sp = discretize_speed(speed)
            ang = discretize_angle(car_angle)
            full = np.concatenate([rays, np.array([sp, ang])])
            state = discretize_state(full, STATE_PRECISION)

            if state not in Q:
                Q[state] = np.zeros(len(ACTIONS))

            # epsilon-greedy
            if random.random() < epsilon:
                action_id = np.random.randint(len(ACTIONS))
            else:
                action_id = int(np.argmax(Q[state]))

            obs_new, env_reward, terminated, truncated, _ = env.step(ACTIONS[action_id])
            obs_new_proc = obs_new[::DOWNSAMPLE, ::DOWNSAMPLE] if DOWNSAMPLE>1 else obs_new

            step_reward, on_road = compute_reward(obs_new_proc, step, speed)
            total_reward += step_reward

            # sortie totale de piste
            if not on_road:
                #print(">>> KILL by offroad at step", step) utile pour débuggage
                outside_frame += 1
                total_reward += OFFROAD_PENALTY * (outside_frame/10)
                if outside_frame >= 10:
                    terminated = True  # tue l'agent après 10 steps offroad
            else:
                outside_frame = 0

            # Q-update
            car2 = env.unwrapped.car
            vx2, vy2 = car2.hull.linearVelocity
            speed2 = math.sqrt(vx2*vx2 + vy2*vy2)
            rays2 = get_ray_distances(obs_new_proc, car2.hull.angle)
            sp2 = discretize_speed(speed2)
            ang2 = discretize_angle(car2.hull.angle)
            full2 = np.concatenate([rays2, np.array([sp2, ang2])])
            next_state = discretize_state(full2, STATE_PRECISION)
            if next_state not in Q:
                Q[next_state] = np.zeros(len(ACTIONS))

            Q[state][action_id] += ALPHA * (
                step_reward + GAMMA * np.max(Q[next_state]) - Q[state][action_id]
            )

            obs = obs_new
            if terminated or truncated:
                break
            count_steps += 1

        # reward final pour checkpoints (si tu implémentes une fonction compter_checkpoints)
        total_reward += CHECKPOINT_REWARD * checkpoints_passed

        append_score(ep, total_reward, len(Q), paths["scores"])
        print(f"Ep {ep}/{N_EPISODES} | eps={epsilon:.4f} | total={total_reward:.2f} | Q={len(Q)} | steps={nbr_steps} | done_steps={count_steps}")

        if ep % 5 == 0:
            save_q_table(Q, paths["q_table"])

    save_q_table(Q, paths["q_table"])
    env.close()
    print("✅ Phase 1 terminée")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    path_scores = f"{Q_TABLE_FOLDER}/scores.csv"
    MAX_STEPS = 300 + (3*count_lines(path_scores)//5 if count_lines(path_scores)%5 == 0 else 3*(count_lines(path_scores)-count_lines(path_scores)%5)//5)
    MAX_STEPS = min(MAX_STEPS, MAX_STEPS_LIMIT)

    EPSILON_START = 0.7
    EPSILON_END = 0.2
    EPSILON_DECAY = 0.998
    epsilon = EPSILON_START * (EPSILON_DECAY ** count_lines(path_scores))

    run_worker(1, MAX_STEPS, epsilon)