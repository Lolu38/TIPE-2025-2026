import gymnasium as gym
import random
import math
import ast
import csv, json, os
import numpy as np

# Pour exécuter sans ouvrir de fenêtre graphique
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Paramètres ---
N_EPISODES = 1000
MAX_STEPS = 300
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.5

Q_TABLE_FILE = "Dossier_enregistrement/stop_herbe/q_table.csv"
SCORES_FILE = "Dossier_enregistrement/stop_herbe/scores.csv"

def count_lines(filename=SCORES_FILE):
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        line_count = sum(1 for _ in reader)
    return line_count


# --- Initialisation ---
env = gym.make("CarRacing-v3", render_mode=None)
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # accélérer
    np.array([0.0, 0.0, 0.8]),   # frein
    np.array([-1.0, 0.5, 0.0]),  # tourner gauche + accélérer
    np.array([1.0, 0.5, 0.0]),   # tourner droite + accélérer
    np.array([0.0, 0.0, 0.0])    # rien faire
]

# --- Fonctions utilitaires de Q-learning ---
def discretize_state(state, precision=4):
    return tuple(round(float(x), precision) for x in state)

def save_q_table(Q, filename=Q_TABLE_FILE):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "value"])
        for state, actions in Q.items():
            # convert state to plain Python list (no numpy types)
            state_list = [int(x) if (hasattr(x, "dtype") or isinstance(x, (np.integer, np.floating))) else x for x in state]
            state_json = json.dumps(state_list)
            for a, val in enumerate(actions):
                writer.writerow([state_json, a, float(val)])

def load_q_table(filename=Q_TABLE_FILE):
    Q = {}
    if os.path.exists(filename):
        with open(filename, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state_list = json.loads(row["state"])
                state = tuple(state_list)
                action = int(row["action"])
                value = float(row["value"])
                if state not in Q:
                    # allocate array sized to your action count (example 5)
                    Q[state] = np.zeros(len(ACTIONS))  # ACTIONS doit être défini avant
                Q[state][action] = value
    return Q

def append_score_to_csv(episode, total_score, q_size, filename=SCORES_FILE):
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "total_score", "q_size"])
        writer.writerow([episode, total_score, q_size])

def is_road_pixel(px):
    r, g, b = px
    if px.max() <= 1.0:
        r, g, b = int(r*255), int(g*255), int(b*255)
    mean = (r+g+b)/3
    if abs(g-r) > 40 and abs(g-b) > 40 and mean < 180:
        return True
    return False

def get_ray_distances(obs, car_angle=0.0, num_rays=7, max_distance=60, step=3, cone=np.pi/2):
    """
    Calcule les distances à l'herbe pour plusieurs rayons autour de la voiture.
    Utilise un pas large pour aller vite, puis affine localement lorsqu'on touche l'herbe.
    """
    h, w, _ = obs.shape
    cx, cy = w // 2, h // 2  # centre (voiture)
    ray_angles = np.linspace(-cone, cone, num_rays)
    distances = []

    for a in ray_angles:
        angle = car_angle + a
        dx, dy = math.cos(angle), math.sin(angle)
        distance = max_distance

        for d in range(0, max_distance, step):
            x = int(round(cx + d * dx))
            y = int(round(cy - d * dy))
            if not (0 <= x < w and 0 <= y < h):
                distance = d
                break

            px = obs[y, x]
            if px.max() <= 1.0:
                px = (px * 255).astype(int)
            else:
                px = px.astype(int)

            if not is_road_pixel(px):
                # --- Raffinement local ---
                for dd in range(max(0, d - step), d):
                    x2 = int(round(cx + dd * dx))
                    y2 = int(round(cy - dd * dy))
                    if not (0 <= x2 < w and 0 <= y2 < h):
                        distance = dd
                        break
                    px2 = obs[y2, x2]
                    if px2.max() <= 1.0:
                        px2 = (px2 * 255).astype(int)
                    else:
                        px2 = px2.astype(int)
                    if not is_road_pixel(px2):
                        distance = dd
                        break
                break  # on a trouvé la bordure

        distances.append(distance / float(max_distance))  # normalisation [0,1]

    return np.array(distances)

# --- Boucle principale ---
Q = load_q_table()

for episode in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        car = env.unwrapped.car
        car_angle = car.hull.angle
        state = get_ray_distances(obs, car_angle)
        #print(state) # test de ma fonction ray distances
        state_key = discretize_state(state, 1)
        #print(state_key) # test de ma fonction ray distances

        if state_key not in Q:
            Q[state_key] = np.zeros(len(ACTIONS))

        # epsilon-greedy
        if np.random.rand() < EPSILON:
            action_id = np.random.randint(len(ACTIONS))
        else:
            action_id = np.argmax(Q[state_key])

        action = ACTIONS[action_id]
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs[::4, ::4, :]  # downsampling
        done = terminated or truncated or steps >= MAX_STEPS

        new_state = get_ray_distances(obs, car_angle)
        new_key = discretize_state(new_state, 1)

        if new_key not in Q:
            Q[new_key] = np.zeros(len(ACTIONS))

        Q[state_key][action_id] += ALPHA * (reward + GAMMA * np.max(Q[new_key]) - Q[state_key][action_id])

        total_reward += reward
        steps += 1

        #print(f"Épisode {episode+1}, step {steps}, reward {reward:.1f}, action {action_id}", end='\r')
    EPSILON = max(0.01, EPSILON * (0.995 ** count_lines()))
    ALPHA = max(0.01, ALPHA * (0.995 ** count_lines()))


    append_score_to_csv(episode+1, total_reward, len(Q))
    if (episode+1) % 10 == 0:
        print(f"--- Épisode {episode+1} terminé --- Score={total_reward:.2f} | États Q={len(Q)}")
        save_q_table(Q)

save_q_table(Q)
env.close()