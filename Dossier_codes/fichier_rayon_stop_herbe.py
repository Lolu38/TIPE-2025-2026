import gymnasium as gym
import numpy as np
import random
import math
import os

# Pour exécuter sans ouvrir de fenêtre graphique
os.environ["SDL_VIDEODRIVER"] = "dummy"


# --- Paramètres ---
N_EPISODES = 300        # Nombre de runs
MAX_STEPS = 300        # Limite de steps par run
ALPHA = 0.01           # Vitesse d'apprentissage (incrément)

# --- Initialisation ---
env = gym.make("CarRacing-v3", render_mode=None)
n_actions = env.action_space.shape[0]  # Mais CarRacing a des actions continues
# Pour simplifier trial & error : on réduit à un set discret
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # accélérer
    np.array([0.0, 0.0, 0.8]),   # frein
    np.array([-1.0, 0.5, 0.0]),  # tourner gauche + accélérer
    np.array([1.0, 0.5, 0.0]),   # tourner droite + accélérer
    np.array([0.0, 0.0, 0.0])    # rien faire
]

# Paramètres du Q-learning
ALPHA = 0.1   # taux d'apprentissage
GAMMA = 0.95  # importance du futur
EPSILON = 0.3 # exploration initiale
Q = {}  # dictionnaire des valeurs Q

def count_lines(filename):
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        line_count = sum(1 for _ in reader)
    return line_count

def discretize_state(state, precision=1): # Pour avoir plus de chances de retrouver un état déjà vu
    # Ex: precision=1 -> 0.0, 0.1, 0.2...
    return tuple(np.round(state, precision))


def is_road_pixel(px):
    r, g, b = px
    if px.max() <= 1.0:
        r, g, b = int(r*255), int(g*255), int(b*255)

    # route = ton gris foncé à moyen, pas vert ni bleu
    mean = (r+g+b)/3
    if abs(g-r) > 40 and abs(g-b) > 40 and mean < 180:
        return True
    return False


def get_ray_distances(obs, car_angle=0.0, num_rays=7, max_distance=100, step=1, cone=np.pi/2, debug=False):
    h, w, _ = obs.shape
    cx, cy = w // 2, h // 2  # obs centrée sur la voiture

    ray_angles = np.linspace(-cone, cone, num_rays)
    distances = []

    for i, a in enumerate(ray_angles):
        angle = car_angle + a
        dx = math.cos(angle)
        dy = math.sin(angle)
        distance = max_distance

        for d in range(0, max_distance, step):
            x = int(round(cx + d * dx))
            y = int(round(cy - d * dy))
            if not (0 <= x < w and 0 <= y < h):
                distance = d
                break

            px = obs[y, x]
            if px.max() <= 1.0:
                px_int = (px * 255).astype(int)
            else:
                px_int = px.astype(int)

            if not is_road_pixel(px_int):
                distance = d
                break

        distances.append(distance / float(max_distance))

    return np.array(distances)



for episode in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    step_max = MAX_STEPS + count_lines()

    while not done:
        car = env.unwrapped.car
        car_pos = car.hull.position
        car_angle = car.hull.angle
        state = get_ray_distances(obs, car_angle)
        #print(state) # test de ma fonction ray distances
        state_key = discretize_state(state, 1) # arrondi pour avoir plus de chances de retrouver un état déjà vu

        if state_key not in Q: #si nouvel état, on initialise les valeurs Q
            Q[state_key] = np.zeros(len(ACTIONS))

        # epsilon-greedy : parfois on explore
        if np.random.rand() < EPSILON:
            action_id = np.random.randint(len(ACTIONS))
        else:
            action_id = np.argmax(Q[state_key])

        action = ACTIONS[action_id]
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs[::4, ::4, :] # réduction de la résolution pour accélérer
        done = terminated or truncated or steps >= step_max

        car_pos = car.hull.position
        car_angle = car.hull.angle
        new_state = get_ray_distances(obs, car_angle)
        new_key = discretize_state(new_state, 1) # arrondi pour avoir plus de chances de retrouver un état déjà vu

        if new_key not in Q:
            Q[new_key] = np.zeros(len(ACTIONS))

        # mise à jour du Q-learning
        Q[state_key][action_id] += ALPHA * (reward + GAMMA * np.max(Q[new_key]) - Q[state_key][action_id])

        state = new_state
        total_reward += reward
        steps += 1
        #print(f"Épisode {episode+1}, step {steps}, reward {reward:.1f}, action {action_id}", end='\r')
    EPSILON = max(0.01, EPSILON * 0.995)



    print(f"--- Épisode {episode} ---")
    print(f"Taille de la Q-table : {len(Q)}")
        # Affiche 3 états pris au hasard pour voir leurs valeurs
        """for s in list(Q.keys())[:3]:
            print(f"État : {s}")
            print(f"Valeurs Q : {Q[s]}")""" #Servait pour afficher trois valeurs Q d'états au hasard mais au final juste la taille est interessante


    print(f"Épisode {episode+1} terminé, score total = {total_reward}")


env.close()




