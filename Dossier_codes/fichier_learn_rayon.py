import gymnasium as gym
import numpy as np
import random
import math
import os

# Pour exécuter sans ouvrir de fenêtre graphique
os.environ["SDL_VIDEODRIVER"] = "dummy"


# --- Paramètres ---
N_EPISODES = 20        # Nombre de runs
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

# Tableau de "valeur de préférence" pour chaque action (trial & error)
#action_scores = np.zeros(len(ACTIONS))
"""action_scores[0] = 0.5  # on initialise en favorisant l'accélération
action_scores[1] = 0.1  # un peu de frein
action_scores[2] = 0.2  # aime quand même bien tourner
action_scores[3] = 0.2  # aimer quand même bien tourner
action_scores[4] = 0.0  # détester ne rien faire"""

moyenne_score = 0


def get_ray_distances(obs, car_pos, car_angle, n_rays=7, max_distance=30):
    """Calcule les distances aux bords de la route dans plusieurs directions (rayons)."""
    
    h, w, _ = obs.shape
    cx, cy = w // 2, h // 2
    distances = []
    ray_angles = np.linspace(-math.pi/2, math.pi/2, n_rays)  # -90° à +90°

    for delta in ray_angles:
        angle = car_angle + delta
        distance = 0
        
        for step in range(1, max_distance):
            x = int(cx + math.cos(angle) * step)
            y = int(cy - math.sin(angle) * step)
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            pixel = obs[y, x]
            if pixel[1] > 120:  # vert => herbe
                break
            distance = step
        distances.append(distance / max_distance)

    return np.array(distances)



# Paramètres du Q-learning
ALPHA = 0.1   # taux d'apprentissage
GAMMA = 0.95  # importance du futur
EPSILON = 0.2 # exploration
ACTIONS = [
    np.array([0, 1, 0]),    # accélère
    np.array([0, 0, 0]),    # rien
    np.array([-1, 1, 0]),   # tourne gauche
    np.array([1, 1, 0]),    # tourne droite
    np.array([0, 0, 1]),    # freine
]

Q = {}  # dictionnaire des valeurs Q

for episode in range(50):  # 50 runs pour tester
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        car = env.unwrapped.car
        car_pos = car.hull.position
        car_angle = car.hull.angle
        state = get_ray_distances(obs, car_pos, car_angle)
        print(state) # test de ma fonction ray distances
        state_key = tuple(np.round(state, 2))

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
        done = terminated or truncated or steps >= MAX_STEPS

        car_pos = car.hull.position
        car_angle = car.hull.angle
        new_state = get_ray_distances(obs, car_pos, car_angle)
        new_key = tuple(np.round(new_state, 2))

        if new_key not in Q:
            Q[new_key] = np.zeros(len(ACTIONS))

        # mise à jour du Q-learning
        Q[state_key][action_id] += ALPHA * (reward + GAMMA * np.max(Q[new_key]) - Q[state_key][action_id])

        state = new_state
        total_reward += reward
        steps += 1
        print(f"Épisode {episode+1}, step {steps}, reward {reward:.1f}, action {action_id}", end='\r')
    EPSILON = max(0.01, EPSILON * 0.995)

    print(f"Épisode {episode+1} terminé, score total = {total_reward:.1f}")


env.close()
moyenne_score /= N_EPISODES
#print("Scores finaux des actions :", action_scores)
print("Moyenne des scores :", moyenne_score)



