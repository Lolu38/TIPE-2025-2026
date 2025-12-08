import gymnasium as gym
import numpy as np
import random

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
action_scores = np.zeros(len(ACTIONS))
action_scores[0] = 0.5  # on initialise en favorisant l'accélération
action_scores[1] = 0.1  # un peu de frein
action_scores[2] = 0.2  # aime quand même bien tourner
action_scores[3] = 0.2  # aime quand même bien tourner
action_scores[4] = 0.0  # détester ne rien faire

moyenne_score = 0

# --- Boucle d'apprentissage ---
for episode in range(N_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(MAX_STEPS):
        # Choisir une action proportionnellement aux scores actuels
        probs = np.exp(action_scores) / np.sum(np.exp(action_scores))  # softmax
        action_idx = np.random.choice(len(ACTIONS), p=probs)
        action = ACTIONS[action_idx]

        # Jouer l'action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Mettre à jour le score de l’action choisie (trial & error incrémental)
        action_scores[action_idx] += ALPHA * reward

        total_reward += reward
        if done:
            break

    print(f"Episode {episode+1}/{N_EPISODES} - Reward total : {total_reward:.1f}")
    moyenne_score += total_reward

env.close()
moyenne_score /= N_EPISODES
print("Scores finaux des actions :", action_scores)
print("Moyenne des scores :", moyenne_score)
