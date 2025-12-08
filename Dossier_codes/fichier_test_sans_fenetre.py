import gymnasium as gym
import numpy as np
import time
start = time.perf_counter()

# Nombre de runs
N_RUNS = 100
scores = []

# Création de l'environnement
# render_mode=None = pas de fenêtre graphique -> plus rapide
env = gym.make("CarRacing-v3", render_mode=None)

for i in range(N_RUNS):
    # Reset = recommencer une nouvelle partie
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Action aléatoire (dans l'espace d'actions de CarRacing)
        action = env.action_space.sample()
        # Avancer d'une étape
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Fin d'épisode si la voiture a fini ou si le temps est trop long
        done = terminated or truncated
    
    scores.append(total_reward)
    if (i+1) % 1 == 0:
        print(f"Run {i+1}/{N_RUNS} terminé → Score = {total_reward}")

env.close()

# Résultats numériques
print("\n===== Résultats globaux =====")
print("Moyenne :", np.mean(scores))
print("Écart-type :", np.std(scores))
print("Min :", np.min(scores))
print("Max :", np.max(scores))

end = time.perf_counter()
print(f"Durée totale : {end - start:.2f} secondes")
print(f"Durée moyenne par run : {(end - start)/N_RUNS:.2f} secondes")


