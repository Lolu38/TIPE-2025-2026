import gymnasium as gym
import numpy as np
import multiprocessing as mp
import time


def worker(n_runs, return_dict, worker_id):
    """Exécute n_runs épisodes et renvoie la liste des scores."""
    env = gym.make("CarRacing-v3", render_mode=None)
    scores = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 700:
            action = env.action_space.sample()  # stratégie aléatoire
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        scores.append(total_reward)
    env.close()
    return_dict[worker_id] = scores


if __name__ == "__main__":
    N_RUNS = 24
    N_WORKERS = 4    # nombre de cœurs à utiliser

    start = time.perf_counter()

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # Lancer les workers en parallèle
    runs_per_worker = N_RUNS // N_WORKERS
    for i in range(N_WORKERS):
        p = mp.Process(target=worker, args=(runs_per_worker, return_dict, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Rassembler les résultats
    all_scores = []
    for i in range(N_WORKERS):
        all_scores.extend(return_dict[i])

    print("\n===== Résultats globaux =====")
    print("Moyenne :", np.mean(all_scores))
    print("Écart-type :", np.std(all_scores))
    print("Min :", np.min(all_scores))
    print("Max :", np.max(all_scores))


    end = time.perf_counter()
    print(f"Durée totale : {end - start:.2f} secondes")
    print(f"Durée moyenne par run : {(end - start)/N_RUNS:.2f} secondes")
