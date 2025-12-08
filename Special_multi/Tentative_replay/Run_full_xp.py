import gymnasium as gym
import csv, json
import numpy as np

ACTIONS = [
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.8]),
    np.array([-1.0, 0.5, 0.0]),
    np.array([1.0, 0.5, 0.0]),
]

def play_recording(path="recorded_run"):
    with open(path + "_meta.json", "r") as f:
        meta = json.load(f)

    seed = meta["seed"]

    actions = []
    with open(path + ".csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append(int(row["action_id"]))

    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset(seed=seed)

    for act in actions:
        obs, reward, terminated, truncated, _ = env.step(ACTIONS[act])
        if terminated or truncated:
            break

    env.close()
    print("✔ Rejeu terminé !")

if __name__ == "__main__":
    play_recording("Save_folder/recorded_run")