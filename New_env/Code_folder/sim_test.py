import os
os.environ["PYGLET_HEADLESS"] = "true"
import gym
import f110_gym
import numpy as np

DEFAULT_PARAMS = {
    'mu': 1.0489,
    'C_Sf': 4.718,
    'C_Sr': 5.4562,
    'lf': 0.15875,
    'lr': 0.17145,
    'h': 0.074,
    'm': 3.74,
    'I': 0.04712,
    's_min': -0.4189,
    's_max': 0.4189,
    'sv_min': -3.2,
    'sv_max': 3.2,
    'v_switch': 7.319,
    'a_max': 9.51,
    'v_min': -5.0,
    'v_max': 20.0,
    'width': 0.31,
    'length': 0.58,
}

def main():
    params = DEFAULT_PARAMS.copy()

    env = gym.make(
        "f110_gym:f110-v0",
        map="/mnt/c/Users/Kiketdule/Desktop/TIPE_Chiappetta_Lucas/New_env/Maps/vegas",
        map_ext=".png",
        params=params
    )
    #print(env.map_path)
    #print(env.map_ext)


    poses = np.array([
    [-11.6, -26.5, 0.0],   # voiture 1
    [-11.2, -26.5, 0.0]    # voiture 2
    ])
    poses = np.array(poses)
    obs, _, _, info = env.reset(poses = poses)
    env.render(mode = "human_fast")

    done = False

    while not done:
        action = [[0.5, 0.0], [0.5, 1.0]]
        action = np.array(action)  # throttle, steering
        obs, reward, terminated, info = env.step(action)
        env.render(mode = "human_fast")
        done = terminated

    env.close()

if __name__ == "__main__":
    main()
