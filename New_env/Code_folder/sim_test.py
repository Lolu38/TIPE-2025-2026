import gymnasium as gym
import f1tenth_gym

def main():
    env = gym.make("f1tenth_gym:f1tenth-v0",
                   map="example_map",
                   map_ext=".png",
                   params={"lidar_num_beams": 1080})

    obs, info = env.reset()
    done = False

    while not done:
        action = [0.5, 0.0]  # throttle, steering
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()
