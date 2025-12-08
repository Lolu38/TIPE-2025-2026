import gymnasium as gym

# Créer l'environnement CarRacing
env = gym.make("CarRacing-v3", render_mode="human")

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # Action aléatoire (accélère / freine / tourne)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    env.render()

env.close()
print("Score total :", total_reward)