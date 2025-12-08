import gymnasium as gym
import os
import pygame
import numpy as np
import cv2

# --- Paramètres ---
SAVE_FOLDER = "scan_images"
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, _ = env.reset(seed=4)

# 1. Faire un premier rendu pour que 'surf' soit généré
env.render()

# 2. Récupérer l'image pygame
surf = env.unwrapped.surf

if surf is None:
    raise RuntimeError("surf est None : Gymnasium n'a pas encore généré l'image.")

# 3. Convertir pygame.Surface → array numpy
pg_img = pygame.surfarray.array3d(surf)     # format (W,H,3), RGB

# 4. Repasser en format image normal (H,W,3)
pg_img = np.transpose(pg_img, (1, 0, 2))

# 5. Sauvegarde OpenCV
cv2.imwrite("full_track_seed4.png", cv2.cvtColor(pg_img, cv2.COLOR_RGB2BGR))

print("Image complète exportée → full_track_seed4.png")


env.close()
