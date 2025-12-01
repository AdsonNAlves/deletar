# run_trained.py
import numpy as np
from stable_baselines3 import SAC
from drone_env import DroneEnv


if __name__ == "__main__":

    # Carrega modelo
    model = SAC.load("./models/sac_drone_50000_steps.zip")

    # Cria ambiente
    env = DroneEnv()

    for ep in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        final_dist = np.linalg.norm(obs[0:3])
        status = "SUCESSO" if (final_dist < 0.3 and terminated) else "FALHA"

        print(f"Ep {ep+1}: {status} | Reward: {total_reward:.2f} | Steps: {steps} | Dist: {final_dist:.2f}m")

    env.close()
