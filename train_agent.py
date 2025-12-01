from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
from drone_env import DroneEnv  

#tensorboard --logdir ./sac_drone_tensorboard/ --port 6006

if __name__ == "__main__":

    # Configurações de execução
    SCENE_PATH = "/home/adson/Desktop/unicamp/new_swarm/multiagent/scenario/swarm/scenario_empty_swarm.ttt"
    HEADLESS_MODE = True  # Mude para True para treinar em background (sem abrir janela)

    # Cria ambiente 
    env = DroneEnv(
        scene_file=SCENE_PATH,
        headless=HEADLESS_MODE
    )

    # Callback para salvar modelo periodicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='sac_drone',
        save_replay_buffer=True
    )

    # Cria modelo SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=5_000_000,
        learning_starts=10000,
        batch_size=4000,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,

        # Archs
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],          # Policy: (64, tanh, 64, tanh)
                qf=[256, 256]         # Q-function: (256, relu, 256, relu)
            ),
            activation_fn=torch.nn.Tanh,  # Aplica tanh em TODAS as camadas
            # log_std_init=-3,            # Controla exploração inicial (opcional)
        ),

        verbose=1,
        tensorboard_log="./sac_drone_tensorboard/",
        device="auto"  # Usa GPU se disponível
    )

    # Treina
    model.learn(
        total_timesteps=1000,   # 500_000,
        callback=checkpoint_callback,
        log_interval=10,
        reset_num_timesteps=False
        # progress_bar=True # Mostra barra de progresso
    )

    # Salva modelo final
    model.save("sac_drone_final")