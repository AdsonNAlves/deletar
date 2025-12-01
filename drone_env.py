import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import subprocess
import os

class DroneEnv(gym.Env):
    """Ambiente customizado para treinamento do drone no CoppeliaSim"""

    def __init__(self, scene_file=None, headless=False, max_velocity=100, pwm_scale: float = 1.0, torque_coeff: float = 0.001):
        super(DroneEnv, self).__init__()
        
        # Conecta ao CoppeliaSim
        #self.client = RemoteAPIClient()
        #self.sim = self.client.require('sim')
        # self.sim.setStepping(True)
        
        # --- LÓGICA PARA ABRIR O COPPELIASIM ---
        # "coppeliaSim.sh" deve estar no PATH
        coppelia_exec = "coppeliaSim.sh" 
        
        args = [coppelia_exec]
        if headless:
            args.append("-h")  # Modo headless (sem GUI)
        
        if scene_file:
            args.append(scene_file)  # Abre já carregando a cena
            
        print(f"Iniciando CoppeliaSim: {' '.join(args)}")
        sim_env = os.environ.copy()
        sim_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        sim_env.pop("QT_PLUGIN_PATH", None)

        try:
            self.sim_process = subprocess.Popen(args, env=sim_env)
        except FileNotFoundError:
            raise RuntimeError(f"Erro: '{coppelia_exec}' não encontrado! Adicione ao PATH ou edite o caminho em drone_env.py")

        # Loop de tentativa de conexão (aguarda o simulador abrir)
        print("Aguardando conexão com a API (pode levar alguns segundos)...")
        
        connected = False
        
        for i in range(20):  # Tenta por ~40 segundos
            try:
                self.client = RemoteAPIClient()
                self.sim = self.client.require('sim')
                self.sim.getSimulationTime()  # Teste de vida
                connected = True
                print("Conectado ao CoppeliaSim!")
                break
            except Exception:
                time.sleep(2)
        
        if not connected:
            self.sim_process.terminate()
            raise ConnectionError("Não foi possível conectar ao CoppeliaSim ZMQ API.")
        # ---------------------------------------
        
        # Carrega handles dos objetos
        self._load_handles()

        self.joints_max_velocity = max_velocity
        self.joints_min_velocity = 5.0

        # Scaling/tuning for PWM mapping and yaw torque sensitivity
        self.pwm_scale = float(pwm_scale)
        self.torque_coeff = float(torque_coeff)

        # Número de ações e observações
        self.num_act = len(self.joints)  # 4 propellers
        
        self.sim.startSimulation()
        self.sim.setStepping(True)
        time.sleep(0.1)
        
        self.num_obs = len(self._get_observation())

        # Para simulação (será reiniciada no reset)
        self.sim.stopSimulation()
        time.sleep(0.2)

        # Define espaço de ação (ajuste conforme seu num_act e limites)
        #self.action_space = spaces.Box(
        #low=np.zeros(self.num_act, dtype=np.float32),
        #high=np.full(self.num_act, self.joints_max_velocity, #dtype=np.float32),
        #dtype=np.float32)
        
        # Espaço de ação normalizado [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_act,),
            dtype=np.float32
        )
        
        # Define espaço de observação (ajuste conforme seu lst_obs) ATENÇÃO (REFATORAR MUDANDO -INF E INF PELO VALOR MÁXIMO E MÍNIMO ACEITO) 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        
        # Parâmetros de episódio
        self.max_steps = 250
        self.current_step = 0
        self.last_pos = None
        self.last_distance = None  # Guarda distância anterior para calcular progresso
        self.rel_pos = None # Posição relativa ao target
        
        # Ticks para posição inicial aleatória
        self.x_ticks, self.y_ticks, self.z_ticks, self.ang_ticks = self._create_discretized_uniform_list()
        
    def _load_handles(self):
        """Carrega todos os handles necessários"""
        self.quad_handle = self.sim.getObject('/Quadricopter')
        self.target_handle = self.sim.getObject('/Quadricopter_target')
        
        propeller_names = [
            '/Quadricopter_propeller_respondable1',
            '/Quadricopter_propeller_respondable2',
            '/Quadricopter_propeller_respondable3',
            '/Quadricopter_propeller_respondable4'
        ]
        self.propellers = [self.sim.getObject(name) for name in propeller_names]
        
        joint_names = [
            '/Quadricopter_propeller_joint1',
            '/Quadricopter_propeller_joint2',
            '/Quadricopter_propeller_joint3',
            '/Quadricopter_propeller_joint4'
        ]
        self.joints = [self.sim.getObject(name) for name in joint_names]
        
        force_sensor_names = [
            '/Quadricopter_propeller1',
            '/Quadricopter_propeller2',
            '/Quadricopter_propeller3',
            '/Quadricopter_propeller4'
        ]
        self.force_sensor_handles = [self.sim.getObject(name) for name in force_sensor_names]
        
        corner_names = [f'/corner{i}' for i in range(1, 8)]
        self.corner_handles = [self.sim.getObject(name) for name in corner_names]

        # Nomes dos sensores ultrassônicos
        ultrasonic_names = [
            "drone_ultrasonic0", "drone_ultrasonic1", "drone_ultrasonic2", "drone_ultrasonic3",
            "drone_ultrasonic4", "drone_ultrasonic5", "drone_ultrasonic6", "drone_ultrasonic7",
            "drone_ultrasonic_up", "drone_ultrasonic_down"
        ]
        self.ultrasonic_handles = [self.sim.getObject(f'/Quadricopter/Quadricopter_base/{name}') for name in ultrasonic_names]

    
    def _create_discretized_uniform_list(self):
        """Gera posições iniciais discretizadas"""
        x_ticks = np.round(np.linspace(-9.00, 12.00, 11), 2)
        y_ticks = np.round(np.linspace(-8.00, 13.00, 11), 2)
        z_ticks = np.round(np.linspace(0.5, 2.5, 7), 2)
        ang_ticks = np.round(np.linspace(-0.785, 0.785, 15), 2)
        return x_ticks, y_ticks, z_ticks, ang_ticks
    
    def reset(self, seed=None, options=None):
        """Reinicia o episódio"""
        super().reset(seed=seed)
        
        # Para e reinicia simulação
        self.sim.stopSimulation()
        time.sleep(0.5)
        self.sim.startSimulation()
        
        self.sim.setStepping(True)
        time.sleep(0.1)
        
        # Define posição e orientação inicial aleatória
        init_x = np.random.choice(self.x_ticks)
        init_y = np.random.choice(self.y_ticks)
        init_z = np.random.choice(self.z_ticks)
        
        init_roll = np.random.choice(self.ang_ticks)
        init_pitch = np.random.choice(self.ang_ticks)
        init_yaw = np.random.choice(self.ang_ticks)
        
        self.sim.setObjectPosition(self.quad_handle, -1, [init_x, init_y, init_z])
        self.sim.setObjectOrientation(self.quad_handle, -1, [init_roll, init_pitch, init_yaw])
        
        self.current_step = 0
        self.last_pos = np.array(self.sim.getObjectPosition(self.quad_handle, -1))
        
        # Retorna observação inicial
        obs = self._get_observation()
        self.last_distance = np.linalg.norm(obs[0:3])  # Distância inicial ao target
        
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Executa uma ação e retorna (obs, reward, terminated, truncated, info)"""
        
        # Aplica ação (PWM nos propellers)
        self._apply_action(action)
        
        # Avança simulação
        self.sim.step()
        self.current_step += 1
        
        # Coleta observação
        obs = self._get_observation()
        
        # Calcula recompensa
        reward = self._compute_reward(obs)
        
        # Verifica condições de término
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step,
            'position': self.sim.getObjectPosition(self.quad_handle, -1)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Coleta vetor de observação"""
        lst_obs = []
        
        # Posição relativa ao alvo
        self.rel_pos = self.sim.getObjectPosition(self.quad_handle, self.target_handle)
        lst_obs.extend(self.rel_pos)
        
        # Matriz de rotação
        m = self.sim.getObjectMatrix(self.quad_handle, -1)
        g11, g12, g13 = m[0], m[1], m[2]
        g21, g22, g23 = m[4], m[5], m[6]
        g31, g32, g33 = m[8], m[9], m[10]
        lst_obs.extend([g11, g12, g13, g21, g22, g23, g31, g32, g33])
        
        # Velocidades
        vel, ang = self.sim.getObjectVelocity(self.quad_handle)
        lst_obs.extend(ang)  # velocidades angulares
        lst_obs.extend(vel)  # velocidades lineares
        
        return np.array(lst_obs, dtype=np.float32)
    
    def _apply_action(self, action):
        """Aplica PWM nos propellers"""
        
        t = self.sim.getSimulationTime()
        
        count = 1
        
        for k, (propeller, joint, pwm) in enumerate(zip(self.propellers, self.joints, action)):

            # Mapeia [-1, 1] para [pwm_min, pwm_max]
            pwm = self.joints_min_velocity + (pwm + 1.0) / 2.0 * (self.joints_max_velocity - self.joints_min_velocity)
            pwm = np.clip(pwm, self.joints_min_velocity, self.joints_max_velocity)
            
            pwm = float(pwm) * self.pwm_scale
            force = 1.5618e-4*pwm*pwm + 1.0395e-2*pwm + 0.13894
            
            #if force_zero:
            #    force = 0
            
            # Matriz de rotação
            rot_matrix = self.sim.getObjectMatrix(self.force_sensor_handles[k], -1)
            rot_matrix[3] = rot_matrix[7] = rot_matrix[11] = 0.0
            
            # Força no eixo Z
            z_force = np.array([0.0, 0.0, force])
            applied_force = list(self._transform_vector(rot_matrix, z_force))
            
            # Torque alternado
            if count % 2:
                z_torque = np.array([0.0, 0.0, -self.torque_coeff * pwm])
            else:
                z_torque = np.array([0.0, 0.0, self.torque_coeff * pwm])
            
            applied_torque = list(self._transform_vector(rot_matrix, z_torque))
            
            self.sim.addForceAndTorque(propeller, applied_force, applied_torque)
            self.sim.setJointPosition(joint, t * 10)
            count += 1
    
    def _transform_vector(self, matrix, vector):
        """Multiplica matriz de rotação por vetor"""
        mat3 = np.array([
            [matrix[0], matrix[1], matrix[2]],
            [matrix[4], matrix[5], matrix[6]],
            [matrix[8], matrix[9], matrix[10]]
        ])
        return np.dot(mat3, vector)
    
    #def _compute_reward(self, obs):
    #    """Calcula recompensa baseada no estado"""
    #    weight_dict = {
    #        'r_alive': 1.5,
    #        'radius': -1.00,
    #        'roll_vel': -0.05,
    #        'pitch_vel': -0.05,
    #        'yaw_vel': -0.1
    #   }
    #    
    #    reward = (weight_dict['r_alive'] + 
    #              weight_dict['radius'] * (abs(obs[0]) + abs(obs[1]) + abs(obs[2])) + 
    #              weight_dict['roll_vel'] * abs(obs[12]) + 
    #              weight_dict['pitch_vel'] * abs(obs[13]) + 
    #              weight_dict['yaw_vel'] * abs(obs[14]))
    #    
    #    return reward
    
    #def _compute_reward(self, obs):
    #    """Calcula recompensa baseada no estado"""
    #    
    #    # Posição relativa ao alvo
    #    dist = np.linalg.norm(obs[0:3])
    #    
    #    # Velocidades angulares
    #    roll_rate = abs(obs[12])
    #    pitch_rate = abs(obs[13])
    #    yaw_rate = abs(obs[14])
    #    
    #    # Velocidades lineares
    #    lin_vel = np.linalg.norm(obs[15:18])
    #    
    #    # Recompensa progressiva
    #    # Recompensa por sobrevivência (incentiva episódios longos)
    #    reward = 1.0
    #    
    #    # Recompensa por proximidade (exponencial para incentivar chegar perto)
    #    dist_reward = np.exp(-0.5 * dist)  # ~1.0 quando perto, ~0.0 quando longe
    #    reward += 2.0 * dist_reward
    #    
    #    # Penalidade suave por velocidades angulares (estabilidade)
    #    stability_penalty = 0.1 * (roll_rate + pitch_rate + yaw_rate)
    #    reward -= stability_penalty
    #    
    #    # Penalidade por movimento excessivo
    #    velocity_penalty = 0.05 * lin_vel
    #    reward -= velocity_penalty
    #    
    #    # Bônus por estar muito estável E perto do alvo
    #    if dist < 0.5 and (roll_rate + pitch_rate + yaw_rate) < 0.1:
    #        reward += 5.0
    #    
    #    return reward

    def _compute_reward(self, obs):
        """
        Calcula recompensa focada em RAPIDEZ + ESTABILIDADE para alcançar o target
        
        Filosofia:
        - Penaliza tempo (cada step custa)
        - Recompensa progresso em direção ao target
        - Bônus massivo por alcançar o target
        - Penalidade por instabilidade (mas não domina a recompensa)
        """
        
        # === 1. DISTÂNCIA AO TARGET ===
        dist = np.linalg.norm(obs[0:3])
        
        # === 2. PROGRESSO (redução de distância) ===
        # Calcula quanto o drone se aproximou desde o último step
        progress = self.last_distance - dist
        self.last_distance = dist  # Atualiza para próximo step
        
        # === 3. VELOCIDADES (estabilidade) ===
        roll_rate = abs(obs[12])
        pitch_rate = abs(obs[13])
        yaw_rate = abs(obs[14])
        angular_instability = roll_rate + pitch_rate + yaw_rate
        
        lin_vel = np.linalg.norm(obs[15:18])
        
        # ====================================
        # FUNÇÃO DE RECOMPENSA OTIMIZADA
        # ====================================
        
        reward = 0.0
        
        # 1️⃣ PENALIDADE POR TEMPO (incentiva rapidez)
        # -0.1 por step → Em 250 steps acumula -25.0
        # Drone precisa compensar isso chegando rápido no target
        reward -= 0.1
        
        # 2️⃣ RECOMPENSA POR PROGRESSO (movimento em direção ao target)
        # +10.0 por metro reduzido → Incentiva movimento ativo
        if progress > 0:  # Se aproximou
            reward += 10.0 * progress
        else:  # Se afastou
            reward += 5.0 * progress  # Penalidade menor (exploração)
        
        # 3️⃣ RECOMPENSA POR PROXIMIDADE (gradiente forte perto do target)
        # Cresce exponencialmente quando dist < 2m
        if dist < 2.0:
            reward += 5.0 * np.exp(-dist)  # ~5.0 em 0m, ~1.8 em 1m, ~0.7 em 2m
        
        # 4️⃣ PENALIDADE POR INSTABILIDADE (controle suave)
        # Não domina, mas incentiva estabilidade
        reward -= 0.05 * angular_instability
        reward -= 0.02 * lin_vel
        
        # 5️⃣ BÔNUS MASSIVO POR ALCANÇAR O TARGET
        if dist < 0.3:  # Dentro de 30cm do target
            if angular_instability < 0.1:  # E estável
                reward += 100.0  # JACKPOT! Episódio de sucesso
                print(f"🎯 TARGET ALCANÇADO! Step: {self.current_step}, Dist: {dist:.3f}m")
        
        # 6️⃣ BÔNUS POR PROXIMIDADE ESTÁVEL (zona intermediária)
        elif dist < 1.0 and angular_instability < 0.2:
            reward += 10.0  # Incentiva manter-se perto
        
        return reward

    def _check_termination(self):
        """Verifica se episódio deve terminar"""
        # Verifica alcance do target
        dist_to_target = np.linalg.norm(self.rel_pos)
        if dist_to_target < 0.3:
            return True
        
        # Verifica limites
        if self._is_out_of_bounds():
            return True
        
        # Verifica colisão
        if self._check_collision():
            return True
        
        return False
    
    def _is_out_of_bounds(self):
        """Verifica se drone saiu dos limites"""
        pos = self.sim.getObjectPosition(self.quad_handle, -1)
        x, y, z = pos
        
        pts = [self.sim.getObjectPosition(h, -1) for h in self.corner_handles]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        limits = {
            'x_min': min(xs), 'x_max': max(xs),
            'y_min': min(ys), 'y_max': max(ys),
            'z_min': 0.02, 'z_max': 3.00
        }
        
        out_xyz = not (limits['x_min'] <= x <= limits['x_max'] and 
                       limits['y_min'] <= y <= limits['y_max'] and 
                       limits['z_min'] <= z <= limits['z_max'])
        
        return out_xyz
    
    def _check_collision(self):
        """Verifica colisão com objetos externos"""
        drone_tree = self.sim.getObjectsInTree(self.quad_handle, self.sim.object_shape_type, 0)
        all_shapes = self.sim.getObjectsInTree(self.sim.handle_scene, self.sim.object_shape_type, 0)
        external_objects = [obj for obj in all_shapes if obj not in drone_tree]
        
        drone_pos = np.array(self.sim.getObjectPosition(self.quad_handle, -1))
        
        for obj in external_objects:
            try:
                respondable = self.sim.getObjectInt32Param(obj, self.sim.shapeintparam_respondable)
                if not respondable:
                    continue
            except:
                continue
            
            obj_pos = np.array(self.sim.getObjectPosition(obj, -1))
            distance = np.linalg.norm(drone_pos - obj_pos)
            
            if distance > 5.0:
                continue
            
            if self.sim.checkCollision(self.quad_handle, obj):
                if distance < 5.0:
                    return True
        
        return False
    
#    def close(self):
#        """Finaliza simulação"""
#        self.sim.stopSimulation()

    def close(self):
        """Finaliza simulação e mata o processo do CoppeliaSim"""
        try:
            self.sim.stopSimulation()
        except:
            pass
            
        if hasattr(self, 'sim_process'):
            print("Fechando CoppeliaSim...")
            self.sim_process.terminate()
            try:
                self.sim_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sim_process.kill()