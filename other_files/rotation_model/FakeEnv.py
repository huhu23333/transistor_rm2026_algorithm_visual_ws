import math
import time
import random
import math
random.seed(42)

class FakeEnv:
    #def __init__(self, center_x=-100, vx=1000, center_y=2000, vy=300, center_z=500, vz=0, yaw=0.1, vyaw=0.8*math.pi, r=276, init_time=0):
    def __init__(self, center_x=-100, vx=0, center_y=2000, vy=0, center_z=500, vz=0, yaw=0.1, vyaw=0.8*math.pi, r=276, init_time=0, self_round_move = False):
        self.center_x = center_x
        self.vx = vx
        self.center_y = center_y
        self.vy = vy
        self.center_z = center_z
        self.vz = vz
        self.yaw = yaw
        self.vyaw = vyaw
        self.r = r
        self.init_time = init_time
        self.self_round_move = self_round_move
        self.origin_center_x = self.center_x
        self.origin_center_y = self.center_y

    def observeData(self, obs_time, rand_jump = False):
        delta_t = obs_time - self.init_time
        target_yaw = delta_t * self.vyaw + self.yaw
        if self.self_round_move:
            self.center_x = self.origin_center_x + 500 * math.cos(delta_t * 0.3 * math.pi)
            self.center_y = self.origin_center_y + 500 * math.sin(delta_t * 0.3 * math.pi)
        while target_yaw > math.pi / 3:
            target_yaw -= math.pi * 2 / 3
        while target_yaw < -math.pi / 3:
            target_yaw += math.pi * 2 / 3
        if rand_jump:
            if target_yaw - math.pi * 2 / 3 >= -math.pi / 3:
                if random.random() < 0.5:
                    target_yaw -= math.pi * 2 / 3
            if target_yaw + math.pi * 2 / 3 <= math.pi / 3:
                if random.random() < 0.5:
                    target_yaw += math.pi * 2 / 3
        obs_x = self.center_x + self.r * math.sin(target_yaw) + delta_t * self.vx
        obs_y = self.center_y - self.r * math.cos(target_yaw) + delta_t * self.vy
        obs_z = self.center_z + delta_t * self.vz
        obs_yaw = target_yaw
        return [obs_x, obs_y, obs_z, obs_yaw]
    
    def observeDataWithNoise(self, obs_time):
        obs_x, obs_y, obs_z, obs_yaw = self.observeData(obs_time, rand_jump = True)
        distance = math.sqrt(obs_x ** 2 + obs_y ** 2 + obs_z ** 2)
        obs_x += 0.01 * distance * random.normalvariate(0, 1)
        obs_y += 0.02 * distance * random.normalvariate(0, 1)
        obs_z += 0.01 * distance * random.normalvariate(0, 1)
        obs_yaw += max(min(0.00005*distance/(math.pi/2-abs(obs_yaw)),1.5),0) * random.normalvariate(0, 1)
        return [obs_x, obs_y, obs_z, obs_yaw]

    def get_true_params(self, current_time):
        delta_t = current_time - self.init_time
        """ current_yaw = delta_t * self.vyaw + self.yaw
        true_x = self.center_x + self.r * math.sin(current_yaw) + delta_t * self.vx
        true_y = self.center_y - self.r * math.cos(current_yaw) + delta_t * self.vy
        true_z = self.center_z + delta_t * self.vz """
        true_x, true_y, true_z, current_yaw = self.observeData(current_time)
        return {
            'center_x': self.center_x + delta_t * self.vx,
            'center_y': self.center_y + delta_t * self.vy,
            'center_z': self.center_z + delta_t * self.vz,
            'yaw': current_yaw,
            'vyaw': self.vyaw,
            'r': self.r,
            'true_x': true_x,
            'true_y': true_y,
            'true_z': true_z
        }