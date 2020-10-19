import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Push_StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._pushed = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.5
        mujoco_env.MujocoEnv.__init__(self, 'pushing_striker_update.xml', 5)

    def step(self, a):
        print("object_xpos: ",self.get_body_com("object"))
        print("goal_xpos: ", self.get_body_com("goal"))
        print("pushing_hand_1_xpos: ", self.get_body_com("pushing_hand_1"))
        print("pushing_hand_2_xpos: ", self.get_body_com("pushing_hand_2"))
        print("pushing_hand_3_xpos: ", self.get_body_com("pushing_hand_3"))

        vec_1 = self.get_body_com("object") - self.get_body_com("pushing_hand_1")
        vec_2 = self.get_body_com("object") - self.get_body_com("pushing_hand_2")
        vec_3 = self.get_body_com("object") - self.get_body_com("pushing_hand_3")
        vec_4 = self.get_body_com("object") - self.get_body_com("goal")

        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_4))
        print("np.linalg.norm(vec_1): ", np.linalg.norm(vec_1))
        print("np.linalg.norm(vec_2): ", np.linalg.norm(vec_2))
        print("np.linalg.norm(vec_3): ", np.linalg.norm(vec_3))
        if np.linalg.norm(vec_1) < self.strike_threshold or \
                np.linalg.norm(vec_2) < self.strike_threshold or np.linalg.norm(vec_3) < self.strike_threshold:
            min_vec = None
            max_val = 0
            self._pushed = True
            if np.linalg.norm(vec_1) < self.strike_threshold:
                #self._strike_pos = self.get_body_com("tips_arm")
                if max_val < abs(self.strike_threshold-np.linalg.norm(vec_1)):
                    max_val = abs(self.strike_threshold-np.linalg.norm(vec_1))
                    min_vec = "vec_1"

            if np.linalg.norm(vec_2) < self.strike_threshold:
                #self._strike_pos = self.get_body_com("tips_arm")
                if max_val < abs(self.strike_threshold-np.linalg.norm(vec_2)):
                    max_val = abs(self.strike_threshold-np.linalg.norm(vec_2))
                    min_vec = "vec_2"

            if np.linalg.norm(vec_3) < self.strike_threshold:
                #self._strike_pos = self.get_body_com("tips_arm")
                if max_val < abs(self.strike_threshold-np.linalg.norm(vec_3)):
                    max_val = abs(self.strike_threshold-np.linalg.norm(vec_3))
                    min_vec = "vec_3"

            if min_vec == "vec_1":
                self._strike_pos = self.get_body_com("pushing_hand_1")
            elif min_vec == "vec_2":
                self._strike_pos = self.get_body_com("pushing_hand_2")
            elif min_vec == "vec_3":
                self._strike_pos = self.get_body_com("pushing_hand_3")

        if self._pushed:
            vec_5 = self.get_body_com("object") - self._strike_pos
            reward_near = - np.linalg.norm(vec_5)
        else:
            reward_near = - min(np.linalg.norm(vec_1), np.linalg.norm(vec_2), np.linalg.norm(vec_3))

        """vec_1 = self.get_body_com("object") - self.get_body_com("pushing_hand_2")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._pushed = True
            self._strike_pos = self.get_body_com("pushing_hand_2")

        if self._pushed:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = - np.linalg.norm(vec_3)
        else:
            reward_near = - np.linalg.norm(vec_1)
        """
        reward_dist = - np.linalg.norm(self._min_strike_dist)
        reward_ctrl = - np.square(a).sum()
        reward = 3 * reward_dist + 0.3 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        #notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .1)
        #done = not notdone
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._pushed = False
        self._strike_pos = None

        """qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        while True:
            self.goal = np.concatenate([
                    self.np_random.uniform(low=0.15, high=0.7, size=1),
                    self.np_random.uniform(low=0.1, high=1.0, size=1)])
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[7:] = 0"""
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("pushing_hand_1"),
            self.get_body_com("pushing_hand_2"),
            self.get_body_com("pushing_hand_3"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
"""class Push_StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pushing_striker.xml', 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent"""