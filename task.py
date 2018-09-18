import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.last_dist = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # travel with least angular motion
        # reach the destination

        #out of bounds penalty and positive reward for staying in bounds
        pos_reward = 1
        if self.sim.done:
            for ii in range(3):
                if self.sim.pose[ii] <= self.sim.lower_bounds[ii]:
                    pos_reward = -10
                    break
                elif self.sim.pose[ii] > self.sim.upper_bounds[ii]:
                    pos_reward = -10
                    break
        
        #distance from target
        dist_target = np.sqrt(np.power(self.sim.pose[:3] - self.target_pos, 2).sum())
        
        correct_direction = min(1 - self.last_dist/(dist_target + 0.001), 5)
        correct_direction = max(correct_direction, -5)
        self.last_dist = dist_target
            
        
        #z_penal = np.abs(self.sim.pose[2] - self.target_pos[2])
        #euler angles cost
        angles = np.sqrt(np.power(self.sim.pose[3:], 2).sum())
        angles_cost = angles/360
        
        # approx dist after action repeats
        approx_dist = np.sqrt(np.power(self.target_pos - self.sim.pose[:3] - self.action_repeat/50 * self.sim.v, 2).sum())
        approx_dist = min(approx_dist, 100)
        approx_dist = max(approx_dist, -100)
        
        reward = -0.001 * dist_target - 0.0001 * angles_cost + pos_reward + correct_direction - 1/approx_dist
        

        # distance and velocity relation. max time to travel a dimension
        #max_time = np.log(0.1 + 
        #   (np.abs(self.target_pos - self.sim.pose[:3]) / (0.1 + np.abs(self.sim.v))).sum()
        #)
        # z velocity reward
        #v_reward = np.abs(self.sim.v[2])
        
        #+ v_reward * 0.001 - 0.0001 * approx_dist
        #- 0.001 * z_penal - max_time - 0.01 * max_time + 
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state