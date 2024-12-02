import numpy as np


class RewardCalculator(object):
    def __init__(self):
        self.job_dags = set()
        self.prev_time = 0

    def get_reward(self, job_dags, curr_time):
        reward = 0

        # add new job into the store of jobs
        for job_dag in job_dags:
            self.job_dags.add(job_dag)

        reward -= (curr_time - self.prev_time) / 1.0

        self.prev_time = curr_time

        return reward

    def reset(self):
        self.job_dags.clear()
        self.prev_time = 0
