import numpy as np
from cs285.critics.dqn_critic import DQNCritic


class ArgMaxPolicy(object):

    def __init__(self, critic: DQNCritic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## TODO Done : return the action that maxinmizes the Q-value
        # at the current observation as the output
        q = self.critic.qa_values(observation)
        action = np.argmax(q)
        return action.squeeze()
