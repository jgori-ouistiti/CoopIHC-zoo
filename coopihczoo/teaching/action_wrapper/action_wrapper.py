from gym import ActionWrapper
from gym.spaces import Discrete


class AssistantActionWrapper(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(env.action_space["assistant_action"].n)

    def action(self, action):
        return {"assistant_action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action"]
