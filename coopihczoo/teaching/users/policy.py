from coopihc import BasePolicy


class UserPolicy(BasePolicy):
    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        item = agent_observation["task_state"]["item"]
        reward = 0

        _action_value = int(
            self.state.recall_probabilities[int(item)] > self.get_rng().random()
        )

        return _action_value, reward

    def reset(self, random=True):
        self.action_state["action"] = 0

