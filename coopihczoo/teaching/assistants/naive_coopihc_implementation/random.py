from coopihc import BaseAgent, State, BasePolicy, cat_element


class RandomTeacher(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self):

        n_item = self.parameters["n_item"]

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = BasePolicy(action_state=action_state)

        # Use default
        observation_engine = None
        inference_engine = None

        self._attach_policy(agent_policy)
        self._attach_observation_engine(observation_engine)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        pass
