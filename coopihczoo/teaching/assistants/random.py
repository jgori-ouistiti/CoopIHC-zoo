from coopihc import BaseAgent, State, BasePolicy, cat_element


class RandomTeacher(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self):

        n_item = int(self.bundle.game_state.task_state.n_item[0, 0])

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = BasePolicy(action_state=action_state)

        # Use default
        observation_engine = None
        inference_engine = None

        self.attach_policy(agent_policy)
        self.attach_observation_engine(observation_engine)
        self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """
        pass
