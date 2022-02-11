from coopihc import InteractionTask, State, StateElement, autospace


class Env(InteractionTask):
    def __init__(self, N, *args, **kwargs):
        self.N = N
        self.state = State()
        self.state["visible_graph"] = StateElement(
            0, autospace([i for i in range(1, 2 ** N)])
        )

    def reset(self):
        self.state["visible_graph"][:] = 0

    def user_step(self, *args, **kwargs):
        return self.state, -1, False

    def assistant_step(self, *args, **kwargs):
        self.state["visible_graph"][:] = self.assistant_action
