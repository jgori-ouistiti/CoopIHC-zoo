# The point of this example is to show how different modules can be composed in CoopIHC.
# We are going to create a complex observation engine which uses a foveal vision model to track the cursor movement.
# The foveal vision model needs the current as well as last position in the task state, so we start by adapting the existing task.

from coopihczoo.pointing.envs import SimplePointingTask
import copy


# Add a state to the SimplePointingTask to memorize the old position
class oldpositionMemorizedSimplePointingTask(SimplePointingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memorized = None

    def reset(self, dic={}):
        super().reset(dic=dic)
        self.state["oldposition"] = copy.deepcopy(self.state["position"])

    def user_step(self, *args, **kwargs):
        self.memorized = copy.deepcopy(self.state["position"])
        obs, rewards, is_done = super().user_step(*args, **kwargs)
        obs["oldposition"] = self.memorized
        return obs, rewards, is_done

    def assistant_step(self, *args, **kwargs):
        self.memorized = copy.deepcopy(self.state["position"])
        obs, rewards, is_done = super().assistant_step(*args, **kwargs)
        obs["oldposition"] = self.memorized
        return obs, rewards, is_done


pointing_task = oldpositionMemorizedSimplePointingTask(
    gridsize=31, number_of_targets=8, mode="position"
)


from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye

from coopihc.bundle.Bundle import Bundle

fitts_W = 4e-2
fitts_D = 0.8
perceptualnoise = 0.2
oculomotornoise = 0.2
task = ChenEyePointingTask(fitts_W, fitts_D, dimension=1)
user = ChenEye(perceptualnoise, oculomotornoise, dimension=1)
obs_bundle = Bundle(task=task, user=user)


from coopihc.observation.WrapAsObservationEngine import WrapAsObservationEngine


class ChenEyeObservationEngineWrapper(WrapAsObservationEngine):
    def __init__(self, obs_bundle):
        super().__init__(obs_bundle)

    def observe(self, game_state):

        # Deal with the case where the cursor is in the same position as the target. This is needed to have a non singular matrix (i.e. a matrix that can be inverted) for
        if (
            game_state["task_state"]["position"]
            == game_state["task_state"]["oldposition"]
        ):
            return game_state, -1

        # set observation bundle to the right state and cast it to the right space
        target = game_state["task_state"]["position"].cast(
            self.game_state["task_state"]["target"]
        )
        fixation = game_state["task_state"]["oldposition"].cast(
            self.game_state["task_state"]["fixation"]
        )

        reset_dic = {"task_state": {"target": target, "fixation": fixation}}
        self.reset(dic=reset_dic, turn=0)
        # perform the run
        is_done = False
        rewards = 0
        while True:
            obs, reward_dic, is_done = self.step()
            rewards += sum(reward_dic.values())
            if is_done:
                break

        # cast back to initial space and return
        obs["task_state"]["fixation"].cast(game_state["task_state"]["oldposition"])
        obs["task_state"]["target"].cast(game_state["task_state"]["position"])

        return game_state, rewards


from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.CascadedObservationEngine import CascadedObservationEngine
from coopihc.observation.utils import base_user_engine_specification

# Define cascaded observation engine
cursor_tracker = ChenEyeObservationEngineWrapper(obs_bundle)

default_observation_engine = RuleObservationEngine(
    deterministic_specification=base_user_engine_specification,
)
observation_engine = CascadedObservationEngine(
    [cursor_tracker, default_observation_engine]
)


from coopihczoo.pointing.users import CarefulPointer
from coopihczoo.pointing.assistants import BIGGain

binary_user = CarefulPointer(override_observation_engine=(observation_engine, {}))
BIGpointer = BIGGain()

bundle = Bundle(task=pointing_task, user=binary_user, assistant=BIGpointer)
game_state = bundle.reset(turn=1)
bundle.render("plotext")
reward_list = []
while True:
    obs, rewards, is_done = bundle.step()
    reward_list.append(rewards)
    bundle.render("plotext")
    if is_done:
        break
