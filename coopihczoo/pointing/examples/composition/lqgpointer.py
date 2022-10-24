from coopihc import SimplePointingTask, WrapAsObservationEngine


# # Error linked to LQGpointer
#     # Add a state to the SimplePointingTask to memorize the old position
#     class oldpositionMemorizedSimplePointingTask(SimplePointingTask):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.memorized = None

#         def reset(self, dic={}):
#             super().reset(dic=dic)
#             self.state["oldposition"] = copy.deepcopy(self.state["position"])

#         def user_step(self, *args, **kwargs):
#             self.memorized = copy.deepcopy(self.state["position"])
#             obs, rewards, is_done, _doc = super().user_step(*args, **kwargs)
#             obs["oldposition"] = self.memorized
#             return obs, rewards, is_done, _doc

#         def assistant_step(self, *args, **kwargs):
#             self.memorized = copy.deepcopy(self.state["position"])
#             obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
#             obs["oldposition"] = self.memorized
#             return obs, rewards, is_done, _doc

#     pointing_task = oldpositionMemorizedSimplePointingTask(
#         gridsize=31, number_of_targets=8, mode="gain"
#     )

#     fitts_W = 4e-2
#     fitts_D = 0.8
#     perceptualnoise = 0.2
#     oculomotornoise = 0.2
#     task = ChenEyePointingTask(fitts_W, fitts_D, dimension=1)
#     user = ChenEye(perceptualnoise, oculomotornoise, dimension=1)
#     obs_bundle = SinglePlayUserAuto(task, user, start_at_action=True)

#     class ChenEyeObservationEngineWrapper(WrapAsObservationEngine):
#         def __init__(self, obs_bundle):
#             super().__init__(obs_bundle)

#         def observe(self, game_state):
#             # set observation bundle to the right state and cast it to the right space
#             target = game_state["task_state"]["position"].cast(
#                 self.game_state["task_state"]["targets"]
#             )
#             fixation = game_state["task_state"]["oldposition"].cast(
#                 self.game_state["task_state"]["fixation"]
#             )
#             reset_dic = {"task_state": {"targets": target, "fixation": fixation}}
#             self.reset(dic=reset_dic)

#             # perform the run
#             is_done = False
#             rewards = 0
#             while True:
#                 obs, reward, is_done, _doc = self.step()
#                 rewards += reward
#                 if is_done:
#                     break

#             # cast back to initial space and return
#             obs["task_state"]["fixation"].cast(game_state["task_state"]["oldposition"])
#             obs["task_state"]["targets"].cast(game_state["task_state"]["position"])

#             return game_state, rewards

#     # Define cascaded observation engine
#     cursor_tracker = ChenEyeObservationEngineWrapper(obs_bundle)
#     base_user_engine_specification = [
#         ("turn_index", "all"),
#         ("task_state", "all"),
#         ("user_state", "all"),
#         ("assistant_state", None),
#         ("user_action", "all"),
#         ("assistant_action", "all"),
#     ]
#     default_observation_engine = RuleObservationEngine(
#         deterministic_specification=base_user_engine_specification,
#     )
#     observation_engine = CascadedObservationEngine(
#         [cursor_tracker, default_observation_engine]
#     )

lqg_user = LQGPointer(observation_engine=observation_engine)
unitcdgain = ConstantCDGain(1)
bundle = PlayNone(pointing_task, lqg_user, unitcdgain)
game_state = bundle.reset()
bundle.render("plotext")
