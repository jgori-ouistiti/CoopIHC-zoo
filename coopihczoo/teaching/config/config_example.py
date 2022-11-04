import numpy


n_item = 20

# Some arguments are only for the evaluate_naive_implementation implementation
task_kwargs = dict(
    n_item=20,
    inter_trial=3,
    n_iter_per_ss=100,
    break_length=60 * 15,  # 4000,  # 24*60**2 - inter_trial*n_iter_per_ss
    time_before_exam=0.,  # 60 * 60,  # 4000,
    is_item_specific=False,
    n_session=2,
    thr=0.9,
)

# Only for evaluate_naive_implementation implementation
user_kwargs = {
    "param": (0.01, 0.2)
}


user_per_item_kwargs = dict(
    param=numpy.concatenate(
        (numpy.full((n_item, 1), 0.01), numpy.full((n_item, 1), 0.2)), axis=1
    )
)

user_act_r_kwargs = dict(
    param=[-0.704, 0.255, 0.217, 0.177]  # tau, s, c, a
    # https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog0000_14
)

teaching_orchestrator_kwargs = dict(
    n_iter_per_ss=[100, 100],
    breaks=[30],
    time_before_exam=60,
    inter_trial=3,
    exam_threshold=0.9,
)

random_teacher_kwargs = dict()

leitner_kwargs = dict(delay_factor=2, delay_min=1)
