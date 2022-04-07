task_kwargs = dict(
    n_item=20,
    inter_trial=3,
    n_iter_per_ss=100,
    break_length=3,  # 4000,  # 24*60**2 - inter_trial*n_iter_per_ss
    n_session=1,
    time_before_exam=3,  # 4000,
    is_item_specific=False,
    thr=0.9,
)

user_kwargs = dict(
    param=(0.01, 0.2),
    seed=123
)

random_teacher_kwargs = dict(
    seed=123
)

leitner_kwargs = dict(
    delay_factor=2,
    delay_min=1
)
