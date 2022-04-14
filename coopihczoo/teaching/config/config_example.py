import numpy


n_item = 20

task_kwargs = dict(
    n_item=20,
    inter_trial=3,
    n_iter_per_ss=100,
    break_length=3,  # 4000,  # 24*60**2 - inter_trial*n_iter_per_ss
    time_before_exam=3,  # 4000,
    is_item_specific=False,
    n_session=2,
    thr=0.9,
)


user_kwargs = dict(param=[0.01, 0.2])

user_per_item_kwargs = dict(
    param=numpy.concatenate(
        (numpy.full((n_item, 1), 0.01), numpy.full((n_item, 1), 0.2)), axis=1
    )
)

teaching_orchestrator_kwargs = dict(
    n_iter_per_ss=[100, 100], breaks=[3], time_before_exam=3, exam_threshold=0.9
)

random_teacher_kwargs = dict()

leitner_kwargs = dict(delay_factor=2, delay_min=1)
