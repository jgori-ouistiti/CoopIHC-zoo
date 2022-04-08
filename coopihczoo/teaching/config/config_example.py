task_kwargs = {
    "n_item": 20,
    "inter_trial": 3,
    "n_iter_per_ss": 40,
    "break_length": 1,  # 4000,  # 24*60**2 - inter_trial*n_iter_per_ss
    "n_session": 4,
    "time_before_exam": 1,  # 4000,
    "is_item_specific": False,
    "thr": 0.9,
}

user_kwargs = {"param": (0.01, 0.2)}
