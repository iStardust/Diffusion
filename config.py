CONFIG = {
    "num_timesteps": 1000,
    "lr": 2e-4,
    "bs": 10,
    "epochs": 50,
    "lr_step": 10,
    "category": "airplane",  # choose among airplane, chair and table
    "num_workers": 4,
    "beta_min": 0.0001,
    "beta_max": 0.02,
    "gen_bs": 1,  # 一次生成几个模型
    "gen_sample": 6000,  # 生成时采样点的数量
}
