import rustuna


def test_study_best_trial_minimize_direction():
    study = rustuna.create_study(direction="minimize")
    for value in [1.0, 9.0, 3.0]:
        trial = study.ask()
        study.tell(trial.number, value)

    assert study.best_trial.number == 0
    assert study.best_trial.value == 1.0


def test_study_best_trial_maximize_direction():
    study = rustuna.create_study(direction="maximize")
    for value in [1.0, 9.0, 3.0]:
        trial = study.ask()
        study.tell(trial.number, value)

    assert study.best_trial.number == 1
    assert study.best_trial.value == 9.0


def test_study_reuses_sampler_callback_storage_without_json_roundtrip():
    import cProfile
    import pstats

    class CapturingSampler:
        support_joint_sampling = False

        def before_trial(self, ctx, storage):
            self.ctx = ctx
            self.storage = storage

        def after_trial(self, ctx, storage, state, values=None):
            self.ctx = ctx
            self.storage = storage

    sampler = CapturingSampler()
    original = rustuna.create_study(sampler=sampler)
    original.set_user_attr("settings", {"nested": [1, True, "value"]})
    trial = original.ask()
    trial.set_user_attr("payload", {"nested": [1, 2, 3]})
    original.tell(trial.number, 1.0)
    persisted = sampler.storage.get_study(sampler.ctx.study_id)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        reconstructed = rustuna.Study(
            study_id=sampler.ctx.study_id,
            name=persisted.name,
            directions=persisted.directions,
            storage=sampler.storage,
            sampler=rustuna.samplers.RandomSampler(),
        )
        trials = reconstructed.get_trials()
    finally:
        profiler.disable()

    stats = pstats.Stats(profiler)
    json_calls = [
        key
        for key in stats.stats
        if key[0].replace("\\", "/").endswith("json/__init__.py")
        and key[2] in ("loads", "dumps")
    ]
    assert json_calls == [], json_calls
    assert len(trials) == 1
    assert trials[0].user_attrs["payload"] == {"nested": [1, 2, 3]}
    running = original.ask()
    running.set_user_attr("payload", {"new": 4})
    live_trial = reconstructed.get_trials()[-1]
    running.set_user_attr("payload", {"new": 5})
    assert live_trial.user_attrs["payload"] == {"new": 5}
    reconstructed.set_user_attr("from_callback", [5, 6])
    assert original.get_user_attr("from_callback") == [5, 6]
