import pytest
from clip_diffusion.config import Config


config = Config()


def test_schedules():
    """
    測試schedule建立是否正常
    """

    overview_cut_schedule = (14,) * 200 + (12,) * 200 + (4,) * 400 + (0,) * 200
    inner_cut_schedule = (2,) * 200 + (4,) * 200 + (2,) * 400 + (12,) * 200
    inner_cut_size_power_schedule = (5,) * 1000
    cut_gray_portion_schedule = (0.7,) * 100 + (0.6,) * 100 + (0.45,) * 100 + (0.3,) * 100 + (0,) * 600

    assert config.overview_cut_schedule == overview_cut_schedule, "diff found at overview_cut_schedule"
    assert config.inner_cut_schedule == inner_cut_schedule, "diff found at inner_cut_schedule"
    assert config.inner_cut_size_power_schedule == inner_cut_size_power_schedule, "diff found at inner_cut_size_power_schedule"
    assert config.cut_gray_portion_schedule == cut_gray_portion_schedule, "diff found at cut_gray_portion_schedule"

    with pytest.raises(AssertionError):
        config.create_schedule(values=(1, 2, 3), steps=(100, 500))


def test_seed():
    """
    測試是否有拿到種子
    """

    assert config.seed is not None, "seed is None"
