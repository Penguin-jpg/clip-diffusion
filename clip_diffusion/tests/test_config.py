import pytest
from clip_diffusion.text2image.config import Config
from clip_diffusion.utils.functional import create_schedule


def test_schedules():
    """
    測試schedule建立是否正常
    """

    overview_cut_schedule = (14,) * 200 + (12,) * 200 + (4,) * 400 + (0,) * 200
    inner_cut_schedule = (2,) * 200 + (4,) * 200 + (2,) * 400 + (12,) * 200
    inner_cut_size_power_schedule = (5,) * 1000
    cut_gray_portion_schedule = (0.7,) * 100 + (0.6,) * 100 + (0.45,) * 100 + (0.3,) * 100 + (0,) * 600

    assert Config.num_overview_cuts_schedule == overview_cut_schedule, "diff found at overview_cut_schedule"
    assert Config.num_inner_cuts_schedule == inner_cut_schedule, "diff found at inner_cut_schedule"
    assert Config.inner_cut_size_power_schedule == inner_cut_size_power_schedule, "diff found at inner_cut_size_power_schedule"
    assert Config.cut_gray_portion_schedule == cut_gray_portion_schedule, "diff found at cut_gray_portion_schedule"

    with pytest.raises(AssertionError):
        create_schedule(values=(1, 2, 3), steps=(100, 500))


def test_adjust_settings():
    """
    測試是否有改動設定
    """

    Config.change(
        width=512,
        height=512,
        num_cutout_batches=1,
        chosen_clip_models=("ViT-B/32",),
        clamp_max=0.007,
        tv_scale=50,
        range_scale=100,
        sat_scale=20,
    )

    assert Config.width == 512, "width not set"
    assert Config.height == 512, "height not set"
    assert Config.num_cutout_batches == 1, "num_cutout_batches not set"
    assert Config.chosen_clip_models == ("ViT-B/32",), "chosen_clip_models not set"
    assert Config.grad_threshold == 0.007, "clamp_max not set"
    assert Config.tv_scale == 50, "tv_scale not set"
    assert Config.range_scale == 100, "range_scale not set"
    assert Config.sat_scale == 20, "sat_scale not set"
