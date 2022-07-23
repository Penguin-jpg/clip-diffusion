import unittest
from clip_diffusion.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def tearDown(self):
        return super().tearDown()

    def test_schedules(self):
        """
        測試schedule建立是否正常
        """

        print("testing schedules")

        overview_cut_schedule = (14,) * 200 + (12,) * 200 + (4,) * 400 + (0,) * 200
        inner_cut_schedule = (2,) * 200 + (4,) * 200 + (2,) * 400 + (12,) * 200
        inner_cut_size_power_schedule = (5,) * 1000
        cut_gray_portion_schedule = (0.7,) * 100 + (0.6,) * 100 + (0.45,) * 100 + (0.3,) * 100 + (0,) * 600

        self.assertTupleEqual(self.config.overview_cut_schedule, overview_cut_schedule, "diff found at overview_cut_schedule")
        self.assertTupleEqual(self.config.inner_cut_schedule, inner_cut_schedule, "diff found at inner_cut_schedule")
        self.assertTupleEqual(
            self.config.inner_cut_size_power_schedule,
            inner_cut_size_power_schedule,
            "diff found at inner_cut_size_power_schedule",
        )
        self.assertTupleEqual(
            self.config.cut_gray_portion_schedule, cut_gray_portion_schedule, "diff found at cut_gray_portion_schedule"
        )

    def test_seed(self):
        """
        測試是否有拿到種子
        """

        print("testing seed")

        self.assertIsNotNone(self.config.seed, "seed is None")

    def test_chosen_clip_models(self):
        """
        測試chosen_clip_models是否有正常判斷
        """

        print("testing chosen clip models")

        with self.assertRaises(AssertionError):
            self.config.adjust_settings(chosen_clip_models=("ABC"))


if __name__ == "__main__":
    unittest.main()
