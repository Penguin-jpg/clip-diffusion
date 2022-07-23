import unittest
from clip_diffusion.preprocessing import _get_text_and_weight, prompts_preprocessing


class TestPreprocessing(unittest.TestCase):
    def test_text_and_weight(self):
        """
        測試文字和權重是否有正確分離
        """

        print("testing text and weight")

        text1, weight1 = _get_text_and_weight("a cute golden retriever, trending on artstation.")
        text2, weight2 = _get_text_and_weight("a robot dog:5.")
        self.assertEqual(text1, "a cute golden retriever, trending on artstation.")
        self.assertEqual(weight1, 1.0)
        self.assertEqual(text2, "a robot dog")
        self.assertEqual(weight2, 5.0)

    def test_prompt_translation(self):
        """
        測試prompt_preprocessing
        """

        print("testing prompt translation")

        english_prompts = ["A castle on the hill."]
        chinese_prompts = ["山上的一座城堡。"]

        self.assertListEqual(english_prompts, prompts_preprocessing(english_prompts))
        self.assertListEqual(english_prompts, prompts_preprocessing(chinese_prompts))


if __name__ == "__main__":
    unittest.main()
