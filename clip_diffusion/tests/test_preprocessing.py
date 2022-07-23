from clip_diffusion.preprocessing import _get_text_and_weight, prompts_preprocessing


def test_text_and_weight():
    """
    測試文字和權重是否有正確分離
    """

    text1, weight1 = _get_text_and_weight("a cute golden retriever, trending on artstation.")
    text2, weight2 = _get_text_and_weight("a robot dog:5.")
    assert text1 == "a cute golden retriever, trending on artstation.", "failed to split text from prompt without weight"
    assert weight1 == 1.0, "failed to automatically append weight"
    assert text2 == "a robot dog", "failed to split text from prompt with weight"
    assert weight2 == 5.0, "failed to split weight from prompt with weight"


def test_prompt_translation():
    """
    測試prompt_preprocessing
    """

    print("testing prompt translation")

    english_prompts = ["A castle on the hill."]
    chinese_prompts = ["山上的一座城堡。"]

    assert english_prompts == prompts_preprocessing(english_prompts), "failed to translate english prompt"
    assert prompts_preprocessing(chinese_prompts) == english_prompts, "failed to translate chinese prompt"
