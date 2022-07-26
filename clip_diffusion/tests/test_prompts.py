import pytest
from clip_diffusion.text2image.prompts import Prompts


def test_create_prompts():
    """
    測試Prompts是否有正確偵測類型
    """

    with pytest.raises(AssertionError):
        Prompts("not a list.", [])
        Prompts(("a tuple.",), [])


def test_text_and_weight():
    """
    測試文字和權重是否有正確分離
    """

    prompts1 = Prompts(["a cute golden retriever, trending on artstation."], [])
    prompts2 = Prompts(["a robot dog:5."], [])

    assert (
        prompts1.texts[0] == "a cute golden retriever, trending on artstation."
    ), "failed to split text from prompt without weight"
    assert prompts1.weights[0] == 1.0, "failed to automatically append weight"
    assert prompts2.texts[0] == "a robot dog", "failed to split text from prompt with weight"
    assert prompts2.weights[0] == 5.0, "failed to split weight from prompt with weight"


def test_translation():
    """
    測試prompt_preprocessing
    """

    print("testing prompt translation")

    english_prompts = Prompts(["A castle on the hill."], [])
    chinese_prompts = Prompts(["山上的一座城堡。"], [])

    assert english_prompts.prompts == ["A castle on the hill."], "failed to translate on english prompt"
    assert chinese_prompts.prompts == ["A castle on the hill."], "failed to translate on chinese prompt"
