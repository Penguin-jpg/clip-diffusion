import pytest
from clip_diffusion.text2image.prompt import Prompt


def test_create_prompt():
    """
    測試Prompt是否有正確偵測類型
    """

    with pytest.raises(AssertionError):
        Prompt(["not a list."], [])
        Prompt(123, [])
        Prompt(("a tuple.",), [])


def test_text_and_weight():
    """
    測試文字和權重是否有正確分離
    """

    prompt1 = Prompt("a cute golden retriever, trending on artstation.", [])
    prompt2 = Prompt("a robot dog:5.", [])

    assert prompt1.text == "a cute golden retriever, trending on artstation.", "failed to split text from prompt without weight"
    assert prompt1.weight == 1.0, "failed to automatically append weight"
    assert prompt2.text == "a robot dog", "failed to split text from prompt with weight"
    assert prompt2.weight == 5.0, "failed to split weight from prompt with weight"


def test_translation():
    """
    測試prompt_preprocessing
    """

    print("testing prompt translation")

    english_prompt = Prompt("A castle on the hill.", [])
    chinese_prompt = Prompt("山上的一座城堡。", [])

    assert english_prompt.prompt == "A castle on the hill.", "failed to translate on english prompt"
    assert chinese_prompt.prompt == "A castle on the hill.", "failed to translate on chinese prompt"
