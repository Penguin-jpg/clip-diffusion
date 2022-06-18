import re
from transformers import pipeline
from opencc import OpenCC

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 中翻英

converter = OpenCC("tw2sp.json")  # 繁體轉簡體


def contains_zh(prompt):
    """
    檢查是否包含中文
    """
    if re.search(r"[\u4e00-\u9FFF]", prompt):
        return True
    return False


def translate_zh_to_en(text_prompts):
    """
    將中文翻譯成英文
    """

    # 先轉簡體，以符合模型輸入
    for index, prompt in enumerate(text_prompts):
        # 如果包含中文
        if contains_zh(prompt):
            prompt = converter.convert(prompt)
            # 翻譯成英文
            result = translator(prompt)[0]
            # 更新prompt
            text_prompts[index] = result["translation_text"]

    return text_prompts


def preprocess_pipeline(text_prompts):
    """
    預處理pipeline
    """

    text_prompts = translate_zh_to_en(text_prompts)
    # ...
    return text_prompts
