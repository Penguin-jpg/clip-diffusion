import re
import random
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from opencc import OpenCC
from clip_diffusion.text2image.models import load_sentence_transformer
from clip_diffusion.utils.functional import get_device

_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 用來中翻英
_converter = OpenCC("tw2sp.json")  # 繁體轉簡體
_prompt_types = {
    "生物": "creature-prompts/",
    "景觀": "environment-prompts/",
    "物件": "object-prompt/",
}  # 可用的隨機prompt類型
sentence_transformer = load_sentence_transformer("sentence-transformers/sentence-t5-base", get_device())


class Prompt:
    """
    負責prompts功能的class
    """

    def __init__(self, prompt):
        assert isinstance(prompt, str), "prompt has to be 'str' type"
        self.prompt = self._preprocess(prompt)  # prompts前處理
        self.text, self.weight = self._get_text_and_weight()  # 文字與權重

    def _contains_zh(self, prompt):
        """
        檢查是否包含中文
        """

        if re.search(r"[\u4e00-\u9FFF]", prompt):
            return True
        return False

    def _translate_zh_to_en(self, prompt):
        """
        將中文翻譯成英文
        """

        global _converter, _translator

        # 如果包含中文
        if self._contains_zh(prompt):
            # 先轉簡體，以符合翻譯模型輸入
            prompt = _converter.convert(prompt)
            # 翻譯成英文
            result = _translator(prompt)[0]
            # 更新prompts
            prompt = result["translation_text"]

        return prompt

    def _preprocess(self, prompt):
        """
        對prompts做需要的前處理: 1. 中翻英 2. prompt engineering
        """

        # 先中翻英
        prompt = self._translate_zh_to_en(prompt)
        # TODO: prompt enginerring
        return prompt

    # 參考並修改自：https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj
    def _get_text_and_weight(self):
        """
        分離prompts的文字與權重
        """

        parsed = self.prompt.split(":", 1)  # prompt的格式為"文字:權重"，所以透過":"進行切割
        parsed = parsed + [1.0] if len(parsed) == 1 else parsed  # 如果原本未標示權重，就補上權重1
        # 儲存文字與權重
        return parsed[0], float(parsed[1])

    @staticmethod
    def random_prompt(prompt_type):
        """
        生成隨機的prompt(生物, 景觀, 物件)
        """

        url = f"https://artprompts.org/{_prompt_types[prompt_type]}"  # 目標url
        request = requests.get(url)
        soup = BeautifulSoup(request.content, "html.parser", from_encoding="iso-8859-1")  # 抓取網頁
        prompt = soup.find_all("div", {"class": "et_pb_text_inner"})
        return prompt[1].text.strip().split("\n")[-1].lstrip("\t") + "."
