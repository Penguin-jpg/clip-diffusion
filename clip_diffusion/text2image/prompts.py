import re
import random
from transformers import pipeline
from opencc import OpenCC

_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 用來中翻英
_converter = OpenCC("tw2sp.json")  # 繁體轉簡體
_STYLES = {
    "奇幻 / 科幻": [
        "James Gurney",
        "Greg Rutkowski",
        "Pascal Blanche",
        "Jakub Rozalski",
        "Federico Pelat",
    ],
    "注重光影": ["James Paick", "Fitz Henry Lane"],
    "立體效果": ["unreal engine"],
    "風景畫": ["Julian Falat", "Isaac Levitan", "John Constable", "Ivan Shishkin"],
    "印象派": ["Van Gogh", "Monet"],
    "早晨": ["morning"],
    "傍晚": ["evening"],
    "夕陽": ["sunset"],
}  # 風格標籤
prompts_type = {'creature' : 'creature-prompts/',
               'environment' : 'environment-prompts/',
               'object' : 'object-prompt/'}

class Prompts:
    """
    負責prompts功能的class
    """

    def __init__(self, prompts, styles=[]):
        assert isinstance(prompts, list), "prompts has to be 'list' type"
        self.prompts = self._preprocess(prompts, styles)  # prompts前處理
        self.texts, self.weights = self._get_texts_and_weights()  # 文字與權重

    def _contains_zh(self, prompt):
        """
        檢查是否包含中文
        """

        if re.search(r"[\u4e00-\u9FFF]", prompt):
            return True
        return False

    def _translate_zh_to_en(self, prompts):
        """
        將中文翻譯成英文
        """

        global _converter, _translator

        # 先轉簡體，以符合模型輸入
        for index, prompt in enumerate(prompts):
            # 如果包含中文
            if self._contains_zh(prompt):
                prompt = _converter.convert(prompt)
                # 翻譯成英文
                result = _translator(prompt)[0]
                # 更新prompts
                prompts[index] = result["translation_text"]

        return prompts

    def _append_styles_to_prompts(self, prompts, styles=[]):
        """
        根據對應的風格加上風格標籤
        """

        # 如果使用者有選擇風格才做
        if styles:
            for index, prompt in enumerate(prompts):
                # 從prompts中暫時移除句點
                if prompt[-1] == ".":
                    append_period = True
                    prompt = prompt[:-1]
                else:
                    append_period = False

                # 一律加入artstation
                prompt += ", artstation"

                # 從選定的風格中挑出一個選項
                for style in styles:
                    prompt += f", {_STYLES[style][random.randint(0, len(_STYLES[style])-1)]}"

                # 需要時補回句點
                if append_period:
                    prompt += "."

                # 更新prompts
                prompts[index] = prompt

        return prompts

    def _preprocess(self, prompts, styles=[]):
        """
        對prompts做需要的前處理: 1. 中翻英 2. prompt engineering
        """

        # 先中翻英
        prompts = self._translate_zh_to_en(prompts)
        # 根據選擇的風格做prompt engineering
        prompts = self._append_styles_to_prompts(prompts, styles)
        return prompts

    # 參考並修改自：https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj
    def _get_texts_and_weights(self):
        """
        分離prompts的文字與權重
        """

        texts = []
        weights = []

        for prompt in self.prompts:
            parsed = prompt.split(":", 1)  # prompt的格式為"文字:權重"，所以透過":"進行切割
            parsed = parsed + [1.0] if len(parsed) == 1 else parsed  # 如果原本未標示權重，就補上權重1
            # 儲存文字與權重
            texts.append(parsed[0])
            weights.append(float(parsed[1]))

        return texts, weights


    def random_prompt_generate(prompt_style):
        """
        生成隨機prompt
        有creature, environment, object三種
        """

        import requests
        from bs4 import BeautifulSoup

        url = "https://artprompts.org/"+prompts_type[prompt_style] #driver.current_url 
        request = requests.get(url)
        soup = BeautifulSoup(request.content, "html.parser", from_encoding="iso-8859-1") #用bs4爬文章

        prompt = soup.find_all("div", {"class":"et_pb_text_inner"})

        return prompt[1].text.strip().split("\n")[-1]
