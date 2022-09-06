import os
import pandas as pd
import re
import numpy as np
from transformers import pipeline
from opencc import OpenCC
from clip_diffusion.config import Config
from clip_diffusion.utils.dir_utils import CSV_PATH, INDEX_PATH
from clip_diffusion.models import load_sentence_transformer
from clip_diffusion.utils.embedding_index import load_faiss_index, get_topk_results

_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 用來中翻英
_converter = OpenCC("tw2sp.json")  # 繁體轉簡體
_sentence_transformer = load_sentence_transformer("sentence-transformers/sentence-t5-base", Config.device)  # 用來找出文字的embedding
# index對應的dataframe
modifier_df = pd.read_csv(os.path.abspath(os.path.join(CSV_PATH, "modifiers.csv")))
# index
modifier_index = load_faiss_index(os.path.abspath(os.path.join(INDEX_PATH, "modifier_embeddings.index")))


class Prompt:
    """負責prompts功能的class"""

    def __init__(self, prompt, use_auto_modifiers, num_modifiers):
        assert isinstance(prompt, str), "prompt has to be 'str' type"
        self.prompt = self._preprocess(prompt, use_auto_modifiers, num_modifiers)  # prompts前處理
        self.text, self.weight = self._get_text_and_weight()  # 文字與權重

    def _contains_zh(self, prompt):
        """檢查是否包含中文"""
        if re.search(r"[\u4e00-\u9FFF]", prompt):
            return True
        return False

    def _translate_zh_to_en(self, prompt):
        """將中文翻譯成英文"""
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

    def _append_modifiers(self, prompt, num_modifiers=1):
        """為prompt加上修飾詞"""
        global _sentence_transformer, modifier_index, modifier_df

        # 補上一維
        text_embedding = np.expand_dims(_sentence_transformer.encode(prompt), axis=0)
        # 算出相似度及index
        similarties, indices = get_topk_results(modifier_index, text_embedding, num_modifiers)
        # 將修飾詞到prompt
        for index in indices[0]:
            prompt += f", {modifier_df.iloc[index]['Keyword']}"
        # 補上trending on artstation
        prompt += ", trending on artstation."
        return similarties[0][:num_modifiers], prompt

    def _preprocess(self, prompt, use_auto_modifiers=True, num_modifiers=1):
        """對prompts做需要的前處理: 1. 中翻英 2. 加上修飾詞"""
        # 先中翻英
        prompt = self._translate_zh_to_en(prompt)
        if use_auto_modifiers:
            # 加上修飾詞
            _, prompt = self._append_modifiers(prompt, num_modifiers)
        return prompt

    # 參考並修改自：https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj
    def _get_text_and_weight(self):
        """分離prompts的文字與權重"""
        parsed = self.prompt.split(":", 1)  # prompt的格式為"文字:權重"，所以透過":"進行切割
        parsed = parsed + [1.0] if len(parsed) == 1 else parsed  # 如果原本未標示權重，就補上權重1
        # 儲存文字與權重
        return parsed[0], float(parsed[1])
