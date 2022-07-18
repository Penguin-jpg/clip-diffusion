import os
import torch
import clip
import requests
import io
from PIL import Image
from clip_retrieval.clip_client import ClipClient, Modality
from clip_diffusion.utils.dir_utils import make_dir
from clip_diffusion.utils.functional import to_clip_image, tokenize, get_text_embedding, get_image_embedding


class QueryClient:
    """
    負責進行query的client
    """

    def __init__(
        self,
        backend_url="https://knn5.laion.ai/knn-service",
        indice_name="laion5B",
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=500,
    ):
        self.client = self._create_clip_client(backend_url, indice_name, aesthetic_score, aesthetic_weight, modality, num_images)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = clip.load("ViT-L/14", device=self._device)
        self._model.eval().requires_grad_(False)

    def _create_clip_client(
        self,
        backend_url="https://knn5.laion.ai/knn-service",
        indice_name="laion5B",
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=500,
    ):
        """
        建立Clip retrieval client
        """

        return ClipClient(
            url=backend_url,
            indice_name=indice_name,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
            modality=modality,
            num_images=num_images,
        )

    def _results_to_json(self, results, output_path):
        """
        將query的結果存成json
        """

        if output_path:
            dir_path = os.path.dirname(output_path)  # 取出output_path中的資料夾名稱
            # 如果output_path包含資料夾路徑
            if dir_path != "":
                make_dir(dir_path)

            with open(output_path, "w") as file:
                import json

                json.dump(results, file)
        else:
            print("path cannot be empty")

    # 參考並修改自：https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3#scrollTo=YHOj78Yvx8jP
    def _fetch_image(self, url):
        if not url.startswith("http://") and not url.startswith("https://"):
            print("not a valid url")
            return
        else:
            request = requests.get(url)
            request.raise_for_status()
            output = io.BytesIO()
            output.write(request.content)
            output.seek(0)  # 回到檔案開頭
            return Image.open(output)

    # 參考並改寫自：https://colab.research.google.com/github/rom1504/clip-retrieval/blob/master/notebook/clip-client-query-api.ipynb?hl=zh-tw#scrollTo=1YSHcuCPgHhY
    def _merge_embeddings(self, embedding1, embedding2):
        """
        合併兩個embedding
        """

        merged = embedding1 + embedding2  # 兩個embedding相加
        l2_norm = torch.norm(merged, p=2, dim=-1, keepdim=True)  # 算出相加後的L2 norm
        return (merged / l2_norm).tolist()  # L2 normalize

    def get_query_results(
        self,
        text=None,
        image_url=None,
        num_results=500,
        to_json=False,
        output_path=None,
    ):
        """
        透過文字或圖片進行query
        """

        assert num_results >= 0, "number of results cannot be zero"

        # 如果text和image_url都有值就透過embedding組合
        if text and image_url:
            text_embedding = get_text_embedding(self._model, tokenize(text, self._device), divided_by_norm=True)[0]
            image_embedding = get_image_embedding(
                self._model,
                to_clip_image(self._preprocess, self._fetch_image(image_url), self._device),
                use_normalize=False,
                divided_by_norm=True,
            )[0]
            results = self.client.query(embedding_input=self._merge_embeddings(text_embedding, image_embedding))
        else:
            results = self.client.query(text=text, image=image_url)

        if num_results > len(results):
            print("excceeds max number of results! automatically shorten to match max length")
        else:
            results = results[:num_results]

        if to_json:
            self._results_to_json(results, output_path)

        return results

    def combine_results(self, results_1, results_2, num_results=1000, to_json=False, output_path=None):
        """
        將兩個results結合
        """

        if num_results < 0:
            print("number of results cannot be zero")
            return

        new_results = results_1 + results_2

        if num_results > len(new_results):
            print("excceeds max number of results! automatically shorten to match max length")
        else:
            new_results = new_results[:num_results]

        if to_json:
            self._results_to_json(new_results, output_path)

        return new_results
