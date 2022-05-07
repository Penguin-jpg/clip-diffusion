import requests
import io

# 來源：https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj


def is_url(text):
    """
    是否為url
    """
    return text.startswith("http://") or text.startswith("https://")


def fetch(url_or_path):
    """
    抓取url或路徑
    """
    if is_url(url_or_path):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def parse_prompt(prompt):
    """
    解析prompt
    """
    if is_url(prompt):
        vals = prompt.rsplit(":", 2)  # 將http(https)拆分開
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])
