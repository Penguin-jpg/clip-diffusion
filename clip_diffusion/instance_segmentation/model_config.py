from mmcv import Config
from mmdet.utils import replace_cfg_vals, setup_multi_processes


def load_config(config_path):
    """
    從檔案讀取模型config
    """

    config = Config.fromfile(config_path)
    config = replace_cfg_vals(config)
    setup_multi_processes(config)
    return config


def show_config(config):
    """
    顯示config資訊
    """

    print(f"config:\n{config.pretty_text}")
