import sys
import subprocess
from clip_diffusion.utils.dir_utils import make_dir, OUTPUT_PATH, MODEL_PATH


class ColabHelper:
    """幫忙包裝、簡化anvil和colab準備工作"""

    def __init__(self):
        repos = (
            "https://github.com/openai/CLIP.git",
            "https://github.com/crowsonkb/guided-diffusion.git",
            "https://github.com/Penguin-jpg/latent-diffusion.git",
            "https://github.com/Penguin-jpg/taming-transformers.git",
            "https://github.com/xinntao/Real-ESRGAN.git",
        )
        repo_folders = (
            "CLIP",
            "guided-diffusion",
            "latent-diffusion",
            "taming-transformers",
            "Real-ESRGAN",
            "clip-diffusion",
        )

        self._clone_dependencies(repos)
        self._install_dependencies_from_repos(repo_folders)
        self._append_paths(repo_folders)
        make_dir(OUTPUT_PATH)
        make_dir(MODEL_PATH)

    def _clone_dependencies(self, repos):
        """將dependency從github clone下來"""
        for repo in repos:
            subprocess.run(["git", "clone", repo])

        print("successfully cloned all repos")

    def _install_dependencies_from_repos(self, repo_folders):
        """透過pip從repo folder安裝dependency"""
        for repo_folder in repo_folders:
            result = subprocess.run(
                ["pip", "install", "-e", repo_folder], stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
            print(result)

    def _install_dependencies_from_pypi(self, module_names):
        """透過pip從pypi安裝dependency"""
        for module_name in module_names:
            result = subprocess.run(["pip", "install", module_name], stdout=subprocess.PIPE).stdout.decode(
                "utf-8"
            )
            print(result)

    def _install_dependencies_from_requirements(self, requirements_file):
        """透過pip從requirements.txt安裝dependency"""
        result = subprocess.run(
            ["pip", "install", "-r", requirements_file], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        print(result)

    def _append_paths(self, module_folders):
        """將需要的module加入到系統路徑"""
        for module_folder in module_folders:
            sys.path.append(module_folder)

    def connect_to_anvil(self, uplink_key):
        """連線到anvil"""
        import anvil.server  # 避免dependency問題

        anvil.server.connect(uplink_key)

    def start_server(self):
        """server開始等到呼叫"""
        import anvil.server

        print("start server!")
        anvil.server.wait_forever()
