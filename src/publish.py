# ./src/publish
import os
from huggingface_hub import HfApi


class VectonicPublisher:
    def __init__(self, title, vectara_userid, vectara_api_key, corpusid, huggingface_api_key, together_api_key, model_name):
        self.title = title
        self.vectara_userid =  os.getenv("VECTARA_USER_ID")
        self.vectara_api_key =  os.getenv("VECTARA_API_KEY")
        self.vectara_corpusid =  os.getenv("VECTARA_CORPUS_ID")
        self.together_api_key =  os.getenv("TOGETHER_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = model_name
        self.hf_token, self.systemprompt, self.userprompt = self.load_environment_variables()

        if not self.hf_token:
            raise ValueError("Hugging Face API key not found. Please ensure it is defined in .env")
    
    @staticmethod
    def load_environment_variables():
        env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
        load_dotenv(dotenv_path=env_path)
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        systemprompt = os.getenv("SYSTEMPROMPT")
        userprompt = os.getenv("USERPROMPT")
        return hf_token , systemprompt , userprompt

    def publish(self):
        deployment_path = self.deploy_router.get_deployment_path()
        title = (self.title[:30])  # Ensuring title does not exceed max bytes
        new_space = self.api.create_repo(
            repo_id=f"Vectonic-{title}",
            repo_type="space",
            exist_ok=True,
            private=False,
            space_sdk="gradio",
            token=self.hf_token,
        )
        for root, dirs, files in os.walk(deployment_path):
            for file in files:
                file_path = os.path.join(root, file)
                path_in_repo = os.path.relpath(file_path, start=deployment_path)
                self.api.upload_file(
                    repo_id=new_space.repo_id,
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    token=self.hf_token,
                    repo_type="space",
                )
        
        self.api.add_space_secret(new_space.repo_id, "HF_TOKEN", self.huggingface_api_key, token=self.huggingface_api_key)
        self.api.add_space_secret(new_space.repo_id, "VECTARA_API_KEY", self.vectara_api_key, token=self.vectara_api_key)
        self.api.add_space_secret(new_space.repo_id, "SYSTEM_PROMPT", self.systemprompt, token=self.hf_token)
        self.api.add_space_secret(new_space.repo_id, "VECTARA_USER_ID", self.vectara_userid, token=self.vectara_userid)
        self.api.add_space_secret(new_space.repo_id, "TOGETHER_API_KEY", self.together_api_key, token=self.together_api_key)
        self.api.add_space_secret(new_space.repo_id, "VECTARA_CORPUS_ID", self.userprompt, token=self.hf_token)

        return f"Published to https://huggingface.co/spaces/{new_space.repo_id}"
