# ./src/publish

class VectonicPublisher:
    def __init__(self, title, api_key, model_name):
        self.title = title
        self.api_key = api_key
        self.model_name = model_name
        # self.api = HfApi()
        # self.deploy_router = DeployRouting(model_name)
        # self.api_key_router = APIKeyRouter(model_name)
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
            repo_id=f"tonic-ai-torchon-{title}",
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
        # Add secrets
        api_key = self.api_key_router.get_key_for_model()
        self.api.add_space_secret(new_space.repo_id, "HF_TOKEN", self.hf_token, token=self.hf_token)
        self.api.add_space_secret(new_space.repo_id, "API_KEY", api_key, token=self.hf_token)
        self.api.add_space_secret(new_space.repo_id, "SYSTEM_PROMPT", self.systemprompt, token=self.hf_token)
        self.api.add_space_secret(new_space.repo_id, "USER_PROMPT", self.userprompt, token=self.hf_token)

        return f"Published to https://huggingface.co/spaces/{new_space.repo_id}"
