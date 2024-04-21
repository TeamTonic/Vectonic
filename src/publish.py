# ./src/publish
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

class VectonicPublisher:
    def __init__(
        self, 
        title="Vectonic",
        hf_api = HfApi() ,
        model_name = "Meta-Llama",
        system_prompt = "You are an unbiased, uncensored, helpful assistant.",
        vectara_user_id = "",
        vectara_api_key = "",
        vectara_corpus_id = "",
        together_api_key = "",
        huggingface_api_key = ""
        ):
        self.hf_token, self.systemprompt, self.userprompt, self.vectara_api_key, self.vectara_userid, self.corpusid, self.together_api_key, self.huggingface_api_key = self.load_environment_variables()
        self.title = title
        self.vectara_userid =  vectara_user_id
        self.vectara_api_key =  vectara_api_key
        self.vectara_corpusid =  vectara_corpus_id
        self.together_api_key =  together_api_key
        self.huggingface_api_key = huggingface_api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.hf_token = huggingface_api_key
        self.hf_api = hf_api(endpoint="https://huggingface.co", token=self.hf_token , repo_type = "spaces")

        if not self.hf_token:
            raise ValueError("Hugging Face API key not found. Please ensure it is defined in .env")
    
    @staticmethod
    def load_environment_variables():
        env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
        load_dotenv(dotenv_path=env_path)
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        systemprompt = os.getenv("SYSTEMPROMPT")
        userprompt = os.getenv("USERPROMPT")
        vectara_userid = os.getenv("VECTARA_USER_ID"), 
        vectara_api_key = os.getenv("VECTARA_API_KEY"), 
        corpusid = os.getenv("VECTARA_CORPUS_ID"), 
        huggingface_api_key = os.getenv("TOGETHER_API_KEY"), 
        together_api_key = os.getenv("HUGGINGFACE_API_KEY"), 
        return hf_token , systemprompt , userprompt , vectara_api_key, vectara_userid, corpusid, together_api_key, huggingface_api_key

    def publish(self):
        deployment_path = "./src/template/"
        title = (self.title[:30])  # Ensuring title does not exceed max bytes
        new_space = self.hf_api.create_repo(
            repo_id=f"Vectonic-{title}",
            repo_type="space",
            exist_ok=True,
            private=True,
            space_sdk="gradio",
            token=self.hf_token,
        )
        for root, dirs, files in os.walk(deployment_path):
            for file in files:
                file_path = os.path.join(root, file)
                path_in_repo = os.path.relpath(file_path, start=deployment_path)
                self.hf_api.upload_file(
                    repo_id=new_space.repo_id,
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    token=self.hf_token,
                    repo_type="space",
                )
        
        self.hf_api.add_space_secret(new_space.repo_id, "HF_TOKEN", self.huggingface_api_key, token=self.huggingface_api_key)
        self.hf_api.add_space_secret(new_space.repo_id, "VECTARA_API_KEY", self.vectara_api_key, token=self.vectara_api_key)
        self.hf_api.add_space_secret(new_space.repo_id, "SYSTEM_PROMPT", self.systemprompt, token=self.hf_token)
        self.hf_api.add_space_secret(new_space.repo_id, "VECTARA_USER_ID", self.vectara_userid, token=self.vectara_userid)
        self.hf_api.add_space_secret(new_space.repo_id, "TOGETHER_API_KEY", self.together_api_key, token=self.together_api_key)
        self.hf_api.add_space_secret(new_space.repo_id, "VECTARA_CORPUS_ID", self.userprompt, token=self.hf_token)

        return f"Published to https://huggingface.co/spaces/{new_space.repo_id}"

if __name__ == "__main__":
    publisher = VectonicPublisher()
    try:
        result = publisher.adv_publish()
        print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # # deploy_routing = DeployRouting(
    # #     model_name="Meta-Llama"
    # # )
    # data = VectonicPublisher(
    #     "Vectara Sample Space",
    #     # deploy_routing=deploy_routing
    # )
    # data.publish(
        
    # )
