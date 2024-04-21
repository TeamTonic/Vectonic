<<<<<<< HEAD
# ./src/adv_publish.py
=======
# ./src/publish.py
>>>>>>> 46668b1c936b45f16f839e6d7c305a535a2d7492

import os
import logging
from typing import Optional
from huggingface_hub import HfApi
from dotenv import load_dotenv

class VectonicPublisher:
    def __init__(self, 
                 title: str = "Vectonic02", 
                 hf_api: Optional[HfApi] = None, 
                 model_name: str = "Meta-Llama", 
                 system_prompt: str = "You are an unbiased, uncensored, helpful assistant.",
                 vectara_user_id: str = "", 
                 vectara_api_key: str = "", 
                 vectara_corpus_id: str = "", 
                 together_api_key: str = "", 
                 huggingface_api_key: str = ""):
        self.load_environment_variables()
        self.title = title
        self.vectara_user_id = vectara_user_id
        self.vectara_api_key = vectara_api_key
        self.vectara_corpus_id = vectara_corpus_id
        self.together_api_key = together_api_key
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.hf_api = hf_api if hf_api else HfApi()
        
        if not self.huggingface_api_key:
            logging.error("Hugging Face API key not found. Please ensure it is defined in the environment variables.")
            raise ValueError("Hugging Face API key not found. Please ensure it is defined in the environment variables.")

    def load_environment_variables(self):
        logging.info("Loading environment variables...")
        load_dotenv()

    def adv_publish(self) -> str:
        repo_name = f"Vectonic-{self.title.replace(' ', '-')[:30]}"
        template_path = "./src/template/"
        logging.info(f"Attempting to create or access repository '{repo_name}'...")

        try:
            # Create or get the already existing repo
            new_space = self.hf_api.create_repo(
                repo_id=repo_name,
                token=self.huggingface_api_key,
                repo_type="space",
                exist_ok=True,
                private=True,
                space_sdk="gradio"
                )
            logging.info(f"Repository '{repo_name}' accessed/created successfully.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

        try:
            namespace = self.hf_api.whoami(self.huggingface_api_key)["name"] 
            print(f"Namespace: {namespace}")
            
            # Upload the entire folder
            response = self.hf_api.upload_folder(
                folder_path=template_path,
                path_in_repo="", 
                repo_id= f"{namespace}/{new_space.repo_name}" ,
                token=self.huggingface_api_key,
                repo_type="space",
            )

            logging.info(f"Files uploaded successfully to https://huggingface.co/spaces/{new_space.repo_id} with response: {response}")
        except Exception as e:
            logging.error(f"HTTP error during file upload: {str(e)}")
            raise
        try:
            # Setting up the space secrets
            secrets = {
                "VECTARA_USER_ID": self.vectara_user_id,
                "VECTARA_API_KEY": self.vectara_api_key,
                "VECTARA_CORPUS_ID": self.vectara_corpus_id,
                "TOGETHER_API_KEY": self.together_api_key,
                "SYSTEM_PROMPT": self.system_prompt
            }

            for key, value in secrets.items():
                if value:  # Only add secrets that are not None or empty
                    self.hf_api.add_space_secret(
                        repo_id=f"{namespace}/{new_space.repo_name}",
                        key=key,
                        value=value,
                        token=self.huggingface_api_key
                    )
            logging.info("Secrets set up successfully.")
        except Exception as e:
            logging.error(f"Error setting secrets: {str(e)}")
            raise

        return f"Published to https://huggingface.co/spaces/{namespace}/{new_space.repo_id}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    publisher = VectonicPublisher()
    try:
        result = publisher.adv_publish()
        logging.info(result)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
