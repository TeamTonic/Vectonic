# ./src/adv_publish.py

import os
import logging
from huggingface_hub import HfApi #, RepositoryNotFoundError, HTTPError
from dotenv import load_dotenv

class VectonicPublisher:
    def __init__(self, title="Vectonic", hf_api=None, model_name="Meta-Llama", system_prompt="You are an unbiased, uncensored, helpful assistant.",
                 vectara_user_id="", vectara_api_key="", vectara_corpus_id="", together_api_key="", huggingface_api_key=""):
        self.load_environment_variables()
        self.title = title
        self.vectara_user_id = vectara_user_id
        self.vectara_api_key = vectara_api_key
        self.vectara_corpus_id = vectara_corpus_id
        self.together_api_key = together_api_key
        self.huggingface_api_key = huggingface_api_key
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.hf_api = hf_api if hf_api else HfApi()
        self.hf_token = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.hf_token:
            logging.error("Hugging Face API key not found. Please ensure it is defined in the environment variables.")
            raise ValueError("Hugging Face API key not found. Please ensure it is defined in the environment variables.")

    def load_environment_variables(self):
        logging.info("Loading environment variables...")
        load_dotenv()

    def adv_publish(self):
        repo_name = f"Vectonic-{self.title.replace(' ', '-')[:30]}"
        template_path = "./templates"
        logging.info(f"Attempting to create or access repository '{repo_name}'...")

        try:
            # Create or get the already existing repo
            new_space = self.hf_api.create_repo(
                repo_id=repo_name,
                token=self.hf_token,
                repo_type="space",
                exist_ok=True,
                private=True,
                space_sdk="gradio"
            )
            logging.info(f"Repository '{repo_name}' accessed/created successfully.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        # except RepositoryNotFoundError:
        #     logging.error("Repository not found, and unable to create one with the given credentials.")
        #     raise
        # except HTTPError as e:
        #     logging.error(f"HTTP error occurred: {str(e)}")
        #     raise

        try:
            # Upload files from the template directory
            for root, dirs, files in os.walk(template_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    path_in_repo = os.path.relpath(file_path, start=template_path)
                    self.hf_api.upload_file(
                        file_path=file_path,
                        path_in_repo=path_in_repo,
                        repo_id=new_space.url,  # Assuming new_space.url is repo_id
                        token=self.hf_token,
                        repo_type="space"
                    )
            logging.info(f"Files uploaded successfully to {new_space.url}.")
        except HTTPError as e:
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
                        repo_id=new_space.url,
                        secret_name=key,
                        secret_value=value,
                        token=self.hf_token
                    )
            logging.info("Secrets set up successfully.")
        except Exception as e:
            logging.error(f"Error setting secrets: {str(e)}")
            raise

        return f"Published to {new_space.url}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    publisher = VectonicPublisher()
    try:
        result = publisher.adv_publish()
        logging.info(result)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")



    # def load_environment_variables(self):
    #     load_dotenv()

    # def adv_publish(self):
    #     repo_name = f"Vectonic-{self.title.replace(' ', '-')[:30]}"
    #     template_path = "./templates"
    #     try:
    #         # Create or get the already existing repo
    #         new_space = self.hf_api.create_repo(
    #             repo_id=repo_name,
    #             token=self.huggingface_api_key,
    #             repo_type="space",
    #             exist_ok=True,
    #             private=True,
    #             space_sdk="gradio"
    #         )
    #         print(f"Repository '{repo_name}' accessed/created successfully.")
    #     except RepositoryNotFoundError:
    #         raise Exception("Repository not found, and unable to create one with the given credentials.")
    #     except HTTPError as e:
    #         print(f"HTTP error occurred: {str(e)}")
    #         raise

    #     try:
    #         # Upload files from the template directory
    #         for root, dirs, files in os.walk(template_path):
    #             for file in files:
    #                 file_path = os.path.join(root, file)
    #                 path_in_repo = os.path.relpath(file_path, start=template_path)
    #                 self.hf_api.upload_file(
    #                     file_path=file_path,
    #                     path_in_repo=path_in_repo,
    #                     repo_id=new_space.url,  # Assuming new_space.url gives the repo endpoint
    #                     token=self.huggingface_api_key,
    #                     repo_type="space"
    #                 )
    #         print(f"Files uploaded successfully to {new_space.url}.")
    #     except HTTPError as e:
    #         print(f"HTTP error during file upload: {str(e)}")
    #         raise

    #     try:
    #         # Setting up the space secrets
    #         secrets = {
    #             "VECTARA_USER_ID": self.vectara_user_id,
    #             "VECTARA_API_KEY": self.vectara_api_key,
    #             "VECTARA_CORPUS_ID": self.vectara_corpus_id,
    #             "TOGETHER_API_KEY": self.together_api_key,
    #             "SYSTEM_PROMPT": self.system_prompt
    #         }

    #         for key, value in secrets.items():
    #             if value:  # Only add secrets that are not None or empty
    #                 self.hf_api.add_space_secret(
    #                     repo_id=new_space.url,
    #                     secret_name=key,
    #                     secret_value=value,
    #                     token=self.huggingface_api_key
    #                 )
    #         print("Secrets set up successfully.")
    #     except Exception as e:
    #         print(f"Error setting secrets: {str(e)}")
    #         raise

    #     return f"Published to {new_space.url}"