# app.py Made with https://github.co/TeamTonic/Vectonic

import os
import gradio as gr
import requests
from vectara_cli.core import VectaraClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration variables
VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
TOGETHER_API_TOKEN = os.getenv("TOGETHER_API_TOKEN")
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
MODEL_NAME = os.getenv("MODEL_NAME", "databricks/dbrx-instruct")
TITLE = os.getenv("TITLE")
DESCRIPTION = os.getenv("DESCRIPTION")
class VectaraClientExtended(VectaraClient):
    """ Enhanced Vectara Client with methods tailored for chatbot application. """
    def __init__(self, customer_id, api_key):
        super().__init__(customer_id, api_key)
    
    def retrieve_context(self, query, corpus_id):
        response = self.advanced_query(query, 1, corpus_id, {}, {})
        if response.get('matches'):
            return response['matches'][0]['document']['text']
        return None

class TogetherAIInterface:
    """ Integration for querying TogetherAI LLM. """
    def __init__(self, token):
        self.headers = {"Authorization": f"Bearer {token}"}

    def generate_response(self, prompt, model="curie", max_tokens=150, temperature=0.7):
        url = 'https://api.together.xyz/v1/completions'
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json().get('choices')[0]['text']

# Application logic
def chatbot_response(user_query):
    vectara_client = VectaraClientExtended(VECTARA_CUSTOMER_ID, VECTARA_API_KEY)
    together_ai = TogetherAIInterface(TOGETHER_API_TOKEN)
    
    # Retrieve context from Vectara based on the user's query
    context = vectara_client.retrieve_context(user_query, CORPUS_ID)
    if not context:
        context = "No relevant context was found for your query."
    
    # Prepare the prompt
    prompt = f"System Prompt: {DEFAULT_SYSTEM_PROMPT}\nContext: {context}\nQuery: {user_query}"

    # Get response from TogetherAI
    model_response = together_ai.generate_response(prompt)
    
    return model_response

# Gradio interface setup
def main():
    iface = gr.Chatbot(
        fn=chatbot_response,
        inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
        outputs="text",
        title="Vectonic ChatBot",
        description="Optimized Using Tonicai. Powered by Vectara and TogetherAI. Ask anything!",
    )
    
    iface.launch()

if __name__ == "__main__":
    main()