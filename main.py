# main.py 

import json
from pathlib import Path

from llama_index.llms.together import TogetherLLM
import os
from src.dataloader import DataProcessor , DocumentLoader
from vectara_cli.core import VectaraClient, QueryRequest, QueryResponse
from vectara_cli.rebel_span.noncommercial.nerdspan import Span
from vectara_cli.rebel_span.commercial.enterprise import EnterpriseSpan
import nest_asyncio
import requests
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import unstructured
import os
from src.dataloader import DataProcessor, DocumentLoader
from src.chunking import MarkdownProcessor
from src.adv_publish import VectonicPublisher
from unstructured.partition.md import partition_md as partition_md
from typing import List, Dict, Optional
from tonic_validate import Benchmark, ValidateScorer, LLMResponse
from tonic_validate.metrics.answer_similarity_metric import  AnswerSimilarityMetric as AnswerSimilarityScore
from tonic_validate.metrics.retrieval_precision_metric import RetrievalPrecisionMetric as RetrievalPrecision
from tonic_validate.metrics.augmentation_accuracy_metric import AugmentationAccuracyMetric as AugmentationAccuracy
from tonic_validate.metrics.answer_consistency_metric import AnswerConsistencyMetric as AnswerConsistency
from tonic_validate.metrics.latency_metric import LatencyMetric as Latency
from tonic_validate.metrics.contains_text_metric import ContainsTextMetric as ContainsText
from dotenv import load_dotenv
from together import Together
from together.resources.completions import Completions
from together.types.abstract import TogetherClient
from together.types.completions import CompletionResponse
load_dotenv()
nest_asyncio.apply()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class DataLoading:
    def __init__(self, folder_path: str, text_folder_path: str):
        self.folder_path = folder_path
        self.text_folder_path = text_folder_path
        os.makedirs(self.text_folder_path, exist_ok=True)

    def process_files(self) -> List[str]:
        """
        Method to process each file in the specified directory, extract Markdown content, and save it to files.
        """
        markdown_paths = []
        print(f"Processing files in folder: {self.folder_path}")
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    reader = DataProcessor.choose_reader(file_path)
                    if not reader:
                        continue

                    documents = reader.load_data(file_path)
                    for doc in documents:
                        text = doc['text']  # Assuming each document has a 'text' key
                        markdown_file_path = os.path.join(self.text_folder_path, f"{Path(file).stem}.md")
                        with open(markdown_file_path, 'w') as md_file:
                            md_file.write(text)
                        markdown_paths.append(markdown_file_path)
                        print(f"Markdown saved to {markdown_file_path}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        return markdown_paths

class Chonker:
    def __init__(self, markdown_files: List[str]):
        self.markdown_files = markdown_files

    def process_markdown_files(self) -> Dict[str, List]:
        """
        Chunks markdown files to extract chunks (elements) using partition_md function.
        """
        md_structure = {}
        for md_file in self.markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                elements = partition_md(text=markdown_content)  # adapted to use text directly
                md_structure[md_file] = elements
                print(f"Processed {md_file}: {len(elements)} elements found")
            except Exception as e:
                print(f"Error processing {md_file}: {str(e)}")
        return md_structure

class VectaraDataIndexer:
    def __init__(self, customer_id: int, api_key: str):
        self.vectara_client = VectaraClient(customer_id, api_key)

    def create_corpus(self, corpus_name: str) -> int:
        corpus_data = {
            "name": corpus_name,
            "description": "A corpus for " + corpus_name,
            "metadata": json.dumps({
                "category": "Vectonic",
            })
        }
        response = self.vectara_client.create_corpus(corpus_data)
        if response.get('status', {}).get('code') == 'OK':
            return response['corpusId']
        else:
            print("Failed to create corpus", response)
            return None

    def index_folder(self, corpus_id: int, folder_path: str):
        results = self.vectara_client.index_documents_from_folder(corpus_id, folder_path)
        return results

    def index_markdown_chunks(self, corpus_id: int, markdown_chunks: Dict[str, List]):
        for filepath, sections in markdown_chunks.items():
            for index, section in enumerate(sections):
                title = f"{Path(filepath).stem}-section-{index}"
                document_id = f"{Path(filepath).stem}-{index}"
                response, status = self.vectara_client.index_document(
                    corpus_id, document_id, title, {"section_number": index}, section
                )
                print(f"Indexed section '{title}' status: {status}")

    def index_markdown_chunks_with_entities(self, corpus_id: int, markdown_chunks: Dict[str, List]):
        self.span_processor = Span(vectara_client=self.vectara_client, text="", model_name="fewnerdsuperfine", model_type="span_marker")
        for filepath, sections in markdown_chunks.items():
            for index, section in enumerate(sections):
                title = f"{Path(filepath).stem}-section-{index}"
                document_id = f"{Path(filepath).stem}-{index}"

                # Set text for NER processing
                self.span_processor.text = section
                output_str, entities = self.span_processor.analyze_text()

                enriched_text = output_str + "\n" + section
                metadata = json.dumps({ent['label']: ent['span'] for ent in entities})

                response, status = self.vectara_client.index_document(
                    corpus_id, document_id, title, {}, enriched_text, metadata_json=metadata
                )
                print(f"Indexed enriched section '{title}' status: {status}")

class Retriever:
    def __init__(self, client: VectaraClient):
        self.client = client

    def retrieve_information(self, query: str, corpus_id: int, num_results: int = 10, 
                             context_config: Optional[dict] = None, summary_config: Optional[dict] = None) -> List[QueryResponse]:
        if not context_config:
            context_config = {}
        if not summary_config:
            summary_config = {}
        
        response = self.client.advanced_query(query, num_results, corpus_id, context_config, summary_config)
        if 'error' in response:
            print(response['error'])
            return []
        context = self.client._parse_query_response(response)
        return context

    def prompt_formatting(self, systemprompt : str, context: str, query: str) -> str:
        formatted_prompt = f"System Message:{systemprompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"
        return json.dumps(formatted_prompt)
    
    def prompt_generator(
        self,
        model="meta-llama/Meta-Llama-3-70B",
        token_limit=500,
        query="Please generate a system prompt"
        ) -> str:
        
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        )
        
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        reponse = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        )
        return reponse.choices[0].message.content
        

    def query_together_llm(
        self, 
        context: str, 
        query: str, 
        model: str, 
        tokens_limit: int = 150, 
        temperature: 
        float = 0.7) -> str:
        prompt = self.prompt_formatting(
            
            context=context, 
            query=query
            )
        llm = TogetherLLM(model=model, max_tokens=tokens_limit, temperature=temperature)
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        )
        print(response.choices[0].message.content)
        
        # Prepare and send the request
        response = llm.complete(prompt)
        
        # Extract and return the response
        if response['choices']:
            return response['choices'][0]['message']['content']
        else:
            return "No response generated."
        
    def use_together_api(self, completion_context: str, model_info: dict):
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}"  # You need to replace this with your actual bearer token
        }
        
        data = {
            "prompt": completion_context,
            "model": model_info['model_string'],
            "max_tokens": model_info['max_tokens'],
            "temperature": model_info['temperature'],
            "top_p": model_info['top_p'] if 'top_p' in model_info else 1,
            "top_k": model_info['top_k'] if 'top_k' in model_info else 40,
            "repetition_penalty": model_info['repetition_penalty'] if 'repetition_penalty' in model_info else 1,
        }
        
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        reponse = client.chat.completions.create(
            
        model=model_info['model_string'],
        max_tokens=model_info['max_tokens'],
        temperature=model_info['temperature'],
        messages=[{"role": "user", "content": completion_context}],
        )
        return reponse

    def process_user_questions(
        client: VectaraClient, 
        questions: List[str], 
        corpus_id: int, 
        model_infos: Dict[str, dict]
    ):
        #  )  -> List[Dict[str]]: 
        retriever = Retriever(client)
        
        temp_name_list = []
        for question in questions:
            print(f"\nProcessing question: {question}")
            results = retriever.retrieve_information(question, corpus_id)
            
            # Assuming you want to use the first result's context for simplicity:
            if results:
                context = results[0].text
                # Iterate over available models and fetch responses
                
                # response = retriever.use_together_api(context + '\n' + question, model_info)
                # meta_data = {"model_info":model_info, "reponse": response, "context":context}
                # for model_name, model_info in model_infos.items():
                for model_name, model_info in model_infos.items():
                    print(f"Using model: {model_name}")
                    response = retriever.use_together_api(context + '\n' + question, model_info)
                    contents = response.choices[0].message.content
                    meta_data = {"model_info":model_info, "reponse": contents, "context":context}
                    temp_name_list.append(meta_data)

                        
                #     print(f"Response from {model_name}: {response.get('choices')[0]['text'] if 'choices' in response else 'No response'}")
        return temp_name_list

class EvaluationModule:
    def __init__(self, client, corpus_id, model_infos,scorer = ValidateScorer([
            # ContainsText(),
            Latency(),
            AnswerConsistency(),
            AugmentationAccuracy(),
            RetrievalPrecision(),
            AnswerSimilarityScore()
        ])):
        self.client = client
        self.corpus_id = corpus_id
        self.model_infos = model_infos
        self.corpus_ids = corpus_id
        # self.scorer = ValidateScorer([
        #     # ContainsText(),
        #     Latency(),
        #     AnswerConsistency(),
        #     AugmentationAccuracy(),
        #     RetrievalPrecision(),
        #     AnswerSimilarityScore()
        # ])
        self.scorer = scorer

    def process_queries(self, user_questions):
        retriever = Retriever(self.client)
        
        sample = retriever.prompt_generator()
        responses = []

        for question in user_questions:
            print(f"\nProcessing question: {question}")
            results = retriever.retrieve_information(question, self.corpus_id)

            if results:
                context = results[0].text
                for model_name, model_info in self.model_infos.items():
                    print(f"Using model: {model_name}")
                    formatted_prompt = retriever.prompt_formatting("Provide a detailed answer:", context, question)
                    for question in user_questions:
                        response_text = retriever.query_together_llm(
                            
                            formatted_prompt, 
                            model=model_info['model_string'],
                            tokens_limit=model_info['max_tokens'],
                            temperature=model_info['temperature'],
                            query=question
                            )
                        
                    print(f"Response from {model_name}: {response_text}")

                    # Prepare response for scoring
                    llm_response = LLMResponse(
                        llm_answer=response_text,
                        benchmark_item=(question, "Paris") # assuming 'Paris' is the correct answer for simplicity
                    )
                    responses.append(llm_response)

                    # Print response for clarity
                    print(f"Model {model_name} responded with: {response_text}")

        benchmark = self.create_benchmark([q for q in user_questions], ["Correct answer"] * len(user_questions))
        evaluation_results = self.evaluate_responses(benchmark, responses)
        return evaluation_results

if __name__ == "__main__":
    customer_id = os.getenv("VECTARA_USER_ID")  # Replace with your customer ID
    api_key = os.getenv("VECTARA_API_KEY")  # Replace with your API key
    corpus_id = os.getenv("VECTARA_CORPUS_ID")

    folder_to_process = './your_data_here'
    markdown_output_folder = './processed_markdown'

    data_loading = DataLoading(folder_path=folder_to_process, text_folder_path=markdown_output_folder)
    markdown_paths = data_loading.process_files()
    chonker = Chonker(markdown_files=markdown_paths)
    md_chunks = chonker.process_markdown_files()
    vectara_indexer = VectaraDataIndexer(customer_id, api_key)
    vectara_client = VectaraClient(customer_id, api_key)
    folder_corpus_id = vectara_indexer.create_corpus("Folder Corpus")
    markdown_corpus_id = vectara_indexer.create_corpus("Markdown Corpus")
    enriched_corpus_id = vectara_indexer.create_corpus("Enriched Markdown Corpus")
    if folder_corpus_id:
        print("Indexing entire folder...")
        vectara_indexer.index_folder(folder_corpus_id, folder_to_process)
    if markdown_corpus_id:
        print("Indexing processed Markdown chunks...")
        vectara_indexer.index_markdown_chunks(markdown_corpus_id, md_chunks)
    if enriched_corpus_id:
        print("Processing and indexing enriched Markdown chunks...")
        vectara_indexer.index_markdown_chunks_with_entities(enriched_corpus_id, md_chunks)
    # Define model information based on Together API details
    model_infos = {
        "Qwen": {
            "model_string": "Qwen/Qwen1.5-72B",
            # "max_tokens": 2000,
            "max_tokens": 10,
            "temperature": 0.7
        },
        "Meta-Llama": {
            "model_string": "meta-llama/Meta-Llama-3-70B",
            # "max_tokens": 4000,
            "max_tokens": 10,
            "temperature": 1,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.0
        }
    }
    
    # Sample questions - Place where user questions array is expected
    user_questions = [
        "What are the current trends in AI?",
        "How is climate change impacting ocean levels?",
        "Discuss the advancements in renewable energy technologies."
    ]
    
    # Process user questions
    # Retriever.process_user_questions(vectara_indexer, user_questions, corpus_id, model_infos)
    sample = Retriever.process_user_questions(vectara_client, user_questions, corpus_id, model_infos)
    # Example use of EvaluationModule
    evaluation_module = EvaluationModule(
        vectara_client, 
        corpus_id=corpus_id, 
        model_infos=model_infos,
        scorer=[AnswerConsistency()],
        )
    
    
    evaluation_module.process_queries(user_questions)
    # Continue
    publisher = VectonicPublisher()
    try:
        result = publisher.adv_publish()
        print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
# class DataLoading:
#     def __init__(self, folder_path: str):
#         self.folder_path = folder_path

#     def process_files(self):
#         """
#         Method to process each file in the specified directory.
#         """
#         # Validate if the folder exists
#         if not os.path.exists(self.folder_path):
#             print(f"Error: The folder {self.folder_path} does not exist.")
#             return

#         print(f"Processing files in folder: {self.folder_path}")
#         for root, dirs, files in os.walk(self.folder_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 try:
#                     # Use DataProcessor's static method to determine appropriate reader
#                     reader = DataProcessor.choose_reader(file_path)
#                     if reader:
#                         print(f"Processing file: {file} with {type(reader).__name__}")
#                         documents = reader.load_data(file_path)  # This assumes a `load_data` method in each reader
#                         print(f"Loaded documents from '{file}': {documents}")
#                     else:
#                         print(f"No appropriate reader found for {file}. Skipping.")
#                 except Exception as e:
#                     print(f"Error processing {file}: {str(e)}")


# if __name__ == "__main__":
#     # Instance of DataLoading, pointing to the desired directory
#     data_loader = DataLoading(folder_path='./your_data_here')
#     data_loader.process_files()
#     #Continue



# Dummy definitions for testing purposes (these should be properly imported or defined based on actual use)
# class Benchmark:
#     def __init__(self, questions, answers):
#         self.items = list(zip(questions, answers))
        
# class ValidateScorer:
#     def score_responses(self, responses):
#         print("Scoring responses...")
#         result_data = {}
#         for response in responses:
#             result_data[response.benchmark_item[0]] = {'score': 5}
# #         return result_data
# # Define custom metrics if necessary
# class CustomMetric(Metric):
#     def compute(self, ground_truth, prediction):
#         # example dummy computation
#         return abs(len(ground_truth) - len(prediction))

# # Instantiate scorer with various metrics
# scorer = ValidateScorer(metrics=[
#     AnswerSimilarityScore(),
#     RetrievalPrecision(),
#     AugmentationAccuracy(),
#     AnswerConsistency(),
#     Latency(),
#     ContainsText(),
#     CustomMetric()
# ])
# If not defined in your environment you would typically do it like this:
# # Note: This is a placeholder and actual class definitions should come from the tonic_validate library.
# class Benchmark:
#     def __init__(self, questions, answers):
#         self.questions = questions
#         self.answers = answers

# class ValidateScorer:
#     def __init__(self, metrics=None):
#         self.metrics = metrics or []

#     def score_responses(self, benchmark, responses):
#         # Dummy scoring logic
#         scored_data = []
#         for response in responses:
#             scores = {type(metric).__name__: metric.compute(response.llm_answer, response.benchmark_item[1]) 
#                       for metric in self.metrics}
#             scored_data.append((response.benchmark_item, scores))
#         return scored_data

# Redefining EvaluateModule to actually incorporate the actual ValidateScorer and Benchmark
# class EvaluateMetrics:
#     def __init__(self):
#         # Setup the metrics
#         self.metrics = [
#             self.AnswerSimilarityMetric(),
#             self.RetrievalPrecisionMetric(),
#             self.AugmentationPrecisionMetric(),
#             self.AugmentationAccuracyMetric(),
#             self.AnswerConsistencyMetric(),
#             self.LatencyMetric(),
#             self.ContainsTextMetric()
#         ]
#         self.scorer = ValidateScorer(self.metrics)

#     def evaluate(self, questions, actual_responses):
        # Create a benchmark for the evaluation
        # expected_answers = ["Paris"] * len(questions) # Example, replace with actual expected answers
        # benchmark = Benchmark(questions, expected_answers)
        
        # Create responses to score
        # responses = [
        #     LLMResponse(llm_answer=response, benchmark_item=item)
        #     for response, item in zip(actual_responses, zip(benchmark.questions, benchmark.answers))
        # ]
        
        # scored = self.scorer.score_responses(benchmark, responses)
        
        # for item, result in scored:
        #     print(f"Question: {item[0]}, Expected: {item[1]}, Scores: {result}")