
  ##Code Sample 3 
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient
import os
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class AIJudge:
    def __init__(self):
        self.LlamaModel_location = "LLama3Model/Llama3_8B_Ins/Model"
        self.LlamaTokenizer_location = "LLama3Model/Llama3_8B_Ins/Tokenizer"
        self.NLLBModel_location = "/NLLB-600/Model"
        self.NLLBTokenizer_location = "/NLLB-600/Tokenizer"

        self.Llama3Tokenizer = AutoTokenizer.from_pretrained(self.LlamaTokenizer_location)

        self.Llama3Model = AutoModelForCausalLM.from_pretrained(
            self.LlamaModel_location,
            torch_dtype=torch.bfloat16,
            device_map="auto")
        
        self.STAllMini = SentenceTransformer('/model_ST_AllMini')
        self.STAllMiniDEF = SentenceTransformer("/model_ST_ALLMiniDIfferentiator")
        self.UrduFlag = False
        self.EngFlag = False
        self.NLLBTokenizer = AutoTokenizer.from_pretrained(self.NLLBTokenizer_location)
        self.NLLBModel = AutoModelForSeq2SeqLM.from_pretrained(self.NLLBModel_location)
        self.chunk_size = 200
        self.urdu_range = (0x0600, 0x06FF)
        self.drt_path = '/judicary_backend/Embeddings (complete  case)'

    def generateJudgment(self, text):

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        messages = [
            {"role": "system", "content": f" "},
            {"role": "user", "content": text},]
        self.Llama3Tokenizer = AutoTokenizer.from_pretrained(self.LlamaTokenizer_location)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = self.Llama3Tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        terminators = [
            self.Llama3Tokenizer.eos_token_id,
            self.Llama3Tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        print("here1")
        self.Llama3Model.to(device)
        print("here2")
        outputs = self.Llama3Model.generate(
            input_ids,


            max_new_tokens=800,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.6,
        )
        response = outputs[0][input_ids.shape[-1]:]
        ald = self.Llama3Tokenizer.decode(response, skip_special_tokens=True)
        del self.Llama3Model
        del self.Llama3Tokenizer
        torch.cuda.empty_cache()
        return ald

    def translate_long_text(self,long_text):
        if self.UrduFlag ==True:
            lg_code="eng_Latn"
            sentences = long_text.split("۔")

        else :
            lg_code = "urd_Arab"
            sentences = long_text.split(".")
        #sentences = long_text.split(".")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        inputs = self.NLLBTokenizer(sentences, return_tensors="pt",padding=True)
        inputs.to(device)
        self.NLLBModel.to(device)
        translated_tokens = self.NLLBModel.generate(
        **inputs, forced_bos_token_id=self.NLLBTokenizer.lang_code_to_id[lg_code], max_length=300)
        translated_sentence = self.NLLBTokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[:]
        translated_article = " ".join(translated_sentence)
        print(translated_article )
        return translated_article

    def is_urdu(self,text):
    # Urdu Unicode range
    
        print(text)
        # Check if any character in the text falls within the Urdu Unicode range
        for char in text:
            if ord(char) >= self.urdu_range[0] and ord(char) <= self.urdu_range[1]:
                return True
        return False

    def Retrieve_Summaries(self,filesnames):
     

        # Connect to MongoDB
        client = MongoClient('')

        # Access the database
        db = client.get_default_database()  # Assuming 'Judiciary_Database' is the default database

        # Access the 'cases' collection
        cases_collection = db['case']


        summaries = []
        for filename in filesnames:
                # Find cases with matching filename (case-insensitive)
            matching_cases = cases_collection.find({'FileName': {'$regex': filename, '$options': 'i'}})

                # Extract and append summaries
            for case in matching_cases:
                summaries.append(case['ExtractiveSummary'])
        return summaries

    def _compute_embedding(self,text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.STAllMini.to(device)
        embedding = self.STAllMini.encode(text, convert_to_tensor=True).tolist()
        return embedding
    
    def _extract_file_names(self,data):
        file_names = []
        for item in data:
            file_name = item[0].split('_')[0]  # Splitting and extracting the first part
            file_names.append(file_name)
        return file_names

    def compare_embeddings_with_files(self,input_text):
        # Compute embedding for input text
        input_embedding = self._compute_embedding(input_text)
        directory_path = self.drt_path
        
        # List to store file names and their scores
        file_scores = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assuming input_embedding and file_embeddings are on GPU
        input_embedding = torch.tensor(input_embedding, dtype=torch.float32, device=device)

        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        file_scores = []

        for filename in tqdm(files, desc="Comparing embeddings", unit="file"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    file_embedding = json_data.get('embedding')  # Assuming the key for embedding is 'embedding'
                    if file_embedding is not None:
                        # Calculate cosine similarity between input embedding and file embedding
                        file_embedding = torch.tensor(file_embedding, dtype=torch.float32, device=device)
                        score = torch.dot(input_embedding, file_embedding) / (torch.norm(input_embedding) * torch.norm(file_embedding))
                        score = score.item()  # Convert tensor to scalar
                        file_scores.append((filename, score))
            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")

        # Sort files based on scores
        file_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 10 files with scores
        return self._extract_file_names(file_scores[:8])

    

    def ProcessPrompt(self,prompt):
        self.UrduFlag= self.is_urdu(prompt)
        print(self.UrduFlag)
        if self.UrduFlag  == True:
            prompt = self.translate_long_text(prompt)
            prompt += '۔شکری. ہ.'

            self.EngFlag =True
            self.UrduFlag = False

        sim_cases = self.compare_embeddings_with_files(prompt)
        complete_prompt = ""
        generatedOutput = self.generateJudgment(complete_prompt)
        if self.EngFlag == True:
            generatedOutput = self.translate_long_text( generatedOutput )
            generatedOutput = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'\n\n**\1**\n\n', generatedOutput )
        print("==========================================================================================================================================================================")
        print("==========================================================================================================================================================================")
        return sim_cases,generatedOutput



#####################################################################################################################################
#####################################################################################################################################