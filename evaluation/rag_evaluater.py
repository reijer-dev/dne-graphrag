from evaluation.rag_system import RagSystem
from evaluation.thuisarts_scraper import ThuisartsScraper
import os, os.path
import time
import csv
import google.generativeai as genai
import json
import numpy as np
import random 
import concurrent.futures
import warnings

class RagEvaluater:

    def __init__(self, count=None):
        self.rag_systems = []
        self.documents_subdir = "evaluation/documents"
        self.questions_file = "evaluation/questions.json"
        self.memory_used = {}
        self.training_times = {}
        self.inference_times = {}
        self.inference_doc_scores = {}
        self.inference_global_scores = {}
        self.count = count
        # Configure your API key
        apikey = open("evaluation/apikey.txt", "r").read()
        genai.configure(api_key=apikey) # Replace with your actual API key

        # Set up the model
        self.gemini = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro-vision' for multimodal models

        #Disable some annoying warnings
        warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
        self.question_answer_pairs = []
    
    def add_system(self, rag_system: RagSystem):
        self.rag_systems.append(rag_system)

    def evaluate_training(self):
        self.__ensure_documents()
        all_docs = os.listdir(self.documents_subdir)
        zero_docs = [x for x in all_docs if x.endswith('-0.txt')]
        other_docs = [x for x in all_docs if not x.endswith('-0.txt')]
        all_docs = zero_docs + other_docs[:100]
        for rag_system in self.rag_systems:
            i = 0
            if not os.path.exists(rag_system.base_directory):
                os.makedirs(rag_system.base_directory)
            initial_size = self.__get_dir_size(rag_system.base_directory)
            initial_time = time.time()
            for document in all_docs:
                i += 1
                print(f'Processing document {i}/{len(all_docs)}')
                rag_system.process_document(f'{self.documents_subdir}/{document}')
            rag_system.save()
            
            posterior_size = self.__get_dir_size(rag_system.base_directory)
            posterior_time = time.time()
            self.memory_used[rag_system.name] = posterior_size - initial_size
            self.training_times[rag_system.name] = posterior_time - initial_time
            print(f'{rag_system.name} used {self.memory_used[rag_system.name]/1000} KB and {self.training_times[rag_system.name]} seconds')
        
        with open(f"evaluation-training-{time.time()}.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["rag_system","doc_amount","memory_used","training_time"])
            for rag_system in self.rag_systems:
                writer.writerow([rag_system.name,len(all_docs),self.memory_used[rag_system.name],self.training_times[rag_system.name]])
        
    def evaluate_inference(self) -> str:
        self.__ensure_questions()
        questions = json.loads(open(self.questions_file, "r").read())
        questions = random.sample(questions, self.count) if self.count < len(questions) else questions

        for rag_system in self.rag_systems:
            i = 0
            self.inference_doc_scores[rag_system.name] = []
            self.inference_global_scores[rag_system.name] = []
            self.inference_times[rag_system.name] = []

            #with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for question in questions:
                i += 1
                self.evaluate_question(question, len(questions), i, rag_system)
                    #executor.submit(self.evaluate_question, question, len(questions), i, rag_system)

        csv_file = f"evaluation-inference-{time.time()}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["rag_system","avg_time", "doc_questions", "avg_doc_score", "global_questions", "avg_global_score"])
            for rag_system in self.rag_systems:
                writer.writerow([rag_system.name,self.np_mean(self.inference_times[rag_system.name]),len(self.inference_doc_scores[rag_system.name]),self.np_mean(self.inference_doc_scores[rag_system.name]),len(self.inference_global_scores[rag_system.name]),self.np_mean(self.inference_global_scores[rag_system.name])])

        #Export raw question/answers
        with open(csv_file.replace('.csv', '.txt'), "w", newline="", encoding="utf-8") as txt_file:
            for pair in self.question_answer_pairs:
                txt_file.write(pair)
                txt_file.write('\n-----------------------------------------\n')

        print('Evaluation done!')
        return csv_file

    def evaluate_question(self, question, len_questions, i, rag_system):
        print(f'Processing question {i}/{len_questions}', end='\r')

        initial_time = time.time()
        answer = rag_system.query(question.get('question'))
        posterior_time = time.time()
        self.inference_times[rag_system.name].append(posterior_time - initial_time)
        if not question.get('doc') is None:
            if answer is None or len(answer) == 0:
                self.inference_doc_scores[rag_system.name].append(0)
            else:
                doc_text = open(f"{self.documents_subdir}/{question['doc']}", "r").read()
                validation_question = f"I'm testing my medical chatbot, which is a RAG system. I have a source document, put at the end between brackets. I asked my chatbot the question (\"{question}\"), and got this answer: \"{answer}\". Please rate this answer on correctness/completeness between 0 and 100. Respond with only the score as an integer. Document: [{doc_text}]"
                validation_answer = self.gemini.generate_content(validation_question).text
                score = int(validation_answer)
                self.inference_doc_scores[rag_system.name].append(score)
                self.question_answer_pairs.append(f'System: {rag_system.name}\nQuestion: {question.get('question')}\nAnswer: {answer}\nScore: {score}')
        elif not question.get('answer') is None:
            if answer is None or len(answer) == 0:
                self.inference_global_scores[rag_system.name].append(0)
            else:
                validation_question = f"I'm testing my medical chatbot, which is a RAG system. I asked my chatbot the question (\"{question}\"), and got this answer: \"{answer}\". The correct answer is: \"{question.get('answer')}\". Please rate the RAG system's answer on correctness/completeness between 0 and 100. Respond with only the score as an integer."
                validation_answer = self.gemini.generate_content(validation_question).text
                score = int(validation_answer)
                self.inference_global_scores[rag_system.name].append(score)    
                self.question_answer_pairs.append(f'System: {rag_system.name}\nQuestion: {question.get('question')}\nAnswer: {answer}\nScore: {score}')

    def __ensure_questions(self):
        questions = json.loads(open(self.questions_file, "r").read()) if os.path.exists(self.questions_file) else []
        if len(questions) > 0:
            return
        
        #Generate 50 global questions
        print(f'Generating 50 global questions...')
        all_0_texts = [open(f'{self.documents_subdir}/{x}', "r").read() for x in os.listdir(self.documents_subdir) if x.endswith('-0.txt')]
        separator = '\n---\n'
        joined_texts = separator.join(all_0_texts)
        initial_question = f'''I'm testing my medical chatbot, which is a RAG system, on its global answering abilities. I have a set of introductions per topic, put at the end between brackets. Please ask me 50 very simple global questions and answers about these topics. The question should require knowledge of dozens of documents to answer it. You should only output the questions and answers as a JSON list. Format: [{{"question":"Question", "answer": "Answer"}}]. NB: Answer in Dutch. Topics: [{joined_texts}]'''
        global_qas_json = self.gemini.generate_content(initial_question).text.replace('```', '')[4:]
        print(global_qas_json)
        global_qas = json.loads(global_qas_json)
        for qa in global_qas:
            questions.append({'question': qa.get('question'), 'answer': qa.get('answer')})

        #Generate 100 document-based questions
        all_docs = os.listdir(self.documents_subdir)
        all_docs = [x for x in all_docs if not x.endswith('-0.txt')][:100]
        i = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            print(f'Generating 100 doc questions...')
            for doc in all_docs:
                executor.submit(self.generate_doc_question, doc, questions, len(all_docs))              

        with open(self.questions_file, "w") as f:
            f.write(json.dumps(questions))
    
    def generate_doc_question(self, doc, questions, total_docs):
        print(f'Generating doc questions {len(questions)}/{total_docs}', end='\r')
        text = open(f'{self.documents_subdir}/{doc}', "r").read()
        initial_question = f"I'm testing my medical chatbot, which is a RAG system. I have a source document, put at the end between brackets. Please ask me a question about the document. The question should be standalone, so do not refer to the document in the question, because my chatbot does not know which document the question is about. You should only output the question as a single line. NB: Answer in Dutch. [{text}]"
        question1 = self.gemini.generate_content(initial_question).text
        questions.append({'doc': doc, 'question': question1})
        i += 1

    def __ensure_documents(self):
        print(self.documents_subdir)
        amount_of_documents = len([name for name in os.listdir(self.documents_subdir)])
        print(amount_of_documents)
        if amount_of_documents == 0:
            ThuisartsScraper().download_documents(self.documents_subdir)
        return
    
    def __get_dir_size(self, start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size
    
    def np_mean(self, values):
        if len(values) == 0:
            return 0
        return np.mean(values)
