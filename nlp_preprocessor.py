import re
import nltk
import pandas as pd
import torch
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import random
import difflib

class NLPPreprocessor:
    def __init__(self, model_dir=None, dataset_path='sampel.csv'):
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            print("NLTK resources download might have issues")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.stop_words = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'ini', 'itu'}
        
        if model_dir and os.path.exists(model_dir):
            print(f"Loading model from {model_dir}")
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForQuestionAnswering.from_pretrained(model_dir)
        else:
            print("Loading pretrained model")
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        
        self.dataset = pd.read_csv(dataset_path)
        
        self.answer_templates = {
            'apa': [
                "Berdasarkan informasi yang tersedia, {answer}.",
                "{answer} adalah jawaban untuk pertanyaan Anda.",
                "Menurut data kami, {answer}."
            ],
            'bagaimana': [
                "Untuk {question_keyword}, Anda perlu {answer}.",
                "Caranya adalah dengan {answer}.",
                "Langkah yang bisa diambil adalah {answer}."
            ],
            'berapa': [
                "Dari informasi yang ada, {answer}.",
                "Jumlahnya adalah {answer}.",
                "{answer} berdasarkan data kami."
            ],
            'kapan': [
                "Jadwalnya adalah {answer}.",
                "{answer} adalah waktunya.",
                "Berdasarkan informasi, {answer}."
            ],
            'siapa': [
                "{answer} adalah orang yang Anda tanyakan.",
                "Menurut informasi, {answer}.",
                "Orangnya adalah {answer}."
            ],
            'default': [
                "Berdasarkan informasi yang tersedia, {answer}.",
                "Jawabannya adalah {answer}.",
                "{answer}."
            ]
        }
        
        self.connectors = [
            "dan", "kemudian", "setelah itu", "selain itu", "juga", 
            "selanjutnya", "dengan", "untuk", "sebagai"
        ]

    def find_best_context(self, query):
        """Cari konteks paling sesuai dari dataset"""
        query_lower = query.lower()
        
        # Tambahkan pengecekan dataset kosong
        if self.dataset.empty:
            print("[DEBUG] Dataset kosong")
            return "Tidak ditemukan konteks yang sesuai."
            
        if 'question' not in self.dataset.columns:
            print("[DEBUG] Kolom 'question' tidak ditemukan dalam dataset")
            return "Tidak ditemukan konteks yang sesuai."

        def similarity_score(row):
            if isinstance(row.get('question'), str):
                similarity = difflib.SequenceMatcher(None, query_lower, row['question'].lower()).ratio()
                return similarity
            return 0

        try:
            self.dataset['similarity'] = self.dataset.apply(similarity_score, axis=1)
            
            if len(self.dataset) == 0:
                print("[DEBUG] Dataset kosong setelah diproses")
                return "Tidak ditemukan konteks yang sesuai."
                
            best_match = self.dataset.sort_values('similarity', ascending=False).iloc[0]

            print(f"\n[DEBUG] Query: {query}")
            print(f"[DEBUG] Best Match: {best_match}")
            print(f"[DEBUG] Best Match Type: {type(best_match)}")

            if 'context' in best_match and isinstance(best_match['context'], str):
                print(f"[DEBUG] Context: {best_match['context']}")
                return best_match['context']
            else:
                print("[DEBUG] Konteks tidak ditemukan dalam best_match atau bukan string")
                return "Tidak ditemukan konteks yang sesuai."
                
        except Exception as e:
            print(f"[DEBUG] Error dalam find_best_context: {e}")
            return "Tidak ditemukan konteks yang sesuai."

    def process_query(self, query, context=None, optimize_answer=True):
        """Process a single query"""
        try:
            if context is None:
                context = self.find_best_context(query)
                
            if not isinstance(context, str):
                print(f"[DEBUG] Context bukan string: {type(context)}")
                context = "Tidak ditemukan konteks yang sesuai."
                
            print(f"[DEBUG] Context: {context}")
            
            if "Tidak ditemukan" in context:
                return {
                    "original_query": query, 
                    "context": context, 
                    "raw_answer": "", 
                    "optimized_answer": context
                }
                
            inputs = self.tokenizer(query, context, return_tensors='pt')
            outputs = self.model(**inputs)

            answer_start = outputs.start_logits.argmax()
            answer_end = outputs.end_logits.argmax()

            if answer_end < answer_start:
                answer_end = answer_start

            raw_answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1])
            )

            final_answer = self.optimize_answer(query, raw_answer, context) if optimize_answer else raw_answer

            return {
                "original_query": query,
                "context": context,
                "raw_answer": raw_answer,
                "optimized_answer": final_answer
            }
            
        except Exception as e:
            print(f"[DEBUG] Error dalam process_query: {e}")
            return {
                "original_query": query,
                "context": "Terjadi kesalahan saat memproses pertanyaan",
                "raw_answer": "",
                "optimized_answer": f"Mohon maaf, terjadi kesalahan: {str(e)}"
            }
    
    def get_question_type(self, query):
        """Menentukan jenis pertanyaan berdasarkan kata tanya"""
        query_lower = query.lower()
        
        if 'apa' in query_lower:
            return 'apa'
        elif 'bagaimana' in query_lower:
            return 'bagaimana'
        elif 'berapa' in query_lower:
            return 'berapa'
        elif 'kapan' in query_lower:
            return 'kapan'
        elif 'siapa' in query_lower:
            return 'siapa'
        else:
            return 'default'
    
    def extract_keyword(self, query):
        """Ekstrak keyword utama dari pertanyaan"""
        question_words = ['apa', 'bagaimana', 'berapa', 'kapan', 'siapa', 'dimana', 'mengapa']
        query_words = query.lower().split()
        
        for word in question_words:
            if word in query_words:
                query_words.remove(word)
        
        query_words = [word for word in query_words if word not in self.stop_words]
        
        if query_words:
            return ' '.join(query_words[:2])  
        return ""
    
    def optimize_answer(self, query, raw_answer, context):
        """Meningkatkan kualitas jawaban dengan menyusun ulang dan menambahkan kata-kata yang sesuai"""
        clean_answer = raw_answer.strip()
        clean_answer = re.sub(r'\s+', ' ', clean_answer)
        
        if len(clean_answer) < 2 or clean_answer.lower() in ['', '[cls]', '[sep]']:
            sentences = re.split(r'[.!?]', context)
            query_keywords = [word for word in query.lower().split() if word not in self.stop_words]
            
            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in query_keywords) and len(sentence) > 10:
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                clean_answer = relevant_sentences[0]
        
        if len(clean_answer) < 2:
            clean_answer = "Mohon maaf, silahkan hubungi helpdesk untuk pertanyaan terkait hal tersebut."
            return clean_answer
        
        question_type = self.get_question_type(query)
        templates = self.answer_templates.get(question_type, self.answer_templates['default'])
        
        keyword = self.extract_keyword(query)
        
        template = random.choice(templates)
        
        if question_type == 'bagaimana' and keyword:
            formatted_answer = template.format(question_keyword=keyword, answer=clean_answer)
        else:
            formatted_answer = template.format(answer=clean_answer)
            
        return formatted_answer

    def save_model(self, output_dir="./saved_model"):
        """Save the trained model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    def train_model(self, dataset_path, output_dir="./saved_model"):
        """Train the BERT model on a dataset and save it"""
        df = pd.read_csv(dataset_path)
        
        df['answer'] = df['context'].apply(lambda x: x.split('.')[0])
        
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        def preprocess_function(examples):
            questions = examples['question']
            contexts = examples['context']
            answers = examples['answer']
            
            tokenized = self.tokenizer(
                questions,
                contexts,
                truncation=True,
                padding=True,
                max_length=512,
                return_offsets_mapping=True
            )
            
            start_positions = []
            end_positions = []
            
            for i, (context, answer) in enumerate(zip(contexts, answers)):
                answer_start = context.lower().find(answer.lower())
                answer_end = answer_start + len(answer)
                
                offset_mapping = tokenized.offset_mapping[i]
                
                start_token = 0
                end_token = 0
                
                for j, (start, end) in enumerate(offset_mapping):
                    if start <= answer_start < end:
                        start_token = j
                    if start < answer_end <= end:
                        end_token = j
                
                start_positions.append(start_token)
                end_positions.append(end_token)
            
            tokenized.pop("offset_mapping")
            
            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions
            
            return tokenized
        
        tokenized_train = train_dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        tokenized_test = test_dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=test_dataset.column_names
        )
        
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")
        print(f"Loss: {eval_results['eval_loss']}")
        
        self.save_model(output_dir)
        
        return eval_results