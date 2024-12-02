import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)



class AdvancedQNAChatbot:
    def __init__(self, json_path, model_name='deepset/roberta-base-squad2'):
        """
        Initialize the advanced Q&A chatbot with ML-powered answer retrieval.
        
        :param json_path: Path to the JSON file containing questions and answers
        :param model_name: Pre-trained question answering model
        """
        # Load training data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.qa_pairs = data.get('data', [])
        
        # Prepare questions and answers
        self.questions = [qa['question'] for qa in self.qa_pairs]
        self.answers = [qa['answer'] for qa in self.qa_pairs]
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
        # Initialize NLP model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.model, 
            tokenizer=self.tokenizer
        )
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing and removing stopwords.
        
        :param text: Input text
        :return: Preprocessed text
        """
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        processed_text = ' '.join([w for w in tokens if w not in stop_words])
        return processed_text
    
    def find_similar_question(self, user_question):
        """
        Find the most similar question using TF-IDF and cosine similarity.
        
        :param user_question: User's input question
        :return: Most similar question and its index
        """
        # Vectorize the user question
        user_vector = self.vectorizer.transform([user_question])
        
        # Compute cosine similarities
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]
        
        # Find the index of the most similar question
        most_similar_index = similarities.argmax()
        similarity_score = similarities[most_similar_index]
        
        return most_similar_index, similarity_score
    
    def generate_answer(self, user_question):
        """
        Generate an answer using multiple strategies.
        
        :param user_question: User's input question
        :return: Generated answer
        """
        # Find most similar question
        similar_index, similarity_score = self.find_similar_question(user_question)
        
        # If similarity is high enough, return corresponding answer
        if similarity_score > 0.5:
            return self.answers[similar_index]
        
        # Try question-answering pipeline with context
        try:
            # Use the most similar question's context
            context = self.questions[similar_index] + " " + self.answers[similar_index]
            qa_result = self.qa_pipeline({
                'question': user_question,
                'context': context
            })
            
            # If pipeline confidence is high, return its answer
            if qa_result['score'] > 0.5:
                return qa_result['answer']
        except Exception as e:
            print(f"QA Pipeline error: {e}")
        
        # Fallback: return a default message
        return "I'm sorry, I couldn't find a precise answer to your question."
    
    def interactive_chat(self):
        """
        Interactive chat interface with the Q&A bot.
        """
        print("Advanced Q&A Chatbot: Hello! I'm ready to answer your questions. Type 'exit' to quit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Advanced Q&A Chatbot: Goodbye!")
                break
            
            response = self.generate_answer(user_input)
            print(f"Chatbot: {response}")
    
    def batch_evaluate(self, test_questions):
        """
        Batch evaluation of the chatbot's performance.
        
        :param test_questions: List of test questions
        :return: Performance metrics
        """
        correct_answers = 0
        total_questions = len(test_questions)
        
        for test_q in test_questions:
            # In a real-world scenario, you'd have ground truth answers
            generated_answer = self.generate_answer(test_q)
            # Add your evaluation logic here
            # For example, checking against a predefined set of answers
            correct_answers += 1  # Placeholder
        
        accuracy = correct_answers / total_questions
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy
        }

# Example Usage
if __name__ == "__main__":
    # Path to your JSON file
    chatbot = AdvancedQNAChatbot('data_set.json')
    
    # Start interactive chat
    chatbot.interactive_chat()