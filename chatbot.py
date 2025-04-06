import nltk
import random
import string 
import warnings
import nltk.stem.wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

CORPUS = """
Hello! I am your friendly chatbot TechBot.
I can talk about Python, AI, and Machine Learning.
Python is a popular programming language.
It is used in web development, data science and AI.
AI stands for Artificial Intelligence.
Artificial Intelligence enables machines to think like humans.
Machine Learning is a subset of AI that uses data to learn patterns.
Tell me something about python.
What is machine learning?
How is AI used in real life?
Explain deep learning.
How do I start learning python?
Can machines learn?
What is the use of python in AI?
"""

sent_tokens=nltk.sent_tokenize(CORPUS)
lemmer=nltk.stem.WordNetLemmatizer()

def LemNormalize(text):
    tokens=nltk.word_tokenize(text.lower())
    return [lemmer.lemmatize(token) for token in tokens if token not in string.punctuation]

def response(user):
   sent_tokens.append(user)
   vectorizer=TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
   tfidf=vectorizer.fit_transform(sent_tokens)
   similarity=cosine_similarity(tfidf[-1],tfidf[:-1])
   idx=similarity.argsort()[0][-1]
   flat=similarity.flatten()
   flat.sort()
   score=flat[-1]
   sent_tokens.pop()
   if score>0.3:
       return sent_tokens[idx]
   else:
       user=user.lower()
       if 'python' in user:
           return "Python is great for web apps, AI, scripting, and more."
       elif 'ai' in user or 'artificial' in user:
           return "AI is a branch of computer science focused on smart machines."
       elif 'machine learning' in user or 'ml' in user:
           return "Machine Learning lets system learn from data."
       else:
           return "I'm sorry, I didn't understand that. Try asking about Python, AI or ML."
   
def chat():
    print("TechBot: Hello! Ask me something. Type 'bye' to exit. ")
    while True:
        user=input("You: ")
        if user.lower() in ['bye','exit','quit']:
            print("TechBot: Goodbye! Have a nice day.")
            break
        else:
            print("TechBot:",response(user))

chat()          