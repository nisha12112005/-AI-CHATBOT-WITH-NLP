import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

# Download necessary NLTK data (run only once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Intent patterns with corresponding responses
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "take care"],
    "thanks": ["thanks", "thank you", "thx"],
    "name": ["what is your name", "who are you", "your name"],
    "weather": ["what is the weather", "how's the weather", "tell me the weather"],
    "age": ["how old are you", "your age"]
}

# Response database for each intent
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!", "Greetings!"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Glad to help!"],
    "name": ["I'm a chatbot built with NLTK.", "My name is Chatbot, nice to meet you!"],
    "weather": ["I'm not connected to a weather service, but I hope it's nice outside!"],
    "age": ["I don't age. I am just a program.", "I don't have an age, I'm just a bot!"]
}

# Preprocessing function: Tokenize and lemmatize input text
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize the sentence
    return [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each token

# Match input to intent based on keywords
def predict_intent(user_input):
    tokens = preprocess(user_input)
    
    # Check for matching intent
    for intent, patterns in intents.items():
        for pattern in patterns:
            pattern_tokens = preprocess(pattern)
            # Check if all pattern tokens are in the user input tokens
            if all(token in tokens for token in pattern_tokens):
                return intent
    
    return "unknown"  # If no match is found

# Chatbot interaction loop
def chatbot():
    print("🤖 NLTK Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot:", random.choice(responses["goodbye"]))
            break
        
        # Get the intent and respond accordingly
        intent = predict_intent(user_input)
        
        if intent != "unknown":
            print("Bot:", random.choice(responses[intent]))
        else:
            print("Bot: I'm not sure how to respond to that.")

# Run chatbot
if __name__ == "__main__":
    chatbot()
