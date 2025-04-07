import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure the Gemini API with hardcoded key
GOOGLE_API_KEY = "AIzaSyDgT4U44HLCHk5j8gH4UPGHKPcE-fGlZRI"
genai.configure(api_key=GOOGLE_API_KEY)

# Use Gemini 1.5 Flash model
model_name = "gemini-1.5-flash"
print(f"Using model: {model_name}")

# Initialize the model
model = genai.GenerativeModel(model_name)

def chat_with_gemini():
    # Initialize chat
    chat = model.start_chat(history=[])
    
    print("\nWelcome to Gemini Chatbot! (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Have a great day!")
            break
            
        try:
            # Get response from Gemini
            response = chat.send_message(user_input)
            print("\nGemini:", response.text)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    chat_with_gemini() 