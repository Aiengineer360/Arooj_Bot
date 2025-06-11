from flask import Flask, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import re

# Load environment variables
env_path = Path('config.env')
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OpenAI API key found. Please check your config.env file.")

app = Flask(__name__)

# Enhanced greeting and conversation patterns
CONVERSATION_PATTERNS = {
    # General greetings
    r'\b(hi|hello|hey|greetings)\b': [
        "Hello! 👋 I'm Arooj's AI assistant. I'd be happy to tell you about her work, projects, and experience!",
        "Hi there! I'm here to share information about Arooj's professional journey and achievements.",
        "Greetings! I can help you learn about Arooj's background in technology and development."
    ],
    
    # How are you variations
    r'\b(how are you|how\'s it going|how do you do)\b': [
        "I'm doing great, thank you for asking! I'd love to tell you about Arooj's expertise in AI, web development, and her exciting projects.",
        "Thanks for asking! I'm here and ready to share information about Arooj's professional experience and achievements.",
        "I'm excellent! I specialize in discussing Arooj's work in technology, her projects, and educational background. What would you like to know?"
    ],
    
    # Goodbyes
    r'\b(bye|goodbye|see you|farewell)\b': [
        "Goodbye! Feel free to return if you'd like to learn more about Arooj's work and experience. Have a great day! 👋",
        "Thanks for chatting! Don't hesitate to come back if you have more questions about Arooj's professional journey.",
        "Farewell! I'm always here to share information about Arooj's expertise and achievements."
    ],
    
    # Thank you variations
    r'\b(thanks|thank you|appreciate it)\b': [
        "You're welcome! I'm glad I could help share information about Arooj's work and experience.",
        "My pleasure! Don't hesitate to ask if you'd like to know more about Arooj's projects and expertise.",
        "You're most welcome! Feel free to ask more about Arooj's professional background anytime."
    ]
}

# Common irrelevant topics and polite redirections
IRRELEVANT_TOPICS = {
    r'\b(food|eat|drink)\b': "I apologize, but I don't discuss personal preferences. However, I'd be happy to tell you about Arooj's technical projects and professional experience!",
    r'\b(weather|climate)\b': "While I can't discuss the weather, I can tell you all about Arooj's achievements in technology and development. Would you like to know about her projects?",
    r'\b(movie|music|song|film)\b': "I focus on Arooj's professional work rather than entertainment. Would you like to hear about her technical projects or educational background?",
    r'\b(sports|game|play)\b': "I specialize in discussing Arooj's professional work in technology. Let me know if you'd like to learn about her projects or expertise!",
}

# Load and process PDF
def initialize_pdf_knowledge_base():
    # Load PDF
    loader = PyPDFLoader("my_paper.pdf")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Create retrieval chain
    llm = ChatOpenAI(
        temperature=0.7,
        model_name='gpt-3.5-turbo',
        openai_api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return retrieval_chain

# Initialize the retrieval chain
qa_chain = initialize_pdf_knowledge_base()

def get_random_response(responses):
    """Get a random response from a list of possible responses"""
    from random import choice
    return choice(responses if isinstance(responses, list) else [responses])

def check_conversation_patterns(message):
    """Check message against conversation patterns and return appropriate response"""
    message = message.lower().strip()
    
    # Check for greetings and common patterns
    for pattern, responses in CONVERSATION_PATTERNS.items():
        if re.search(pattern, message):
            return get_random_response(responses)
    
    # Check for irrelevant topics
    for pattern, response in IRRELEVANT_TOPICS.items():
        if re.search(pattern, message):
            return response
            
    return None

def get_ai_response(message):
    try:
        # Check for conversation patterns first
        pattern_response = check_conversation_patterns(message)
        if pattern_response:
            return pattern_response

        # Get response from the QA chain
        result = qa_chain.invoke({"question": message})
        response = result['answer']

        # Check if the response indicates no relevant information
        no_info_indicators = ["don't know", "don't have", "no information", "cannot", "unable to", "not able to"]
        if any(indicator in response.lower() for indicator in no_info_indicators):
            return (
                "I apologize, but I can only provide information about Arooj's professional background, including:\n\n"
                "🎓 Education and academic achievements\n"
                "💻 Technical projects and developments\n"
                "🚀 Professional experiences\n"
                "💡 Skills and expertise\n"
                "🔬 Research interests\n\n"
                "Please feel free to ask about any of these topics!"
            )

        # Enhance the response with a professional context
        enhanced_response = f"{response}\n\nThis information is based on Arooj's documented experience and achievements in technology and development."
        return enhanced_response

    except Exception as e:
        return "I apologize, but I encountered an error. Please try asking your question in a different way or ask about Arooj's professional background and projects."

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/bot.css')
def serve_css():
    return send_from_directory('.', 'bot.css')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get AI response
        ai_response = get_ai_response(user_message)

        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)