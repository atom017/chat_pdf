import os
from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

app = Flask(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=groq_api_key)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_stream):
    text = ""
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to chat with PDF
def chat_with_pdf(pdf_text, user_input):
    prompt_template = PromptTemplate(
        template="You are a knowledgeable assistant. Please answer the user's question concisely and insightfully, without referring to any external text.: \n\n{pdf_text}\n\nUser: {user_input}\nAssistant:",
        input_variables=["pdf_text", "user_input"]
    )
    
    # Correct usage of LLMChain with keyword arguments
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"pdf_text": pdf_text, "user_input": user_input})
    response_text = response.content if hasattr(response, 'content') else str(response)
    return response_text

def chunk_text(text, max_length=1000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Maximum chunk size
    chunk_overlap=150,   # overlap between chunks
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    pdf_bytes = BytesIO(file.read())
    #pdf_path = 'uploaded.pdf'
    file.save(pdf_bytes)

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_bytes)
    return jsonify({"message": "PDF uploaded and processed successfully.", "pdf_text": pdf_text}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("user_input")
    pdf_text = data.get("pdf_text")

    if not user_input or not pdf_text:
        return jsonify({"error": "Missing user input or PDF text"}), 400

    # Split the PDF text into chunks using RecursiveCharacterTextSplitter
    text_chunks = text_splitter.split_text(pdf_text)

    # Limit to a maximum number of chunks (e.g., 5 chunks)
    max_chunks = 5
    limited_chunks = text_chunks[:max_chunks]

    responses = []
    for chunk in limited_chunks:
        response = chat_with_pdf(chunk, user_input)
        responses.append(response.strip())

    # Combine all responses into a single string
    combined_response = " ".join(responses)

    # Generate a final response based on the combined responses
    final_response = generate_final_response(combined_response, user_input)

    return jsonify({"response": final_response}), 200

def generate_final_response(combined_response, user_input):
    prompt_template1 = PromptTemplate(
        
        template="Based on the following information, provide a concise and comprehensive answer to the user's query:\n\n{combined_response}\n\nUser: {user_input}\nAssistant:",
        input_variables=["combined_response", "user_input"]
    )
    prompt_template2 = PromptTemplate(
        template=(
            "Based on the following information, please summarize the key points "
            "that answer the user's question:\n\n{combined_response}\n\n"
            "User's Question: {user_input}\n\nAnswer:"
        ),
        input_variables=["combined_response", "user_input"]
    )
    
    prompt_template = PromptTemplate(
        template=(
            "Using the information provided below, craft a concise and informative response that "
            "addresses the user's question while incorporating the relevant details from the text:\n\n"
            "{combined_response}\n\n"
            "User's Question: {user_input}\n\n"
            "Response:"
        ),
        input_variables=["combined_response", "user_input"]
    )
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"combined_response": combined_response, "user_input": user_input})
    
    return response.content if hasattr(response, 'content') else str(response)

if __name__ == '__main__':
    app.run(debug=True)
