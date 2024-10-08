import os
from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=groq_api_key)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to chat with PDF
def chat_with_pdf(pdf_text, user_input):
    prompt_template = PromptTemplate(
        template="You are a knowledgeable assistant. Given the following PDF content, respond to the user's question: \n\n{pdf_text}\n\nUser: {user_input}\nAssistant:",
        input_variables=["pdf_text", "user_input"]
    )
    
    # Correct usage of LLMChain with keyword arguments
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run({"pdf_text": pdf_text, "user_input": user_input})  # Use a dict for inputs
    return response

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

    # Save the uploaded PDF
    pdf_path = 'uploaded.pdf'
    file.save(pdf_path)

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    return jsonify({"message": "PDF uploaded and processed successfully.", "pdf_text": pdf_text}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("user_input")
    pdf_text = data.get("pdf_text")  # This should be the text extracted earlier.

    if not user_input or not pdf_text:
        return jsonify({"error": "Missing user input or PDF text"}), 400

    response = chat_with_pdf(pdf_text, user_input)
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    app.run(debug=True)
