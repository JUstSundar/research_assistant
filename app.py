from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains.llm_math.base import LLMMathChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# === Flask setup ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === LangChain setup ===
llm = OllamaLLM(
    model="tinyllama",
    base_url="http://localhost:11434",
    temperature=0.7
)
embeddings = OllamaEmbeddings(model="tinyllama")

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Retrieval-based QA
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Calculator tool
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
calculator_tool = Tool.from_function(
    name="Calculator",
    func=math_chain.run,
    description="Use this for answering math questions"
)

# Retrieval tool (from PDF)
retrieval_tool = Tool.from_function(
    name="ResearchDocs",
    func=retrieval_chain.run,
    description="Use this to answer questions based on uploaded research papers or documents"
)



tools = [calculator_tool, retrieval_tool, search_tool]

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    chain_type="stuff",
    max_iterations=3,
    return_source_documents=True
)

# === Upload PDF Endpoint ===
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and split PDF
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Add to vectorstore
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return jsonify({"message": f"'{filename}' uploaded and processed successfully."})


# === Ask Research Question Endpoint ===
@app.route("/research", methods=["POST"])
def research():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "Empty query"}), 400

        # Run query through the reasoning agent
        result = agent.run(query)
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
