from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize LLM and Embeddings
llm = Ollama(model="tinyllama", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="tinyllama")
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize Vector Store
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)  

# Create rag pipeline tools

#tool 1 : retrieval chain from vectorstore for document retrieval.
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine", # "refine" chain type allows for iterative refinement of answers 
    verbose=True
)

#tool 2 : math chain for mathematical calculations. 
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Calculator",
        func=math_chain.run,
        description="Useful for answering math questions"
    ),
    Tool(
        name="ResearchDocs",
        func=retrieval_chain.run,
        description="Useful for answering questions about uploaded documents"
    )
]

template = """ 
You are a helpful research assistant. Answer the user's questions based on the provided context to you.
You can also perform calculations if needed.
Use the tools provided to you to assist in answering the user's query.
If you dont know the answer, just say "I don't know".

Here is the user query: {input}
Here is the context you may find useful: {context}
Here are the tools available to you: {tools}
Here is the list of tool names you can use: {tool_names}
Previous steps and thoughts (if any): {agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template=template)
                            
                                        

# Note: The context will be dynamically filled with the retrieved documents during the agent's execution.
# Create a structured chat agent with the defined tools and prompt
# Initialize Agent
mini_agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent = AgentExecutor.from_agent_and_tools(
    agent=mini_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_source_documents=True,
    memory_key="chat_history"  # Important for maintaining history
)


#create an app route to upload PDF files and store them in vectorstore - it is upload route 
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'): #only allow PDF files
        # Save the file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Add to vectorstore
        vectorstore.add_documents(chunks)
        vectorstore.persist() #store the changes to disk 
        
        return jsonify({"message": f"File {filename} has been uploaded and processed successfully."})
    
    return jsonify({"error": "Invalid file type"}), 400

# route to handle research queries
@app.route("/research", methods=["POST"])
def research():
    try:
        data = request.get_json()
        query = data.get("query")
        chat_history = data.get("chat_history", [])
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
         # Retrieve relevant documents from the vectorstore
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_context = [doc.page_content for doc in retrieved_docs]

        # Prepare inputs
        inputs = {
            "input": query,
            "context": retrieved_context,
            "tool_names": [tool.name for tool in tools],
            "tools": tools,
            "agent_scratchpad": chat_history 
        }

        result = agent.invoke(inputs)
        
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains.llm_math.base import LLMMathChain

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize LLM and Embeddings
llm = Ollama(model="tinyllama", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="tinyllama")

# Initialize Vector Store
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Create Tools
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Calculator",
        func=math_chain.run,
        description="Useful for answering math questions"
    ),
    Tool(
        name="ResearchDocs",
        func=retrieval_chain.run,
        description="Useful for answering questions about uploaded documents"
    )
]

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    
)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        
        # Add to vectorstore
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        
        return jsonify({"message": f"File {filename} uploaded and processed"})
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/research', methods=['POST'])
def research():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        result = agent.run(query)
        return jsonify({"response": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)