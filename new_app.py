from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = OllamaLLM(model="tinyllama", base_url="http://localhost:11434")
template = """ 
You are a helpful research assistant. Answer the user's questions based on the provided context to you.
You can also perform calculations if needed.
If you dont know the answer, just say "I don't know".

Here is the query from the user: {query}
Here is the context provided to you: {context}
"""
#You can use the following tools to assist you: {tools}

prompt = ChatPromptTemplate.from_template(template=template,
                                           #tools=["Calculator", "ResearchDocs"],
                                           query="{input}",
                                           context=MessagesPlaceholder("context"))

chain = prompt | model 

while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    # Invoke the chain with the user input and context
    result = chain.invoke({
        "query": user_input,
        "context": []
    })
    
    print(result)
    
    

