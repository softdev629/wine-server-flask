import dotenv
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
from pandas import DataFrame
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI, PromptTemplate, FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory

load_dotenv()

# loader = TextLoader('./wineinfo.txt')
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# vector_db = FAISS.from_documents(docs, embeddings)
#
# extra_info: DataFrame = pd.read_excel("qa.xlsx")
# qa_list = extra_info.to_dict(orient="records")
# for qa_item in qa_list:
#     document = Document(page_content=f"Q: {qa_item['Prompt']}\nA: {qa_item['Completion']}")
#     vector_db.add_documents([document])
#
llm = ChatOpenAI(temperature=0)
# vector_db.save_local("./embeddings")
vector_db = FAISS.load_local("./embeddings", embeddings)

vector_template = """You are a chatbot having a conversation with a human

Try to avoid repeating one wine name several times and if possible try to mention product url wrapped in <a> tag everytime.
Given the following extracted parts of a long document and a question, create a final answer.

{context}

Human: {human_input}
Chatbot:"""

vector_prompt = PromptTemplate(
    input_variables=["human_input", "context"],
    template=vector_template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
vector_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=vector_prompt, memory=memory)

app = Flask(__name__)
CORS(app)

@app.route('/mad', methods=['POST'])
def mad():
    query = request.form["prompt"]
    docs = vector_db.similarity_search(query)
    vector_output = vector_chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)

    return {"answer": vector_output["output_text"]}

if __name__ == '__main__':
    app.run(debug=True)