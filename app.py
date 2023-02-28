from flask import Flask, request
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

# with open("./wineinfo.txt") as f:
#     state_of_the_union = f.read()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()
# docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
# docsearch.save_local("./embeddings")
docsearch = FAISS.load_local("./embeddings", embeddings)
# print("complete")

template = """You are a chatbot having a conversation with a human

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)

app = Flask(__name__)
CORS(app)

@app.route('/mad', methods=['POST'])
def mad():
    query = request.form["prompt"]
    docs = docsearch.similarity_search(query)
    output = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    return {"answer": output["output_text"]}
    # return {"answer": "answer"}
