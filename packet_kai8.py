import os
import json
import requests
import subprocess
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

import logging
logger = logging.getLogger(__name__)

class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    pass

class AIMessage(Message):
    pass

@st.cache_resource
def load_model():
    try:
        with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
            embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
        return embedding_model
    except Exception as e:
        st.warning(f"CUDA not available or an error occurred: {e}. Falling back to CPU.")
        try:
            with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient (CPU fallback)"):
                embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"})
            return embedding_model
        except Exception as e:
            st.error(f"An error occurred while loading the model on CPU: {e}")
            return None

class ChatWithPCAP:
    def __init__(self, json_path):
        self.json_path = json_path
        self.embedding_model = load_model()
        self.vectordb = None
        self.load_json()
        self.store_in_chroma(self.pages)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.llm_chains = self.initialize_llm_chains()

        self.conversation_history = []

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self, pages):
        text_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile"
        )
        chunks = text_splitter.split_documents(pages)
        return chunks

    def store_in_chroma(self, docs):
        self.vectordb = Chroma.from_documents(docs, embedding=self.embedding_model)

    def initialize_llm_chains(self):
        llm_chains = {}
        models = ["gemma", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat"]

        def create_qa_chain(model):
            llm = Ollama(model=model, base_url=f"http://localhost:80/api/{model}/generate")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectordb.as_retriever(search_kwargs={"k": 10}),
                memory=self.memory
            )
            return qa_chain

        for model in models:
            llm_chains[model] = create_qa_chain(model)
        return llm_chains

    def send_request(self, model, prompt):
        url = f"http://localhost:80/backend/{model}/generate"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

    def chat(self, question):
        all_results = []
        response_placeholders = {}

        # Create a placeholder for each model's response
        for model in self.llm_chains.keys():
            response_placeholders[model] = st.empty()

        for model, qa_chain in self.llm_chains.items():
            try:
                response = qa_chain.invoke(question)
                if response:
                    answer_text = response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
                    response_placeholders[model].write(f"Model: {model}\nResponse: {answer_text}")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": answer_text
                        }
                    )
                else:
                    response_placeholders[model].write(f"Model: {model}\nNo response received.")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": "No response received."
                        }
                    )
            except Exception as e:
                response_placeholders[model].write(f"Model: {model}\nError: {e}")
                all_results.append(
                    {
                        "model": model,
                        "query": question,
                        "answer": f"Error: {e}"
                    }
                )

        consensus_prompt = (
            f"I am asking you to try and come to consensus with other LLMs on the answer to this question: "
            f"{question} Here are the answers from each LLM so far: {all_results}"
        )
        consensus_responses = []
        for model in self.llm_chains.keys():
            consensus_response = self.send_request(model, consensus_prompt)
            st.write(f"Consensus response from {model}: {consensus_response}")
            consensus_responses.append(consensus_response)

        final_consensus_prompt = (
            f"I am asking you to try and come to consensus with other LLMs on the answer to this question: "
            f"{question} Here are the consensus answers from each LLM so far: {consensus_responses}"
        )
        final_consensus_responses = []
        for model in self.llm_chains.keys():
            final_consensus_response = self.send_request(model, final_consensus_prompt)
            st.write(f"Final consensus response from {model}: {final_consensus_response}")
            final_consensus_responses.append(final_consensus_response)

        self.conversation_history.append(HumanMessage(content=question))
        for result in all_results:
            self.conversation_history.append(AIMessage(content=f"Model: {result['model']} - {result['answer']}"))

def model_selection():
    st.title("Select Models")
    all_models = ["gemma", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat"]

    def select_all():
        for model in all_models:
            st.session_state.selected_models[model] = True

    def deselect_all():
        for model in all_models:
            st.session_state.selected_models[model] = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Select All Models'):
            select_all()
    with col2:
        if st.button('Deselect All Models'):
            deselect_all()

    col1, col2, col3 = st.columns(3)
    for idx, model in enumerate(all_models):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.session_state.selected_models[model] = st.checkbox(model, value=st.session_state.selected_models[model], key=model)

    if st.button('Next'):
        st.session_state.page = 2
        st.rerun()

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    st.title('Packet KAI8 - Chat with Packet Captures using Multi-AI Consensus')
    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    if uploaded_file:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        pcap_to_json(pcap_path, json_path)
        st.session_state['json_path'] = json_path
        st.success("PCAP file uploaded and converted to JSON.")
        # Fetch and display the models in a select box
        if st.button("Proceed to Chat"):
            st.session_state.page = 3
            st.rerun()     

# Streamlit UI for chat interface
def chat_interface():
    st.title('Packet KAI8 - Chat with Packet Captures using Multi-AI Consensus')
    json_path = st.session_state.get('json_path')
    if not json_path or not os.path.exists(json_path):
        st.error("PCAP file missing or not converted. Please go back and upload a PCAP file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(json_path=json_path)

    user_input = st.text_input("Ask a question about the PCAP data:")
    if user_input and st.button("Send"):
        with st.spinner('Thinking...'):
            response = st.session_state['chat_instance'].chat(user_input)
            st.markdown("**Synthesized Answer:**")
            if isinstance(response, dict) and 'answer' in response:
                st.markdown(response['answer'])
            else:
                st.markdown("No specific answer found.")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state["page"] = 1
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = {model: False for model in ["gemma", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat"]}

    if st.session_state.page == 1:
        model_selection()
    elif st.session_state.page == 2:
        upload_and_convert_pcap()
    elif st.session_state.page == 3:
        chat_interface()
