import os
import uuid
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
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from chromadb.config import Settings
from chromadb import Client

import logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip('"')

class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    pass

class AIMessage(Message):
    pass

@st.cache_resource
def load_model():
    api_key = OPENAI_API_KEY
    if not api_key:
        st.error("OpenAI API Key is missing. Please set it in the .env file.")
        return None
    try:
        with st.spinner("Loading OpenAI Embeddings..."):
            embedding_model = OpenAIEmbeddings(api_key=api_key)
        return embedding_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to generate priming text based on pcap data
def returnSystemText(pcap_data: str) -> str:
    PACKET_WHISPERER = f"""
        You are an expert assistant specialized in analyzing packet captures (PCAPs) for troubleshooting and technical analysis. Use the data in the provided to answer user questions accurately.

        Your goal is to provide a clear, concise, and accurate analysis of the packet capture data, leveraging the packet details from the uploaded .pcap JSON.
    """
    return PACKET_WHISPERER

class ChatWithPCAP:
    def __init__(self, json_path):
        self.json_path = json_path
        self.priming_text = st.session_state.get('priming_text', "")
        self.embedding_model = load_model()
        self.pages = None
        self.docs = None
        self.vectordb = None
        self.memory = None
        self.llm_chains = None
        self.conversation_history = []

        # Load and process the JSON file
        self.load_json()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.initialize_llm_chains()

    def load_json(self):
        """Load and split JSON data into pages."""
        with st.spinner("Loading JSON data..."):
            # Use jq schema to exclude specific fields
            self.loader = JSONLoader(
                file_path=self.json_path,
                jq_schema="""
                    .[] 
                    | ._source.layers 
                    | del(.data)
                """,
                text_content=False
            )
            self.pages = self.loader.load_and_split()

        if not self.pages:
            st.error("No data loaded from JSON file. Please check the input file.")
            raise ValueError("No data loaded from JSON file.")

    def split_into_chunks(self):
        """Split loaded pages into smaller, meaningful chunks."""
        with st.spinner("Splitting into chunks..."):
            text_splitter = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type="percentile"
            )
            self.docs = text_splitter.split_documents(self.pages)

        if not self.docs:
            st.error("No documents were generated from the PCAP data. Please check the input file.")
            raise ValueError("Document splitting resulted in an empty list.")

    def store_in_chroma(self):
        """Store chunks in Chroma for vector search."""
        with st.spinner("Storing in Chroma..."):
            session_id = st.session_state.get('session_id', str(uuid.uuid4()))
            st.session_state['session_id'] = session_id
            persist_directory = f"chroma_db_{session_id}"
            self.vectordb = Chroma.from_documents(
                self.docs, 
                embedding=self.embedding_model, 
                persist_directory=persist_directory
            )

    def setup_conversation_memory(self):
        """Initialize conversation memory for chat history."""
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def initialize_llm_chains(self):
        llm_chains = {}

        # Map full model names to their URL aliases
        model_aliases = {
            "gemma2": "gemma2",
            "llama3.1": "llama3_1",
            "llama3.2": "llama3_2", 
            "mistral": "mistral",
            "command-r7b": "commandr",
            "phi4": "phi4",
            "deepseek-r1": "deepseek"
        }

        def create_qa_chain(model, alias):
            llm = Ollama(model=model, base_url=f"http://localhost:80/api/{alias}/generate")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectordb.as_retriever(search_kwargs={"k": 10}),
                memory=self.memory
            )
            return qa_chain

        for model, alias in model_aliases.items():
            llm_chains[model] = create_qa_chain(model, alias)

        # FIX: Assign llm_chains to self.llm_chains
        self.llm_chains = llm_chains
    
    def send_request(self, model, prompt):
        # Same alias mapping for the request
        model_aliases = {
            "gemma2": "gemma2",
            "llama3.1": "llama3_1",
            "llama3.2": "llama3_2",  # Add alias for llama3.2
            "mistral": "mistral",
            "command-r7b": "commandr",
            "phi4": "phi4",
            "deepseek-r1": "deepseek"
        }
    
        # Get the alias for the model
        alias = model_aliases.get(model, model)
    
        url = f"http://localhost:80/backend/{alias}/generate"
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

        # Retrieve relevant documents
        retrieved_docs = self.vectordb.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(question)
        retrieved_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Create system text
        system_text = st.session_state.get('priming_text', returnSystemText(retrieved_text))

        # Prepend system and retrieved context to the question
        full_prompt = f"{system_text}\n\nContext:\n{retrieved_text}\n\nQuestion:\n{question}"

        # Query each model
        for model, qa_chain in self.llm_chains.items():
            try:
                response = qa_chain.invoke(full_prompt)
                if response:
                    answer_text = response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
                    st.write(f"**{model} response:**\n{answer_text}")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": answer_text
                        }
                    )
                else:
                    st.write(f"{model}: No response received.")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": "No response received."
                        }
                    )
            except Exception as e:
                st.write(f"{model}: Error: {e}")
                all_results.append(
                    {
                        "model": model,
                        "query": question,
                        "answer": f"Error: {e}"
                    }
                )

        # Initial consensus prompt
        consensus_prompt = (
            f"Hello, esteemed models. I am seeking your collective expertise to reach a consensus on the following question: "
            f"{question}. Below are the individual responses from each model: {all_results}. "
            "Please review these responses carefully and provide a reasoned summary that attempts to align and synthesize the varied perspectives. "
            "Consider the strengths and weaknesses of each response, and aim to identify common themes or points of agreement."
        )
        consensus_responses = []
        for model in self.llm_chains.keys():
            consensus_response = self.send_request(model, consensus_prompt)
            st.write(f"Consensus response from {model}: {consensus_response}")
            st.markdown("""---""")
            consensus_responses.append(consensus_response)

        # Final consensus prompt
        final_consensus_prompt = (
            f"Thank you for your thoughtful responses. Now, I ask you to further refine and come to a final consensus on the answer to this question: "
            f"{question}. Here are the preliminary consensus answers from each model so far: {consensus_responses}. "
            "Please critically evaluate these summaries, identify the most compelling arguments, and work towards a unified, well-supported final answer. "
            "Your final response should integrate the best elements of each perspective and resolve any remaining discrepancies."
            "Keep the original question in mind and do your best to come up with the best, most agreed upon, answer to that original question."
        )

        final_consensus_responses = []
        for model in self.llm_chains.keys():
            final_consensus_response = self.send_request(model, final_consensus_prompt)
            st.write(f"Final consensus response from {model}: {final_consensus_response}")
            st.markdown("""---""")
            final_consensus_responses.append(final_consensus_response)

        self.conversation_history.append(HumanMessage(content=question))
        for result in all_results:
            self.conversation_history.append(AIMessage(content=f"Model: {result['model']} - {result['answer']}"))

def model_selection():
    st.title("Select Models")
    all_models = all_models = ["llama3.1", "llama3.2", "command-r", "mistral", "gemma2", "phi4", "deepseek-r1"]

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

def pcap_to_json(pcap_path, json_path):
    # Convert pcap to JSON
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

    # Remove udp.payload and tcp.payload from the JSON
    try:
        with open(json_path, "r") as file:
            data = json.load(file)  # Load the JSON data
        
        # Process each packet and remove unwanted fields
        for packet in data:
            layers = packet.get("_source", {}).get("layers", {})
            if "udp" in layers and "udp.payload" in layers["udp"]:
                del layers["udp"]["udp.payload"]
            if "tcp" in layers and "tcp.payload" in layers["tcp"]:
                del layers["tcp"]["tcp.payload"]

        # Save the cleaned JSON back to the file
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

    except json.JSONDecodeError as e:
        st.error(f"Error processing JSON file: {e}")
        raise ValueError("Failed to decode JSON file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        raise

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
        st.session_state.selected_models = {model: False for model in ["llama3.1", "llama3.2", "command-r", "mistral", "gemma2", "phi4", "deepseek-r1"]}

    if st.session_state.page == 1:
        model_selection()
    elif st.session_state.page == 2:
        upload_and_convert_pcap()
    elif st.session_state.page == 3:
        chat_interface()
