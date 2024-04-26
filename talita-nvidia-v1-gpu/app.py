import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.chains.question_answering import load_qa_chain
from PIL import Image

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    
    embeddings = NVIDIAEmbeddings(model="playground_nvolveqa_40k",
                                        model_type="passage")
    
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except (ValueError) as e:
        if '401' in str(e):
            st.error('The provided API key is not valid. Please check or regenerate your API key.')
        else:
            raise   # re-raise the exception if it's not a 401 error
    
def get_conversation_chain(vectorstore, model_choice, temperature, top_p, seed):

    llm = ChatNVIDIA(model=model_choice, max_tokens=1024, temperature=temperature, top_p=top_p, seed=seed, streaming=False)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question, nvapi_key2):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Add the model information to the message.content
            model_info = f" (Model used: nvidia-{st.session_state.model_choice})"
            st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            st.markdown(model_info)

def main():
    st.set_page_config(page_title="TALITA - NVIDIA",
                       page_icon=":rocket:", layout="wide", menu_items={'Get Help':'https://talita.lintasarta.net', 'About':'© 2024 - Talita® | Corporate IT | Lintasarta'})
    
    if st.sidebar.text_input:
        nvapi_key2 = st.sidebar.text_input("Enter your NVIDIA API Key:", type="password")
        
    st.sidebar.write("")
    st.sidebar.subheader("📦 Foundation Models")
    models = ["ai-neva-22b", "mixtral_8x7b", "ai-gemma-7b", "ai-llama2-70b", "ai-llama3-8b", "ai-phi-3-mini", "ai-codellama-70b", "ai-arctic", "ai-sdxl-turbo"]
    model_choice = st.sidebar.selectbox("Choose the model", models, help="Click 'Process' after selecting the model." )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Mengontrol tingkat keacakan jawaban. Nilai rendah: jawaban lebih konsisten. Nilai tinggi: jawaban lebih beragam.")
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Mengontrol keragaman kata-kata yang dipilih. Nilai rendah: pilihan kata lebih terbatas tapi relevan. Nilai tinggi: lebih banyak kata potensial dan keragaman.")
    seed = st.sidebar.radio("Seed", [0, 1, 2, 3, 4], horizontal=True, help="Sebagai identitas output acak, hal ini membuat hasil yang didapat bisa diulang dengan hasil yang sama.")

    st.session_state.model_choice = model_choice  # store the chosen model
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    st.session_state.seed = seed

    st.write(css, unsafe_allow_html=True)
    
    if "process_done" not in st.session_state:  # add this to store the state of process
        st.session_state.process_done = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    #opening the image
    image = Image.open('images/talita-nvidia.png')
    #displaying the image on streamlit app
    st.image(image, caption='', width=300)
    st.write("Experience TALITA - Driven by Advanced NVIDIA Technology")
    st.markdown("<hr>", unsafe_allow_html=True)     

    with st.sidebar:
        st.write("")
        st.subheader("📂 Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
        
        if st.button("Process"):
            
            if not nvapi_key2:
                st.warning("API Key should not be empty!")
            elif not pdf_docs:
                st.warning("Please upload a file!")
            else:
                if not nvapi_key2.startswith("nvapi-"):
                    st.error(f"The provided API key is not valid.")
                    return
                else:
                    os.environ["NVIDIA_API_KEY"] = nvapi_key2
            
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore is None:
                        st.write("")
                    else:
                        st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.model_choice, st.session_state.temperature, st.session_state.top_p, st.session_state.seed)

                    st.session_state.process_done = True  # Add this line after the "Process" button is clicked
        
        # Bagian untuk menampilkan hak cipta di bagian bawah halaman
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("© 2024 - Talita® | Lintasarta", unsafe_allow_html=True)

    if st.session_state.process_done:  # add this condition to show st.text_input only after process is done
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question, nvapi_key2)

if __name__ == '__main__':
    main()