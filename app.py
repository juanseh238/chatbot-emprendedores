from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, OpenAI
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
import dotenv
import os

import fitz
from PIL import Image

from chatUtils import process_file, generate_response

# load environment variables
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())

# Global variables
COUNT, N = 0, 0
chat_history = []
chain = ''

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Function to add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

# Function to render a specific page of a PDF file as an image


def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image


def process_file(file):

    # Load PDF file using PyPDFLoader
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    # load embeddings model
    embeddings = OpenAIEmbeddings()

    # load LLM model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )
    
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True)
    return chain


def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1

    result = chain(
        {"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''


# Gradio application setup
with gr.Blocks() as demo:
    # Create a Gradio block

    with gr.Column():
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF',
                                tool='select').style(height=680)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter"
            ).style(container=False)

        with gr.Column(scale=0.15):
            submit_btn = gr.Button('Submit')

        with gr.Column(scale=0.15):
            btn = gr.UploadButton(
                "📁 Upload a PDF", file_types=[".pdf"]).style()

    # Set up event handlers
    # Event handler for uploading a PDF
    btn.upload(fn=render_file, inputs=[btn], outputs=[show_img])

    # Event handler for submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )
demo.queue()
if __name__ == "__main__":
    demo.launch()
