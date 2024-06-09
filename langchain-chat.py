from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()

hugging_face_key = os.getenv('HUGGING_FACE_TOKEN')
if hugging_face_key is None:
    hugging_face_key = st.secrets["HUGGING_FACE_TOKEN"]

# Initialize the Hugging Face Hub model
llm = HuggingFaceHub(
    repo_id="bigscience/bloom",
    model_kwargs={"temperature": 0.2,"max_length":1024, "do_sample": True, "top_p": 0.9 },
    huggingfacehub_api_token=hugging_face_key
)

#make a chat prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("user","Question:{question}")
    ]
)

# streamlit framework
st.title('Langchain Demo With Bloom')
input_text=st.text_area("Type below to search. Note: A good prompt: Do NOT talk to Bloom as an entity, it's not a chatbot but a webpage/blog/article completion model",
                        key="input_box")

# llama2 
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

# write output to screen
if input_text:
    st.write(chain.invoke({"question":input_text}))



