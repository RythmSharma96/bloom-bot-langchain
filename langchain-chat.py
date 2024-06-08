from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

#make a chat prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

# streamlit framework
st.title('Langchain Demo With LLAMA2')
input_text=st.text_input("Type in here to search...", key="input_box")

# ollama llama2 
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

# write output to screen
if input_text:
    st.write(chain.invoke({"question":input_text}))



