#Import dependencies
import os
from decouple import config
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

#Set env variables
os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

#App framework
st.title('Query my LLM')
prompt = st.text_input('Put prompt here')

#Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='Write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template='Write me a youtube video script based on this title TITLE: {title}'
)

#Memory
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

#Llms
llm = OpenAI(temperature=0.7)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

#Show stuff on the screen if there's a prompt
if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)