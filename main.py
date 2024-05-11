import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = "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"

    # Display the Groq logo
    #spacer, col = st.columns([5, 1])  
    #with col:  
    #    st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Counsellor-Bot")
    st.markdown("""
 Hello! I'm your friendly counselling chatbot.
 I can help answer your questions about any of the following topics:
- **Mental Health**
- **Business Health**
- **Investment Health**

I'm also super fast! Let's start our conversation!
""")


    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'llama3-70b-8192']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 100, value = 50)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

  
    user_question = st.chat_input("Ask a question!")


    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input':message['human']},
                {'output':message['AI']}
                )


    

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )


    # If the user has asked a question,
    if user_question:

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=                 
'''You have to help the user by providing relevant advice regarding any topic that they ask about. Primarily, the area
you need to focus on is the mental health domain, business domain and investment domain. 

For mental health, I want you to provide a reassuring and optimistic tone to the user so that they
are not scared by their medical issues and feel comfortable. I want you to display empathy, ask relevant questions and collect relevant information from the user.
You should provide good advice based on what the user is asking and also consider providing relevant contact information regarding experts for the user if neccessary.

For medical, we have the following experts:
Name : ai2
Email: ai2@gmail.com
Expertise Area: Psychologist

Name : ai3
Email: ai3@gmail.com
Expertise Area: Psychitarist

For business, I want you to provide good advice and collect as much information as possible.

Business Experts:
Name : ai4
Email: ai4@gmail.com
Expertise Area: Sports 

Name : ai5
Email: ai5@gmail.com
Expertise Area: Real Estate

For investment, I want you to provide good advice and collect as much information as possible.

Investment Experts:
Name : ai6
Email: ai6@gmail.com
Expertise Area: Cryptocurrency

Name : ai7
Email: ai7@gmail.com
Expertise Area: Stocks


'''
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
            verbose=True,   # Enables verbose output, which can be useful for debugging.
            memory=memory,  # The conversational memory object that stores and manages the conversation history.
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human':user_question,'AI':response}
        st.session_state.chat_history.append(message)

        st.write("Chatbot:", response)


        for message in st.session_state.chat_history:
            st.text(f"Human: {message['human']}")
            st.text(f"Chatbot: {message['AI']}")

if __name__ == "__main__":
    main()