from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
import os 
from dotenv import load_dotenv
import streamlit as st
import logging
from datetime import datetime
import time
from typing import Iterator
load_dotenv()


# Set up logging
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists
log_file = os.path.join(log_dir, f"app_log_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Log app start
logging.info("Streamlit app started.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MODEL_NAME = "llama-3.3-70b-Specdec"
MODEL_NAME = "gemma2-9b-It"
logging.info("GROQ API key loaded.")

# Streamlit app
st.set_page_config(page_title="MlHub", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Financial Analyst Agent")

st.markdown("""
This app allows you to interact with multiple LLMs simultaneously. 
Choose from the available models, ask a question, and compare the answers!
""")


finance_agent = Agent(
    name="Finance Agent",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    model=Groq(id=MODEL_NAME, api_key=GROQ_API_KEY, structured_outputs=True),
    role="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["If internet search is needed, go to the web search agent.",
                  "Format your response using markdown and use tables to display data where possible."],
    markdown=True
)

logging.info("Finance Agent created.")

web_search = Agent(
    name="Web Search Agent",
    tools=[GoogleSearch()],
    role="Search the web for the information",
    instructions=[
        "Always include sources"
    ],
    model=Groq(id=MODEL_NAME, api_key=GROQ_API_KEY, structured_outputs=True),
    show_tool_calls=True,
    markdown=True
)

logging.info("Web Search Agent created.")

# Print the response in the terminal
# team_agent.print_response("Share the NVDA stock price and analyst recommendations", stream=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        

# React to user input
if prompt := st.chat_input("What is up?"):
    # logging.info(f"User input received: {prompt}")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    logging.info(f"User message added to chat history: {prompt}")
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    
    try:
        with st.chat_message("assistant"):
            status = st.empty()
            status.status("Sending request...", state="running")
            time.sleep(1)  # Simulate waiting for the model
            
            status.status("Receiving response...")
            
            conversation_history = ""
            for message in st.session_state.messages:
                conversation_history += f"{message['role']}: {message['content']} \n"
            
            # Stream the response
            response_stream: Iterator[RunResponse] = finance_agent.run(conversation_history, stream=True)
            
            status.status(label="Response complete!", state="complete")
            logging.info("Model response received.")
            
            response_text = ""
            response_placeholder = status.empty()  # Placeholder for the streaming response
            
            for response in response_stream:
                response_text += response.content
                response_placeholder.markdown(response_text)
                
            response_stream = None
                
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        conversation_history += f"'assistant': {response_text} \n"
        logging.info(f"Model response received: {response_text}")
    except Exception as e:
        logging.error(f"Error during model interaction: {e}")
        st.error("An error occurred while processing your request. Please try again.")

# Log app exit
logging.info("Streamlit app execution ended.")