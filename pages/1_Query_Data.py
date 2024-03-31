import streamlit as st
from google.generativeai import ChatSession
from streamlit_extras.colored_header import colored_header
from streamlit_chat import message as st_message
import comment_analysis
from page_design import home_page, custom_css

st.markdown(custom_css, unsafe_allow_html=True)

st.title('Query Data')
# st.sidebar.header('Query Data')
st.markdown("""
To understand the data in great detail you can provide the query you have in the input box, remember this is a 
continuous conversation you are having with the healthbot 
""")

if "history" not in st.session_state:
    st.session_state.history = []


def generate_answer(user_message):
    model_response = comment_analysis.gemini_model_chat(user_message)
    return model_response


# Initialize session state variables
if 'user_responses' not in st.session_state:
    st.session_state['user_responses'] = tuple()

if 'bot_responses' not in st.session_state:
    st.session_state['bot_responses'] = tuple()

user_input = st.chat_input("You: ")

# Generate bot response and store user input
if user_input:
    print("User input:", user_input)  # Debug print
    response = generate_answer(user_input)
    print("Bot response:", response.last.text)  # Debug print
    if response:
        st.session_state['user_responses'] += (user_input,)
        st.session_state['bot_responses'] += (response.last.text,)

# Display all user inputs and corresponding bot responses
# for user_input, bot_response in zip(st.session_state['user_responses'], st.session_state['bot_responses']):
#     print("User input:", user_input)  # Debug print
#     print("Bot response:", bot_response)  # Debug print
#     st.write(f"User: {user_input}")
#     st.write(f"Bot: {bot_response}")

# Ensure bot_responses is initialized as an empty list
# if 'bot_responses' not in st.session_state:
#     st.session_state['bot_responses'] = []  # Initialize as an empty list
# elif isinstance(st.session_state['bot_responses'], str):  # Convert to list if currently a string
#     st.session_state['bot_responses'] = [st.session_state['bot_responses']]

for i, (user_response, bot_response) in enumerate(zip(st.session_state.user_responses, st.session_state.bot_responses)):
    # print((st.session_state['bot_responses']))
    if isinstance(bot_response, str):
        bot_response_text = bot_response
    elif hasattr(bot_response, 'last'):
        bot_response_text = bot_response.last.text
    elif isinstance(bot_response, ChatSession):
        bot_response_text = bot_response.last.text
        print("bot_response_text:", bot_response_text)
    else:
        bot_response_text = bot_response.last.text
    # Append to bot_responses instead of assigning
    st.session_state['bot_responses'] += (bot_response_text,)
    st.session_state['user_responses'] += (user_response,)
    st_message(user_response, is_user=True, key=f"user_response_{i}",
               avatar_style="initials", seed="Myra")
    st_message(bot_response_text, key=f"bot_response_{i}", avatar_style="initials", seed="AI")

# Capture user input
# user_input = st.text_area("You: ", "")
