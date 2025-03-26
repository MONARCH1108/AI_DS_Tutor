import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os


os.environ["GOOGLE_API_KEY"] = "Your_API_Key"

class DataScienceTutor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Science Tutor. 
            Your primary goal is to help users understand data science concepts, 
            resolve their doubts, and provide clear, concise explanations. 
            You can cover topics including:
            - Machine Learning Algorithms
            - Statistical Analysis
            - Data Preprocessing
            - Python Data Science Libraries
            - Data Visualization
            - Model Evaluation
            - Advanced Data Science Techniques

            Always provide:
            1. Clear explanations
            2. Code examples when relevant
            3. Step-by-step guidance
            4. Practical insights

            If a question is outside data science, politely redirect the conversation."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm, 
            prompt=self.prompt, 
            memory=self.memory,
            verbose=True
        )
    
    def get_response(self, user_input):
        """Generate response based on user input"""
        response = self.chain.run(user_input=user_input)
        return response

def main():
    st.title("🧠 Data Science Tutor AI")
    st.markdown("Your personal conversational guide for data science learning!")
    

    if 'tutor' not in st.session_state:
        st.session_state.tutor = DataScienceTutor()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What data science concept can I help you with?"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.tutor.get_response(prompt)
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

if __name__ == "__main__":
    main()
