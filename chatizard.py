import json

import streamlit as st
import numpy as np
import boto3

from PIL import Image
from services.data.oper import fetch_entity_by_id, summarize_schema, execute_sqlite_query
from services.data.store import IndexingService
from services.embeddings.embed import generate_embedding

from dotenv import load_dotenv

import os

load_dotenv()

icon = Image.open("data/eagle_transparent.png")

col1, col2, col3 = st.columns([1, 2, 20])

with col1:
    st.image(icon, width=75)

with col2:
    st.write("")

with col3:
    st.markdown("# ORN Chatbot")

service = IndexingService()
service.load_index("vector_index.bin")

bedrock_client = boto3.client('bedrock-runtime', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                              region_name=os.getenv("AWS_DEFAULT_REGION"),
                              aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))


def request_to_llm(prompt):
    params = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }

    # Make the API call to invoke the model
    response = bedrock_client.invoke_model(
        modelId=params['modelId'],
        contentType=params['contentType'],
        accept=params['accept'],
        body=json.dumps(params['body'])
    )

    # Parse and print the response
    response_body = json.loads(response['body'].read())

    return response_body['content'][0]['text']


if "messages" not in st.session_state:
    st.session_state.messages = []

context = ""

prompt = st.chat_input("Ask a question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

    for i in range(len(st.session_state.messages) - 2, max(-1, len(st.session_state.messages) - 8), -2):
        if i >= 0:
            context = f"User: {st.session_state.messages[i]['content']}\nAssistant: {st.session_state.messages[i + 1]['content']}\n\n" + context

    questions_list = ""
    with st.spinner('Please wait...'):

        prompt_embedding = generate_embedding(context + " " + prompt)
        k = 10

        prompt_for_sql_query_request = (
            f"User has asked the following: {prompt}, and we have the following database "
            f"schema:\n")
        prompt_for_sql_query_request += summarize_schema("data_store")
        prompt_for_sql_query_request += (
            "\nAlso we have fetched the following information that may or may not be relevant "
            "to the user's question:\n")

        labels, distances = service.query(np.array([prompt_embedding]), k=20)

        for i in range(len(labels[0])):
            label = labels[0][i]
            info = fetch_entity_by_id("entities.db", int(label))
            prompt_for_sql_query_request += (str(info) + "\n")

        prompt_for_sql_query_request += (
            "Your task is to utilize all the above information that have been given to you, "
            "to construct a SQLite query that fetches from the database the answer that the "
            "user requests. Give ONLY the SQL query and nothing else.")

        sqlite_query = request_to_llm(prompt_for_sql_query_request)
        print(sqlite_query)
        try:
            result = execute_sqlite_query("data_store", sqlite_query)
            print("Query executed successfully:", result)
        except Exception as e:
            print("Failed to execute query:", e)
            result = "Query failed to execute."

        prompt_for_final_answer = f"""
        Context (if available):
        {context}
        User's question: {prompt}
        The system run the query: {sqlite_query}
        Relevant information:
        {result}
        Instructions:
        If the provided information is relevant and sufficient, give a clear, concise, and direct answer to the user's question.
        If the information is irrelevant or insufficient, politely inform the user that you don't have enough information to provide an answer.
        Do not refer to the query or the information retrieval process in your response.
        For greetings or trivial questions that don't require additional information, respond using your existing knowledge and without refering to context, relevant information etc.
        """

        final_answer = request_to_llm(prompt_for_final_answer)

        assistant_message = final_answer

        prompt_for_relevant_questions = f"""
                # Task Description:
                # Generate a list of 3 questions. These questions should be directly answerable based on the provided context and should help the user explore potential inquiries related to the given information.

                # Provided Information:
                # Context: {context}
                # User's Initial Question: {prompt}
                # Additional SQL Query Context: {prompt_for_sql_query_request}

                # Instructions:
                # Utilize all the provided information to formulate three specific questions. These questions should be crafted in a way that they can be definitively answered using the given context and have a similar sqlite query like the previous to ensure query execution.
                # Output the questions as a numbered list and ensure no additional text is included in the response.
                # Do NOT refer to the query or to technical jargon like sql tables in your questions.
            
                # Example of expected output:
                # 1. Question A
                # 2. Question B
                # 3. Question C
                
                # Note: Replace the example questions with your generated questions based on the actual provided data.

                # End of instructions.
                """

        questions_list = request_to_llm(prompt_for_relevant_questions)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    with st.chat_message("assistant"):

        st.write(assistant_message)
        st.write("Questions you might want to explore:")
        st.write(questions_list)
