import json

import streamlit as st
import numpy as np
import boto3

from services.data.oper import fetch_entity_by_id, summarize_schema, execute_sqlite_query
from services.data.store import IndexingService
from services.embeddings.embed import generate_embedding

st.title("ORN Chatbot")

service = IndexingService()
service.load_index("vector_index.bin")

bedrock_client = boto3.client('bedrock-runtime', aws_access_key_id="AKIAT5V6ZRRFALRQPQAK", region_name="us-west-2",
                              aws_secret_access_key="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")


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

prompt = st.chat_input("Ask a question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Please wait...'):

        prompt_embedding = generate_embedding(prompt)
        k = 10

        prompt_for_sql_query_request = (
            f"User has asked the following: {prompt}, and we have the following database "
            f"schema:\n")
        prompt_for_sql_query_request += summarize_schema("data_store")
        prompt_for_sql_query_request += (
            "\nAlso we have fetched the following information that may or may not be relevant "
            "to the user's question:\n")

        labels, distances = service.query(np.array([prompt_embedding]), k=10)
        print("Query results:", labels, distances)

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
        User's question: {prompt}
        The system run the query: {sqlite_query}
        Relevant information:
        {result}
        Instructions:
        If the information above is relevant to the user's question, provide a clear, concise, and direct answer.
        If the information is not relevant or insufficient to answer the question, simply inform the user that you do not have enough information to provide an answer.
        Do not mention the query or the process of fetching the information in your response.
        """
        print(prompt_for_final_answer)
        final_answer = request_to_llm(prompt_for_final_answer)

        assistant_message = final_answer

    with st.chat_message("assistant"):

        st.write(assistant_message)
