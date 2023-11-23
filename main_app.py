import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import Document

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

def get_text_input():
    text_input = st.text_area("テキストを入力してください:", key="input", height=200)
    return text_input

def summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.

============
    
{text}

============

ここから日本語で書いてね
必ず3段落以内の200文字以内で簡潔にまとめること:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT
        )

        # Create a Document with page_content set to content
        document = Document(
            page_content=docs[0]["content"],
            title=docs[0]["title"]
        )
        response = chain({"input_documents": [document]}, return_only_outputs=True)
        
    return response['output_text'], cb.total_cost

def main():
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        text_input = get_text_input()

    if text_input:
        # Create a dictionary from the text input
        document = [{"content": text_input, "title": "User Input"}]

        with st.spinner("ChatGPT is typing ..."):
            output_text, cost = summarize(llm, document)
        st.session_state.costs.append(cost)
    else:
        output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write("User Input")
    
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()
