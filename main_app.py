import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


def init_page():
    st.set_page_config(
        page_title="要約アプリ",
        page_icon="🧠"
    )
    st.header("要約アプリ 🧠")
    st.sidebar.title("モデル選択")

def init_messages():
    clear_button = st.sidebar.button("履歴削除", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="デモ段階であるため、ただchatgptのapiを使用してwebappを作成しただけになっているが今後要約アプリとして工夫していく")
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

def summarize(llm, text):
    prompt_template = """以下のテキストの簡潔な日本語要約を書いてください。

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
        response = chain({"input_documents": [text]}, return_only_outputs=True)

    return response['output_text'], cb.total_cost

def main():
    init_page()
    llm = select_model()
    init_messages()

    user_input = st.text_area("要約したいテキストを入力してください")

    if st.button("要約する"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT が要約しています..."):
            summary, cost = summarize(llm, user_input)
        st.session_state.messages.append(AIMessage(content=summary))
        st.session_state.costs.append(cost)

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## コスト")
    st.sidebar.markdown(f"**総コスト: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()
