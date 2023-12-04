import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import Document
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def init_page():
    st.set_page_config(
        page_title="要約アプリ",
        page_icon="🧠"
    )
    st.header("要約アプリ 🧠")
    
    # サイドバーのタイトルを表示
    st.sidebar.title("モデル選択")
    st.session_state.costs = []

def init_messages():
    # サイドバーにボタンを設置
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

    return ChatOpenAI(temperature=0, model_name=model_name)

def get_text_input():
    text_input = st.text_area("テキストを入力してください:", key="input", height=200)
    return text_input

def summarize(llm, docs):
    prompt_template = prompt_template = """
#命令書
見出し:
女性美　表現の変遷　新居浜市美術館「描かれた女たち」　明治から現代　日本の洋画８１点　来月２６日まで

本文:
　西洋美術との出合いは、日本人画家が描く人体像にどんな変化をもたらしたのか―。女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並ぶ。６月２６日まで。
　明治期に西洋美術に接した日本画壇は、対象の科学的な捉え方や陰影法などの技法だけでなく、絵画とは何かといった概念も吸収した。特別展では東郷青児、竹久夢二、百武兼行、絹谷幸二、岸田劉生、藤島武二らが描いた「女性美」から、表現の多様性を浮かび上がらせている。展示作品の一部を紹介する。（所蔵は全て日動美術財団）

- 寺島が作成した要約文章
女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並び、展示期間は6月26日までである。

以上のように本文を寺島のように要約して下さい。
#入力する文章
{text:}
#出力形式
要約した文章:

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
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        text_input = get_text_input()

    if text_input:
        
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
