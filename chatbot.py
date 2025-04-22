import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
import asyncio
import nest_asyncio

# --- Enable asyncio inside Streamlit ---
nest_asyncio.apply()

# --- App Config ---
csv_file = 'college_data.csv'
model_name = "llama3-70b-8192"
num_processing_llms = 6

# --- API Key Setup ---
primary_api_keys = [st.secrets.get(f"GROQ_API_KEY{i+1}") for i in range(7)]
backup_api_keys = st.secrets.get("GROQ_BACKUP_API_KEYS", [])

if not all(primary_api_keys) or len(primary_api_keys) != 7:
    st.error("Please set all 7 primary GROQ API keys in Streamlit secrets.")
    st.stop()

if not backup_api_keys:
    st.warning("No backup GROQ API keys were provided. Failover may not work.")

# --- Load Data Once from CSV ---
def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient="records")

if "college_data" not in st.session_state:
    st.session_state.college_data = load_and_preprocess_data(csv_file)
    chunk_size = len(st.session_state.college_data) // num_processing_llms
    st.session_state.data_chunks = [
        st.session_state.college_data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processing_llms - 1)
    ]
    st.session_state.data_chunks.append(st.session_state.college_data[(num_processing_llms - 1) * chunk_size:])

# --- Initialize LLMs Once ---
if "processing_llms" not in st.session_state:
    st.session_state.processing_llms = [
        ChatGroq(api_key=primary_api_keys[i], model_name=model_name)
        for i in range(num_processing_llms)
    ]
    st.session_state.final_llm_key = primary_api_keys[6]

# --- Prompt Templates ---
individual_prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Use the following college information internally to answer the user's question accurately, but do not mention the data or your internal processing in your response.\n\n"
    "{college_details}\n\n"
    "User question:\n{user_query}\n\n"
    "Answer clearly and directly, as if you are speaking naturally to the user."
)

final_prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Here are multiple assistant responses generated internally (do not mention this to the user):\n\n"
    "{responses}\n\n"
    "Now, combine and summarize these into a single, clear, and natural-sounding response to the original user question below.\n"
    "Do not reference the data, processing, or models used. Just give a direct, friendly answer.\n\n"
    "User question:\n{original_user_query}"
)

# --- Streamlit Chat UI ---
st.title("🎓 College Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Async Chunk Processor ---
async def get_chunk_info(data_chunk, llm_index, retry_count=0, backup_index=0):
    for i, item in enumerate(data_chunk):
        item["record_id"] = f"College-{i+1}"
    college_details_str = "\n".join(
        f"{item['record_id']} - " + ", ".join([f"{k}: {v}" for k, v in item.items() if k != 'record_id']) for item in data_chunk
    )
    prompt = individual_prompt_template.format(college_details=college_details_str, user_query=user_query)
    try:
        response = await st.session_state.processing_llms[llm_index].ainvoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"LLM {llm_index+1} failed: {e}")
        if backup_index < len(backup_api_keys):
            backup_key = backup_api_keys[backup_index]
            st.session_state.processing_llms[llm_index] = ChatGroq(api_key=backup_key, model_name=model_name)
            return await get_chunk_info(data_chunk, llm_index, retry_count + 1, backup_index + 1)
        else:
            return ""

# --- Async Final Merger ---
async def get_final_response(responses, original_query, retry_count=0, backup_index=0):
    prompt = final_prompt_template.format(responses="\n\n".join(responses), original_user_query=original_query)
    try:
        final_llm = ChatGroq(api_key=st.session_state.final_llm_key, model_name=model_name)
        final_response = await final_llm.ainvoke(prompt)
        return final_response.content
    except Exception as e:
        st.error(f"Final LLM failed: {e}")
        if backup_index < len(backup_api_keys):
            backup_key = backup_api_keys[backup_index]
            st.session_state.final_llm_key = backup_key
            return await get_final_response(responses, original_query, retry_count + 1, backup_index + 1)
        else:
            return "Sorry, I couldn't generate a response right now."

# --- User Input Form ---
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("You:", placeholder="Ask something about a college...")
    submit = st.form_submit_button("Send")

# --- Async Response Pipeline ---
if submit and user_query:
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    with st.spinner("🤖 Typing..."):
        async def main():
            tasks = [get_chunk_info(chunk, i) for i, chunk in enumerate(st.session_state.data_chunks)]
            responses = await asyncio.gather(*tasks)
            final_response = await get_final_response(responses, user_query)
            st.session_state.chat_history.append({"role": "assistant", "text": final_response})
        asyncio.get_event_loop().run_until_complete(main())

# --- Display Chat ---
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**🧑 You:** {chat['text']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['text']}")
