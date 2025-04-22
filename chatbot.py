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
clarification_model_name = "llama3-70b-8192"

# --- API Key Setup ---
primary_api_keys = [st.secrets.get(f"GROQ_API_KEY{i+1}") for i in range(7)]
backup_api_keys = st.secrets.get("GROQ_BACKUP_API_KEYS", [])

if not all(primary_api_keys) or len(primary_api_keys) != 7:
    st.error("Please set all 7 primary GROQ API keys in Streamlit secrets.")
    st.stop()

# --- Load CSV Once ---
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
    st.session_state.clarification_llm = ChatGroq(api_key=primary_api_keys[0], model_name=clarification_model_name)

# --- Prompt Templates ---
clarification_prompt_template = PromptTemplate.from_template(
    "The user has asked a question about college information. "
    "Based on the following example of college data structure, please try to understand the user's query and rephrase it using the full names of any fields or terms if they used abbreviations or short forms. "
    "If the user's query is already clear and doesn't seem to contain abbreviations, you can return the original query.\n\n"
    "Example College Data Structure (first entry):\n{example_data}\n\n"
    "User Query:\n{user_query}\n\n"
    "Rephrased Query:"
)

individual_prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. You will be provided with college data in the format below.\n"
    "ONLY use this data to answer the user's question, and ONLY if the information pertains to the exact college name mentioned in the user's query — DO NOT use any outside knowledge.\n"
    "If you cannot find an *exact* match for the college name in the user's question within this data, or if you cannot find an answer based on an exact match, say 'Sorry, I don't have information about that specific college based on the provided data.'\n\n"
    "College Data:\n{college_details}\n\n"
    "User Question:\n{user_query}\n\n"
    "Be specific, accurate, and never guess. List only what is directly found in the data for the *exact* college mentioned."
)

final_prompt_template = PromptTemplate.from_template(
    "You are a highly intelligent and discerning assistant. You have received several responses from other assistants who analyzed different parts of a college dataset based on the user's query: '{original_user_query}'.\n"
    "Your task is to synthesize these responses into a single, accurate answer. Consider the following:\n"
    "- **Identify specific information:** Look for responses that provide concrete details directly related to the user's query.\n"
    "- **Handle 'Not Found' responses:** Ignore vague or unhelpful responses.\n"
    "- **Prioritize relevant details:** Combine only the useful content.\n"
    "- **Acknowledge absence of information:** If *none* of the assistants provide relevant data, state it clearly.\n"
    "- **Avoid guessing or repetition.**\n\n"
    "Assistant Responses:\n{responses}\n\n"
    "Final Response:"
    "Give user friendly answers dont talk about data and all"
)

# --- Streamlit UI ---
st.title("🎓 College Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Async LLM Functions ---
async def get_chunk_info(data_chunk, llm_index, clarified_query, retry_count=0, backup_index=0):
    for i, item in enumerate(data_chunk):
        item["record_id"] = f"College-{i+1}"
    college_details_str = "\n".join(
        f"{item['record_id']} - " + ", ".join([f"{k}: {v}" for k, v in item.items() if k != 'record_id']) for item in data_chunk
    )
    prompt = individual_prompt_template.format(college_details=college_details_str, user_query=clarified_query)
    try:
        response = await st.session_state.processing_llms[llm_index].ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[INFO] LLM {llm_index+1} failed: {e}")
        if backup_index < len(backup_api_keys):
            backup_key = backup_api_keys[backup_index]
            st.session_state.processing_llms[llm_index] = ChatGroq(api_key=backup_key, model_name=model_name)
            return await get_chunk_info(data_chunk, llm_index, clarified_query, retry_count + 1, backup_index + 1)
        return ""

async def get_final_response(responses, original_query, retry_count=0, backup_index=0):
    valid_responses = [r for r in responses if r.strip() and "don't have information" not in r.lower()]
    
    if not valid_responses:
        return "Sorry, I couldn't find any information related to your query in the available college data."

    prompt = final_prompt_template.format(responses="\n\n".join(valid_responses), original_user_query=original_query)
    try:
        final_llm = ChatGroq(api_key=st.session_state.final_llm_key, model_name=model_name)
        final_response = await final_llm.ainvoke(prompt)
        return final_response.content.strip()
    except Exception as e:
        print(f"[INFO] Final LLM failed: {e}")
        if backup_index < len(backup_api_keys):
            backup_key = backup_api_keys[backup_index]
            st.session_state.final_llm_key = backup_key
            return await get_final_response(valid_responses, original_query, retry_count + 1, backup_index + 1)
        return "Sorry, I couldn't generate a response at the moment."

async def clarify_user_query(user_query, example_data):
    prompt = clarification_prompt_template.format(example_data=example_data, user_query=user_query)
    try:
        response = await st.session_state.clarification_llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[INFO] Clarification failed: {e}")
        return user_query

# --- User Input ---
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("You:", placeholder="Ask something about a college...")
    submit = st.form_submit_button("Send")

# --- Async Query Processing ---
if submit and user_query:
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    with st.spinner("🤖 Processing..."):
        async def main():
            example_college_data = st.session_state.college_data[0] if st.session_state.college_data else {}
            clarified_query = await clarify_user_query(user_query, example_college_data)
            tasks = [get_chunk_info(chunk, i, clarified_query) for i, chunk in enumerate(st.session_state.data_chunks)]
            responses = await asyncio.gather(*tasks)
            final_response = await get_final_response(responses, user_query)
            st.session_state.chat_history.append({"role": "assistant", "text": final_response})
        asyncio.get_event_loop().run_until_complete(main())

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**🧑 You:** {chat['text']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['text']}")
