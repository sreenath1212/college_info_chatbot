# College Assistant Chatbot README

This Streamlit application provides a chatbot that answers questions about colleges based on data provided in a CSV file (`college_data.csv`). It leverages the Groq API and Langchain for natural language processing.

## Features

* **College Data Interaction:** Answers user questions based on the information present in the `college_data.csv` file.
* **Clarification of Queries:** Attempts to understand and rephrase user queries to use full field names from the data structure, improving accuracy.
* **Parallel Processing:** Utilizes multiple Groq LLM instances to process chunks of the college data concurrently for faster response times.
* **Specific Information Retrieval:** Designed to provide answers only when an exact match for the college name is found in the data.
* **Response Aggregation:** Synthesizes responses from multiple LLM instances into a single, coherent answer.
* **Error Handling:** Includes basic error handling for API key issues and LLM failures, with a fallback mechanism using backup API keys.
* **Chat History:** Maintains a record of the conversation for better user experience.

## Setup

### Prerequisites

* Python 3.6 or higher
* Streamlit (`pip install streamlit`)
* Pandas (`pip install pandas`)
* Langchain (`pip install langchain`)
* `langchain-groq` (`pip install langchain-groq`)
* `nest-asyncio` (`pip install nest-asyncio`)

### Configuration

1.  **Create `college_data.csv`:**
    * Prepare a CSV file named `college_data.csv` containing the information about colleges. The structure of this CSV will be used to understand user queries. Ensure the first row contains headers representing the college data fields.

2.  **Set up Groq API Keys:**
    * You need **seven primary Groq API keys**. These should be stored as Streamlit secrets with the keys `GROQ_API_KEY1`, `GROQ_API_KEY2`, ..., `GROQ_API_KEY7`.
    * Optionally, you can provide backup Groq API keys as a list in a Streamlit secret named `GROQ_BACKUP_API_KEYS`. These will be used if any of the primary keys fail.

    ```toml
    # .streamlit/secrets.toml
    GROQ_API_KEY1 = "YOUR_FIRST_GROQ_API_KEY"
    GROQ_API_KEY2 = "YOUR_SECOND_GROQ_API_KEY"
    GROQ_API_KEY3 = "YOUR_THIRD_GROQ_API_KEY"
    GROQ_API_KEY4 = "YOUR_FOURTH_GROQ_API_KEY"
    GROQ_API_KEY5 = "YOUR_FIFTH_GROQ_API_KEY"
    GROQ_API_KEY6 = "YOUR_SIXTH_GROQ_API_KEY"
    GROQ_API_KEY7 = "YOUR_SEVENTH_GROQ_API_KEY"
    GROQ_BACKUP_API_KEYS = ["YOUR_BACKUP_API_KEY_1", "YOUR_BACKUP_API_KEY_2"] # Optional
    ```

## Running the Application

1.  Save the provided Python code as a `.py` file (e.g., `app.py`).
2.  Ensure that the `college_data.csv` file is in the same directory as the Python script.
3.  Open your terminal or command prompt, navigate to the directory where you saved the files, and run the Streamlit application using the command:

    ```bash
    streamlit run app.py
    ```

4.  The application will open in your web browser. You can then start asking questions about the colleges in your `college_data.csv` file.

## How it Works

1.  **Data Loading and Chunking:** The `college_data.csv` is loaded using Pandas and then split into several chunks. The number of chunks is determined by `num_processing_llms` (set to 6 in this code).

2.  **LLM Initialization:** Seven instances of the Groq Chat model (`llama3-70b-8192`) are initialized, each using one of the primary API keys. An additional LLM is initialized for query clarification.

3.  **Query Clarification:** When a user asks a question, a clarification LLM attempts to rephrase the query to use the full names of the fields present in the `college_data.csv`. This helps in more accurate information retrieval.

4.  **Parallel Information Retrieval:** The clarified query and the data chunks are sent to the multiple processing LLM instances concurrently. Each LLM analyzes its assigned chunk of data to find information related to the user's query, specifically looking for exact college name matches.

5.  **Response Aggregation:** The responses from all the processing LLMs are collected. A final LLM synthesizes these responses, extracting relevant information and ignoring responses that indicate no information was found.

6.  **Final Output:** The synthesized response is displayed to the user in the Streamlit chat interface.

## Important Notes

* The accuracy of the chatbot's answers heavily depends on the data present in the `college_data.csv` file.
* The chatbot is designed to only answer questions about colleges that have an *exact* name match in the provided data.
* Ensure you have a stable internet connection as the application relies on the Groq API.
* Monitor your Groq API usage to avoid unexpected charges.
* The use of multiple API keys is intended to improve reliability and potentially speed up processing, but it also increases API call volume.
* Error handling includes retries with backup API keys in case of primary key failures, but it's essential to have valid API keys configured.# college_info_chatbot
