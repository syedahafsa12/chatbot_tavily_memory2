
# --- Required Libraries ---
import os
import warnings
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai 
from langchain_google_genai import ChatGoogleGenerativeAI 
import requests

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Load API Keys from `.env` File ---
# def get_api_keys():
#     env_path = "./.env"  # Ensure the path is correct
#     if not os.path.exists(env_path):
#         raise FileNotFoundError("`.env` file not found. Please upload your API key file to `/content/.env`.")

#     # Load environment variables from .env
#     load_dotenv(env_path)
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     tavily_api_key = os.getenv("TAVILY_API_KEY")

#     if not gemini_api_key:
#         raise ValueError("Gemini API Key not found in `.env` file. Please ensure `GEMINI_API_KEY` is set.")
#     if not tavily_api_key:
#         raise ValueError("Tavily API Key not found in `.env` file. Please ensure `TAVILY_API_KEY` is set.")

#     return gemini_api_key, tavily_api_key
def load_api_key():
    """Fetch API keys from Streamlit secrets."""
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY")
    if not gemini_api_key or not tavily_api_key:
        raise ValueError("API keys not set in Streamlit secrets.")
    return gemini_api_key, tavily_api_key


# --- Tavily Search Function ---
def perform_tavily_search(query, api_key):
    """Perform a search using Tavily REST API."""
    url = "https://api.tavily.com/search"  # Replace with actual Tavily API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "num_results": 5}  # Adjust payload as needed

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        return response.json().get("results", [])
    except Exception as e:
        return {"error": str(e)}

# --- Generate Short Description ---
def generate_short_description(title, fallback="No description available"):
    """
    Generate a short description if Tavily's response does not provide one.
    Uses the Gemini model to summarize the title if needed.
    """
    try:
        # Ensure the Gemini API is configured
        if not genai._api_key:
            return fallback

        prompt = (
            f"Provide a short, 4-line summary for the following topic:\n\n"
            f"Title: {title}\n\n"
            "Summary:"
        )
        response = genai.generate_text(prompt=prompt)
        return response.generations[0].text.strip()
    except Exception as e:
        return fallback

# --- Streamlit App Setup ---
def create_streamlit_chatbot():
    st.title("🤖 Conversational Chatbot with Gemini & Tavily")
    st.subheader("Ask me anything or search the web!")

    # Initialize API Keys
    try:
        gemini_api_key, tavily_api_key = load_api_key()
        genai.configure(api_key=gemini_api_key)  # Configure the Gemini API
    except Exception as e:
        st.error(f"Error configuring APIs: {e}")
        return

    # Initialize ChatGoogleGenerativeAI
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return

    # Session State for Conversation History
    if "history" not in st.session_state:
        st.session_state["history"] = []  # Initialize conversation history

    # User Choice: Chat or Search
    mode = st.radio("Choose your action:", ["Chat with Gemini", "Search with Tavily"])

    # User Input
    user_input = st.text_input("You💜:", placeholder="Type your question or search query here...")

    if user_input:
        try:
            if mode == "Search with Tavily":
                # Perform Tavily search
                results = perform_tavily_search(query=user_input, api_key=tavily_api_key)
                if "error" in results:
                    st.error(f"Tavily Search Error: {results['error']}")
                elif not results:
                    st.warning("No results found for your query.")
                else:
                    st.markdown("### Tavily Search Results")
                    for i, result in enumerate(results, 1):
                        title = result.get("title", "No title available")
                        snippet = result.get("snippet", None)

                        # If snippet is missing, generate a short description
                        if not snippet:
                            snippet = generate_short_description(title)

                        url = result.get("url", "#")
                        st.write(f"{i}. **{title}**")
                        # st.write(snippet)
                        st.write(f"[Link]({url})")
                        st.divider()

            elif mode == "Chat with Gemini":
                # Format the conversation history as context
                context = "\n".join(
                    [f"You💜: {chat['user']}\nBot🤖: {chat['bot']}" for chat in st.session_state["history"]]
                )
                full_input = f"{context}\nYou💜: {user_input}\nBot🤖:"

                # Get the bot response
                response = llm.invoke(full_input)

                # Add user input and bot response to conversation history
                st.session_state["history"].append({"user": user_input, "bot": response.content})

                # Display conversation history
                st.markdown("### Conversation History")
                for chat in st.session_state["history"]:
                    st.write(f"**You💜:** {chat['user']}")
                    st.write(f"**Bot🤖:** {chat['bot']}")
                    st.divider()

        except Exception as e:
            st.error(f"Error processing your request: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    create_streamlit_chatbot()
