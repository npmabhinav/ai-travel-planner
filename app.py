import streamlit as st
from agent import setup_travel_agent
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Config ---
st.set_page_config(
    page_title="AI Travel Agent 🌍",
    page_icon="✈️",
    layout="centered"
)

# --- Basic Styling ---
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Main UI ---
def main():
    st.title("🌍 AI Travel Planner")
    st.write("Powered by **Gemini 2.5 Flash** & **LangGraph**")

    # Sidebar for API Key
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key from Google AI Studio"
        )
        st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")

    # Input Form
    with st.form("travel_form"):
        col1, col2 = st.columns(2)

        with col1:
            destination = st.text_input("Destination", "Kyoto, Japan")
            start_date = st.text_input("Start Date", "October 10, 2025")
            budget = st.text_input("Budget (with currency)", "$2000 USD")

        with col2:
            days = st.number_input("Duration (Days)", min_value=1, max_value=30, value=5)
            travelers = st.number_input("Travelers", min_value=1, max_value=10, value=2)
            style = st.selectbox(
                "Travel Style",
                ["Relaxing", "Adventure", "Foodie", "Cultural", "Budget"]
            )

        submitted = st.form_submit_button("🚀 Generate Itinerary")

    if submitted:
        if not api_key:
            st.error("❌ Please enter your Gemini API Key.")
            return

        try:
            agent = setup_travel_agent(api_key)

            user_prompt = f"""
            Plan a trip to {destination}.
            Start Date: {start_date}
            Duration: {days} days
            Travelers: {travelers}
            Budget: {budget}
            Style: {style}
            """

            initial_state = {
                "messages": [
                HumanMessage(
                content=f"""
You are given ALL required trip details below.
DO NOT ask questions.
DO NOT request more information.
Generate the FINAL itinerary directly.

Destination: {destination}
Start Date: {start_date}
Duration: {days} days
Travelers: {travelers}
Budget: {budget}
Travel Style: {style}
"""
        )
    ]
}


            with st.spinner("🤖 Planning your trip..."):
                events = agent.stream(initial_state, stream_mode="values")

                final_output = None
                for event in events:
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            final_output = last_msg.content

            st.success("✅ Itinerary Ready!")
            st.markdown("### 🗺️ Your Personalized Itinerary")
            st.markdown(final_output)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    main()
