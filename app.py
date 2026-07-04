import streamlit as st
import requests
import base64

# Configure the Streamlit page
st.set_page_config(page_title="Agentic Audio Gateway", layout="centered")

# UI Headers
st.title("🎙️ Agentic Audio Gateway")
st.markdown("### Compound AI System: RAG + DSP Execution")
st.write("This interface communicates with the local FastAPI agent to diagnose and fix audio issues autonomously.")

# 1. Natural Language Input
query = st.text_area(
    "User Instruction:", 
    value="First, query the audio transcripts for 'File_A' to identify its specific audio issue. Then, run the noise reduction filter using the uploaded audio file to fix it."
)

# 2. File Upload Input (Drag and Drop)
uploaded_file = st.file_uploader("Upload Audio File (WAV)", type=["wav"])

direct_dsp = st.checkbox(
    "Direct DSP only (skip agent — use when Anthropic credits are exhausted)",
    value=True,
)

# Execution Trigger
if st.button("Run Agent Engine"):
    if uploaded_file is None:
        st.warning("Please upload a .wav audio file first to proceed.")
    else:
        with st.spinner(
            "Running direct DSP enhancement..."
            if direct_dsp
            else "Agent is thinking and executing..."
        ):
            try:
                # Convert the uploaded physical file into the base64 string our API needs
                audio_bytes = uploaded_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                if direct_dsp:
                    url = "http://localhost:8000/enhance-audio"
                    payload = {"audio_b64": audio_base64}
                else:
                    url = "http://localhost:8000/process-audio"
                    payload = {
                        "query": query,
                        "audio_b64": audio_base64,
                    }
                
                # Call the live FastAPI backend
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                
                if response.status_code == 200:
                    res_data = response.json()

                    st.success(
                        "DSP Enhancement Complete!"
                        if direct_dsp
                        else "Agent Execution Complete!"
                    )

                    if "processed_audio_b64" in res_data:
                        processed_b64 = res_data["processed_audio_b64"]
                        processed_bytes = base64.b64decode(processed_b64)

                        st.markdown("---")
                        st.subheader("Listen to Processed Audio")
                        st.audio(processed_bytes, format="audio/wav")

                        st.download_button(
                            label="Download Cleaned Audio",
                            data=processed_bytes,
                            file_name="cleaned_audio.wav",
                            mime="audio/wav",
                        )

                        st.caption(
                            f"Processed {len(processed_bytes) / 1024:.1f} KB of audio. "
                            "The enhanced file is in the player above — use the download button to save it."
                        )

                        with st.expander("Response details"):
                            st.json(
                                {
                                    "status": res_data.get("status"),
                                    "result": res_data.get("result"),
                                    "processed_audio_b64": (
                                        f"<{len(processed_b64)} chars omitted — use player above>"
                                    ),
                                }
                            )
                    elif "output_path" in res_data:
                        st.info(
                            f"Audio processed and saved locally at: {res_data['output_path']}"
                        )
                        with st.expander("Response details"):
                            st.json(res_data)
                    else:
                        st.warning("Request succeeded but no processed audio was returned.")
                        with st.expander("Response details"):
                            st.json(res_data)

                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    if response.status_code == 402:
                        st.error(
                            "Anthropic API credits exhausted. Enable **Direct DSP only** above "
                            "to process audio without the agent, or add credits at "
                            "https://console.anthropic.com/settings/billing"
                        )
                    else:
                        st.error(f"Backend Error {response.status_code}: {detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed. Make sure your FastAPI uvicorn server is running on port 8000 in your other terminal tab!")
