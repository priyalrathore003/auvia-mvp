"""
main.py
--------
FastAPI server that exposes the Auvia audio enhancement pipeline as a
LangChain agent endpoint.

Architecture (Airlock Pattern):
  1. Client sends base64-encoded audio + natural-language query
  2. Server decodes base64 → temp file on disk
  3. Agent receives the file path (never the raw base64) and calls tools
  4. Tool writes enhanced audio to another temp file
  5. Server reads the output file, encodes to base64, returns to client
  6. Temp files are cleaned up
"""

from dotenv import load_dotenv

load_dotenv()

import base64
import logging
import os
import uuid

import anthropic
from fastapi import FastAPI, HTTPException
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local tool imports
from agent_gateway import enhance_audio
from dsp_pipeline import apply_timbre_enhancement
from rag_storage import query_audio_transcripts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("auvia_main")

# ---------------------------------------------------------------------------
# LangChain Agent Setup (langchain >= 1.3 API)
# ---------------------------------------------------------------------------

# Collect tools
tools = [enhance_audio, query_audio_transcripts]

# Use Anthropic Claude as the LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# System prompt
system_prompt = (
    "You are an expert audio engineer agent. You have access to two tools:\n"
    "1. enhance_audio(file_path: str, output_path: str) — Applies noise reduction and spectral shaping to an audio file.\n"
    "2. query_audio_transcripts(query: str) — Queries the vector store for known audio issues.\n\n"
    "When the user provides an audio file, first use query_audio_transcripts to check if there's "
    "known information about the audio issue. Then use enhance_audio with the file path to fix it.\n\n"
    "IMPORTANT: The audio file has already been saved to disk by the system. "
    "The file path will be provided in the user's message. Always pass the exact file path to enhance_audio."
)

# Create the agent using the new langchain 1.3+ API
agent = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True,
)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Auvia Audio Agent Gateway")


class AudioRequest(BaseModel):
    query: str
    audio_b64: str


class AudioResponse(BaseModel):
    status: str
    result: str
    processed_audio_b64: str


class DirectAudioRequest(BaseModel):
    audio_b64: str


def _raise_for_anthropic_error(exc: Exception) -> None:
    """Map Anthropic client errors to actionable HTTP responses."""
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, anthropic.BadRequestError) and (
        "credit" in lowered or "billing" in lowered
    ):
        raise HTTPException(
            status_code=402,
            detail=(
                "Anthropic API credits exhausted. Add credits at "
                "https://console.anthropic.com/settings/billing or use "
                "POST /enhance-audio for DSP-only processing without the agent."
            ),
        ) from exc
    if isinstance(exc, anthropic.AuthenticationError):
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Check ANTHROPIC_API_KEY in your .env file.",
        ) from exc
    if isinstance(exc, anthropic.RateLimitError):
        raise HTTPException(
            status_code=429,
            detail="Anthropic rate limit reached. Try again shortly.",
        ) from exc


@app.post("/enhance-audio", response_model=AudioResponse)
async def enhance_audio_direct(request: DirectAudioRequest):
    """
    Run the DSP pipeline directly, bypassing the LangChain agent and LLM.
    Use this when Anthropic credits are unavailable.
    """
    try:
        logger.info(
            "Direct DSP enhancement (%d chars base64)", len(request.audio_b64)
        )
        processed_bytes = apply_timbre_enhancement(
            base64.b64decode(request.audio_b64)
        )
        return AudioResponse(
            status="success",
            result="Direct DSP enhancement complete (agent bypassed).",
            processed_audio_b64=base64.b64encode(processed_bytes).decode("utf-8"),
        )
    except Exception as exc:
        logger.exception("Direct DSP enhancement failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/process-audio", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    """
    Receive a base64-encoded audio file + natural-language query,
    run the agent pipeline, and return the enhanced audio as base64.
    """
    # Generate a unique temp file name to avoid collisions
    file_id = uuid.uuid4().hex
    input_file = f"/tmp/auvia_input_{file_id}.wav"
    output_file = f"/tmp/auvia_output_{file_id}.wav"

    try:
        # ── STEP A: Inbound Airlock — decode base64 to file ──────────
        logger.info("Decoding inbound audio (%d chars base64)", len(request.audio_b64))
        with open(input_file, "wb") as f:
            f.write(base64.b64decode(request.audio_b64))

        # ── STEP B: Craft the safe prompt ────────────────────────────
        safe_query = (
            f"{request.query}\n\n"
            f"CRITICAL INSTRUCTION: The user's audio file has been saved to "
            f"'{input_file}'. You must pass this exact file path to the "
            f"enhance_audio tool. The enhanced output will be written to "
            f"'{output_file}'."
        )

        # ── STEP C: Run the agent ────────────────────────────────────
        logger.info("Invoking agent with query: %.100s", request.query)
        # The new create_agent API expects messages in the input
        agent_response = await agent.ainvoke({
            "messages": [HumanMessage(content=safe_query)]
        })
        logger.info("Agent response received")

        # Extract the final output from the response
        # The response contains a 'messages' list; the last AI message has the content
        messages = agent_response.get("messages", [])
        final_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                final_content = msg.content
                break

        # ── STEP D: Outbound Airlock — read output, encode to base64 ─
        if not os.path.exists(output_file):
            # The tool may have written to a different path; check alternatives
            logger.warning("Expected output %s not found; checking alternatives", output_file)
            # Fall back to the default path the tool uses
            fallback = "output/enhanced.wav"
            if os.path.exists(fallback):
                output_file = fallback
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Audio enhancement failed: output file not found.",
                )

        with open(output_file, "rb") as f:
            cleaned_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ── STEP E: Cleanup ──────────────────────────────────────────
        for path in [input_file, output_file]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("Failed to clean up %s", path)

        return AudioResponse(
            status="success",
            result=final_content or "Processing complete.",
            processed_audio_b64=cleaned_b64,
        )

    except HTTPException:
        raise
    except anthropic.APIError as exc:
        logger.exception("Anthropic API error")
        _raise_for_anthropic_error(exc)
    except Exception as exc:
        logger.exception("Agent pipeline failed")
        # Cleanup on error
        for path in [input_file, output_file]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)