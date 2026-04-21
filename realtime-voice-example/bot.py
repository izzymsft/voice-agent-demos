#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# Monkey-patch to fix Azure Realtime API compatibility
# Azure doesn't support certain fields that OpenAI's API does
from pipecat.services.openai.realtime import events as openai_events
import json

# Patch SessionUpdateEvent
_original_session_update_model_dump = openai_events.SessionUpdateEvent.model_dump

def _patched_session_update_model_dump(self, *args, **kwargs):
    dump = _original_session_update_model_dump(self, *args, **kwargs)
    # Remove 'type' and 'object' fields that Azure doesn't support
    if "session" in dump:
        dump["session"].pop("type", None)
        dump["session"].pop("object", None)
    return dump

openai_events.SessionUpdateEvent.model_dump = _patched_session_update_model_dump

# Patch ResponseCreateEvent
_original_response_create_model_dump = openai_events.ResponseCreateEvent.model_dump

def _patched_response_create_model_dump(self, *args, **kwargs):
    dump = _original_response_create_model_dump(self, *args, **kwargs)
    # Remove 'output_modalities' field that Azure doesn't support
    if "response" in dump and dump["response"]:
        dump["response"].pop("output_modalities", None)
    return dump

openai_events.ResponseCreateEvent.model_dump = _patched_response_create_model_dump

# Azure uses different event names than OpenAI - map them
AZURE_TO_OPENAI_EVENT_MAP = {
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
}

# Patch parse_server_event to handle Azure-specific events
_original_parse_server_event = openai_events.parse_server_event

def _patched_parse_server_event(str_data):
    try:
        # First, try to map Azure event names to OpenAI event names
        event = json.loads(str_data)
        event_type = event.get("type", "")
        
        if event_type in AZURE_TO_OPENAI_EVENT_MAP:
            event["type"] = AZURE_TO_OPENAI_EVENT_MAP[event_type]
            str_data = json.dumps(event)
        
        return _original_parse_server_event(str_data)
    except Exception as e:
        # If it's an unimplemented event type, create a generic ServerEvent
        if "Unimplemented server event type" in str(e):
            event = json.loads(str_data)
            logger.debug(f"Ignoring unhandled Azure event: {event.get('type', 'unknown')}")
            # Return a generic ServerEvent that won't crash the pipeline
            return openai_events.ServerEvent(event_id=event.get("event_id", ""), type=event.get("type", "unknown"))
        raise

openai_events.parse_server_event = _patched_parse_server_event

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
"You are an Azure Realtime LLM Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.

Start by greeting the user warmly and introducing yourself.
"""

azure_api_key=os.getenv("AZURE_REALTIME_API_KEY", "")
azure_base_url=os.getenv("AZURE_REALTIME_BASE_URL", "")

async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    llm = AzureRealtimeLLMService(
        api_key=azure_api_key,
        base_url=azure_base_url,
        settings=AzureRealtimeLLMService.Settings(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            }
        ],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            user_aggregator,
            llm,  # LLM
            pipecat_transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
