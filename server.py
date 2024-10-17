from fastapi import FastAPI, WebSocket
from httpx import AsyncClient
from numpy import frombuffer, max, abs, float32
from io import BytesIO
import asyncio
import soundfile
from starlette.websockets import WebSocketDisconnect

app = FastAPI()
s2t_url = "URLGOESHERE"
INACTIVE_TIMEOUT = 1  # Timeout before processing remaining audio on inactivity
CHUNK_DURATION = 2  # Duration to process audio chunks

# Create the HTTP client globally for reuse
client = AsyncClient()


async def send_audio_to_s2t(audio_data):
    """Send the audio file to the /s2t endpoint via POST request"""
    try:
        # POST the audio to the S2T service
        response = await client.post(s2t_url, files={"audio_input": ('audio.wav', audio_data, 'audio/wav')})
        if response.status_code == 200:
            print(f"Transcription: {response.text}")
        else:
            print(f"Error from S2T: {response.text}")
    except Exception as e:
        print(f"Failed to send audio: {e}")


@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Buffer to store incoming audio chunks
    audio_buffer = bytearray()
    last_received_time = asyncio.get_running_loop().time()

    async def process_and_send_chunk():
        """Process audio buffer and send it to S2T"""
        nonlocal audio_buffer
        if not audio_buffer:
            return

        # Copy the audio buffer to avoid resizing issues
        processing_buffer = audio_buffer[:]
        audio_buffer.clear()  # Clear the original buffer safely after copying

        # Convert buffer to float32 audio data
        audio_data = frombuffer(processing_buffer, dtype=float32)

        # Normalize audio if needed
        max_val = max(abs(audio_data))
        if max_val > 1.0:
            audio_data /= max_val

        # Prepare audio for sending
        audio_buffer_to_send = BytesIO()
        soundfile.write(audio_buffer_to_send, audio_data, 16000, format='WAV')
        audio_buffer_to_send.seek(0)

        # Send audio to S2T
        await send_audio_to_s2t(audio_buffer_to_send)

    async def process_audio_chunks():
        """Periodically process and send chunks of audio"""
        while True:
            await asyncio.sleep(CHUNK_DURATION)
            await process_and_send_chunk()

    process_chunk_task = asyncio.create_task(process_audio_chunks())

    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_buffer.extend(audio_data)  # Append new data to the buffer
            last_received_time = asyncio.get_running_loop().time()
    except WebSocketDisconnect:
        await process_and_send_chunk()  # Process remaining audio on disconnect
        process_chunk_task.cancel()  # Cancel periodic processing
    finally:
        await process_chunk_task  # Ensure task is cleaned up

    # Handle idle timeout to process remaining audio
    while True:
        if asyncio.get_running_loop().time() - last_received_time > INACTIVE_TIMEOUT:
            await process_and_send_chunk()  # Process and send the last chunk if the connection goes idle
            break
