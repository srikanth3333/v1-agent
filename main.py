#
# Minimal Voice Agent for Web - Single File  
# Uses: SmallWebRTC + Deepgram STT/TTS + OpenAI ChatGPT
#
import os
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

SYSTEM_MESSAGE = """
You are a helpful AI voice assistant for web users.
Your goal is to be conversational and helpful.
Your output will be converted to audio so don't include special characters in your answers.
Keep your responses brief and natural. One or two sentences at most.
"""

# Store active WebRTC connections
connections: Dict[str, SmallWebRTCConnection] = {}

# Configure ICE servers for NAT traversal
ice_servers = [
    IceServer(urls="stun:stun.l.google.com:19302"),
]

async def run_bot(webrtc_connection):
    """Main bot pipeline for voice processing"""
    
    # WebRTC transport
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Services: STT, LLM, TTS
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-general",
    )
    
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en"
    )

    # Conversation context
    context = OpenAILLMContext([
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        }
    ])
    
    # Create context aggregator from LLM service
    context_aggregator = llm.create_context_aggregator(context)

    # Create pipeline: Input -> STT -> Context -> LLM -> TTS -> Output
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    # Pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Voice client connected")
        # Start conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Voice client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    logger.info("Starting Voice Agent API")
    yield
    logger.info("Shutting down Voice Agent API")


# FastAPI app setup
app = FastAPI(
    title="Voice Agent API",
    description="Minimal voice agent with SmallWebRTC + Deepgram + OpenAI",
    lifespan=lifespan
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "Voice Agent API running", "version": "1.0"}

@app.post("/api/offer")
async def handle_offer(request: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from client and return SDP answer."""
    pc_id = request.get("pc_id")

    if pc_id and pc_id in connections:
        # Handle reconnections
        webrtc_connection = connections[pc_id]
        await webrtc_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"]
        )
    else:
        # Create new WebRTC connection
        webrtc_connection = SmallWebRTCConnection(ice_servers=ice_servers)
        await webrtc_connection.initialize(
            sdp=request["sdp"],
            type=request["type"]
        )

        # Clean up when client disconnects
        @webrtc_connection.event_handler("closed")
        async def on_closed(connection):
            connections.pop(connection.pc_id, None)
            logger.info(f"WebRTC connection {connection.pc_id} closed")

        # Start bot for this connection
        background_tasks.add_task(run_bot, webrtc_connection)

    answer = webrtc_connection.get_answer()
    connections[answer["pc_id"]] = webrtc_connection
    return answer

@app.get("/client")
async def get_client():
    """Built-in WebRTC client for testing"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Agent - WebRTC Client</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px; 
                text-align: center;
            }
            button { 
                padding: 15px 30px; 
                margin: 10px; 
                font-size: 18px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
            }
            #start { background: #4CAF50; color: white; }
            #stop { background: #f44336; color: white; }
            button:disabled { background: #cccccc; cursor: not-allowed; }
            #status { 
                margin: 20px 0; 
                padding: 10px; 
                border-radius: 5px; 
                font-weight: bold; 
            }
            .ready { background: #e3f2fd; color: #1976d2; }
            .connecting { background: #fff3e0; color: #f57c00; }
            .connected { background: #e8f5e8; color: #388e3c; }
            .error { background: #ffebee; color: #d32f2f; }
            #log { 
                text-align: left; 
                background: #f5f5f5; 
                padding: 15px; 
                border-radius: 5px; 
                height: 300px; 
                overflow-y: auto; 
                margin-top: 20px;
                font-family: monospace;
                font-size: 12px;
                line-height: 1.4;
            }
        </style>
    </head>
    <body>
        <h1>🎤 Voice Agent - WebRTC Client</h1>
        <p>Click "Start Voice Chat" to begin talking with your AI assistant</p>
        
        <button id="start">Start Voice Chat</button>
        <button id="stop" disabled>Stop Voice Chat</button>
        
        <div id="status" class="ready">Ready to start</div>
        <div id="log"></div>

        <script>
            let pc = null;
            let stream = null;
            let pcId = null;
            let isConnected = false;
            
            function log(message) {
                const logDiv = document.getElementById('log');
                const timestamp = new Date().toLocaleTimeString();
                logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function setStatus(message, className) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = className;
            }
            
            function updateButtons(connected) {
                document.getElementById('start').disabled = connected;
                document.getElementById('stop').disabled = !connected;
                isConnected = connected;
            }
            
            document.getElementById('start').onclick = async () => {
                try {
                    setStatus('Requesting microphone...', 'connecting');
                    log('Starting WebRTC voice chat...');
                    
                    // Get microphone access
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true,
                            sampleRate: 16000
                        }
                    });
                    log('✓ Microphone access granted');
                    
                    setStatus('Connecting to voice agent...', 'connecting');
                    
                    // Create WebRTC peer connection
                    pc = new RTCPeerConnection({
                        iceServers: [
                            { urls: 'stun:stun.l.google.com:19302' }
                        ]
                    });
                    
                    // Generate unique ID for this connection
                    pcId = Math.random().toString(36).substring(2, 15);
                    
                    // Add audio track
                    const audioTrack = stream.getAudioTracks()[0];
                    pc.addTrack(audioTrack, stream);
                    log('✓ Added audio track to peer connection');
                    
                    // Handle remote audio
                    pc.ontrack = (event) => {
                        log('✓ Received remote audio stream');
                        const remoteAudio = new Audio();
                        remoteAudio.srcObject = event.streams[0];
                        remoteAudio.autoplay = true;
                    };
                    
                    // Handle connection state changes
                    pc.onconnectionstatechange = () => {
                        log(`Connection state: ${pc.connectionState}`);
                        if (pc.connectionState === 'connected') {
                            setStatus('Connected - Start Speaking!', 'connected');
                            updateButtons(true);
                        } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                            setStatus('Disconnected', 'ready');
                            updateButtons(false);
                        }
                    };
                    
                    pc.oniceconnectionstatechange = () => {
                        log(`ICE connection state: ${pc.iceConnectionState}`);
                    };
                    
                    // Create and send offer
                    const offer = await pc.createOffer({
                        offerToReceiveAudio: true,
                        offerToReceiveVideo: false
                    });
                    
                    await pc.setLocalDescription(offer);
                    log('✓ Created WebRTC offer');
                    
                    // Send offer to server
                    const response = await fetch('/api/offer', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            pc_id: pcId,
                            sdp: offer.sdp,
                            type: offer.type
                        }),
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Server error: ${response.status}`);
                    }
                    
                    const answer = await response.json();
                    log('✓ Received server answer');
                    
                    // Set remote description
                    await pc.setRemoteDescription(new RTCSessionDescription({
                        type: answer.type,
                        sdp: answer.sdp
                    }));
                    log('✓ WebRTC connection established');
                    
                } catch (error) {
                    setStatus('Connection Error', 'error');
                    log(`❌ Error: ${error.message}`);
                    updateButtons(false);
                    console.error('Connection error:', error);
                }
            };
            
            document.getElementById('stop').onclick = async () => {
                log('Stopping voice chat...');
                
                if (pc) {
                    pc.close();
                    pc = null;
                }
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                
                setStatus('Ready to start', 'ready');
                updateButtons(false);
                log('Voice chat stopped');
            };
            
            // Initialize
            log('WebRTC Voice Agent Client loaded');
            log('Requirements: Modern browser with WebRTC support');
            updateButtons(false);
        </script>
    </body>
    </html>
    """)


# Add this route to your existing main.py
@app.get("/")
async def health_check():
    return {"status": "Voice Agent API running", "version": "1.0"}

# Also add this explicit health endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voice-agent"}

if __name__ == "__main__":
    # Validate environment variables
    required_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Create a .env file with your API keys")
        exit(1)
    
    port = int(os.getenv("PORT", 8000))  # Railway compatibility
    
    logger.info("🚀 Starting Voice Agent server")
    logger.info(f"📡 Server: http://localhost:{port}")
    logger.info(f"🎤 WebRTC Client: http://localhost:{port}/client") 
    logger.info(f"🔗 API endpoint: http://localhost:{port}/api/offer")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=port,  # Use dynamic port
        log_level="info"
    )
