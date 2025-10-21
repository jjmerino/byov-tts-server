# Bring your own voice F5 TTS Server

A minimal "BYOV" (bring your own voices) server that currently wraps F5-TTS for generating audio from your own collection of voices stored on disk.

Note: Currently uses the model and default weights from: https://github.com/SWivid/F5-TTS. But more may be supported in the future as needed by my other projects. Feel free to contribute if you need that sooner.

## Features

- 🎯 Simple REST API for text-to-speech generation
- Bring your own voices and variations in data/{voice_id}/{variation_id}.{extension}, by providing a reference audio wav and txt file.

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support (recommended) or local Python environment
- Must provide your own reference files in `data/voices/` directory. See 

### Running with Docker

```bash
# Build and start the server
docker-compose up -d

# Check logs
docker-compose logs -f
```

The API will be available at `http://localhost:7861`

### Running Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Install F5-TTS (if not already installed)
pip install git+https://github.com/SWivid/F5-TTS.git

# Run the server
python3 app.py
```

### Testing the API

Once the server is running, use the included test suite to verify everything works:

```bash
# Run all tests
python3 test_api.py

# Run a specific test (e.g., narrative text)
python3 -c "from test_api import setup_test_directories, test_generate_narrative; setup_test_directories(); test_generate_narrative()"
```

The test suite will:
- ✅ Test all API endpoints (`/health`, `/voices`, `/generate`)
- ✅ Generate sample audio files in `test_data/test_api/`
- ✅ Test various scenarios (speed variations, long text, narrative text, error handling)
- ✅ Provide detailed output showing what passed/failed

**Test outputs include:**
- `basic_generation.wav` - Simple "Hello world" test
- `narrative_text.wav` - Narrative narrator-style speech
- `long_text_generation.wav` - Longer paragraph with descriptive content
- `speed_0.5.wav`, `speed_1.0.wav`, `speed_1.5.wav` - Speed variations
- And more...

## API Endpoints

### Generate Speech

Generate audio using a pre-configured voice.

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
  "voice_id": "test_voice",
  "variation": "variation",
  "text": "Hello, this is a test."
}
```

**Optional Parameters:**
```json
{
  "speed": 1.0,
  "nfe_step": 32,
  "cross_fade_duration": 0.15,
  "seed": -1,
  "remove_silence": false
}
```

**Response:** Audio file (WAV format, 24kHz)

**Example with curl:**
```bash
curl -X POST http://localhost:7861/generate \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "test_voice",
    "variation": "variation",
    "text": "Hello, this is a test."
  }' \
  --output output.wav
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:7861/generate",
    json={
        "voice_id": "test_voice",
        "variation": "variation",
        "text": "Hello, this is a test."
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved to output.wav")
else:
    print(f"Error: {response.json()}")
```

### List Available Voices

Get a list of all available voices and their variations.

**Endpoint:** `GET /voices`

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "test_voice",
      "variations": ["test_voice", "variation"]
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:7861/voices
```

### Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Example:**
```bash
curl http://localhost:7861/health
```

## Voice Data Structure

Voices are stored in the `data/voices/` directory:

```
data/voices/
└── [voice_id]/
    ├── [voice_id].wav      # Default reference audio
    ├── [voice_id].txt      # Default reference text
    ├── [variation].wav     # Optional variation audio
    └── [variation].txt     # Optional variation text
```

### Example Structure

```
data/voices/
└── test_voice/
    ├── test_voice.wav
    ├── test_voice.txt
    ├── variation.wav
    └── variation.txt
```

### Adding a New Voice

1. Create a directory in `data/voices/` with your voice ID
2. Add a reference audio file (WAV, 3-12 seconds recommended), leaving 1 second at the end.
3. Add a text file with the transcription of the audio
4. Optionally add variations with different file names. The default variation has the same name as the folder.

Example:
```bash
mkdir -p data/voices/my_voice
# Add your audio and text files
echo "This is my reference audio." > data/voices/my_voice/my_voice.txt
cp reference.wav data/voices/my_voice/my_voice.wav
```

## Configuration

Environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `7861`)
- `VOICES_DIR`: Path to voices directory (default: `data/voices` for local dev, set to `/app/data/voices` in Docker)
- `MODEL_NAME`: TTS model to use (default: `F5-TTS`)

## Error Responses

- `400`: Invalid parameters or missing text
- `404`: Voice ID or variation not found
- `500`: Model inference error

## Using the Test Suite as a Client Reference

The `test_api.py` file serves as both a test suite and a reference implementation for building clients. Each test function demonstrates:

- How to structure API requests
- How to handle responses and save audio files
- How to use optional parameters (speed, variation, etc.)
- How to handle errors

Example from the test suite:
```python
import requests

response = requests.post(
    "http://localhost:7861/generate",
    json={
        "voice_id": "test_voice",
        "variation": "variation",
        "text": "Your text here"
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
```

## Performance Notes

- First request may be slower as models are loaded (~30-60 seconds)
- GPU is required for reasonable inference speed
- Reference audio should be 3-12 seconds for best results
- The API processes one request at a time to manage GPU memory

## Project Structure

```
byov-tts-server/
├── app.py                  # FastAPI server
├── test_api.py             # Test suite and client reference
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker orchestration
├── README.md               # This file
├── data/
│   └── voices/             # Voice profiles (not in git)
└── test_data/              # Generated test outputs (not in git)
```

## License

This project is under MIT LICENSE, but on runtime, this server will download the F5-TTS default weights which may differ from this license.

See [F5-TTS repository](https://github.com/SWivid/F5-TTS) for latest model license information.