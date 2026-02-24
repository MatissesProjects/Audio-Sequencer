# AudioSequencer 4090 Generation Server

This folder contains the server component meant to run on your **RTX 4090** machine. It listens for requests from the main AudioSequencer app and generates high-quality neural transitions using MusicGen.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server**:
    ```bash
    python server.py
    ```

## Usage

The server will start on `http://0.0.0.0:5000`. 

**On your main machine**:
Update the `REMOTE_GEN_URL` in `src/core/config.py` to point to the IP address of your 4090 machine:
```python
REMOTE_GEN_URL = "http://<4090_MACHINE_IP>:5000/generate"
```
