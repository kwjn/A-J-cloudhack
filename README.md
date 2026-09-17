# Clear Conversation / Hawker Hands

A two-way SgSL communication app. Browser camera capture and recognition run
inside Streamlit through `streamlit-webrtc`; only one application process is
required.

## Installation from the repository root

Use Python 3.11 or 3.12 and run all commands from the directory containing
`app.py`. The root requirements include `recognition/requirements.txt`, so a
separate recognition installation is unnecessary.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Keep the supplied recognition models and `assets/signs/*.mp4` in place.

## ElevenLabs configuration

Create `.env` beside `app.py` by copying `.env.example` only if `.env` does not
already exist. Preserve any existing configuration. Fill in these placeholders
locally:

```dotenv
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=your-elevenlabs-voice-id
```

The API key is needed for transcription and speech generation. The voice ID is
needed for text-to-speech. Never commit actual credentials. Without configured
speech services, sign recognition and written translations remain available.

## Startup

Windows PowerShell, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, use `.\.venv\Scripts\python.exe run_app.py`.

macOS / Linux:

```bash
.venv/bin/python -m streamlit run app.py
```

Open the displayed local URL and allow browser camera and microphone access.
For remote deployment, serve the app over HTTPS for browser media access.
Do not launch a separate predictor process for the Streamlit app.

## Supported spoken replies

Staff recordings are transcribed by ElevenLabs. Complete supported replies map
to the supplied SgSL videos:

| Video | Example replies |
| --- | --- |
| YES | `yes`, `can`, `okay`, `ok`, `sure`, `have`, `I have it`, `we have it`, `I can`, `we can` |
| NO | `no`, `cannot`, `can't`, `don't have`, `not available`, `not okay`, `I do not have it`, `I don't have it` |
| PLEASE_REPEAT | `repeat`, `say again`, `again please`, `pardon` |

Matching ignores case, punctuation, and repeated whitespace, accepts curly
apostrophes, and allows polite wrappers such as `yes please` or `sorry, say
again`. Negated or ambiguous longer speech is never classified merely because
it contains `have`, `can`, or `okay`. Unsupported or mixed replies such as
`yes or no` and `can you check?` remain visible as text without a response video.
This is a limited phrase mapper, not general language understanding.

If transcription fails, select **Retry transcription** to retry the same
recording, or record another reply. Routine UI refreshes do not automatically
retry a failed request. Successful recordings are deduplicated.

`YES` and `NO` are staff-response video intents, not extra classifier classes.
The sign recognizer retains its eight existing intents, including `PLEASE_REPEAT`.

## End-to-end demo and checks

1. Start the app, allow camera access, and face the camera until the signer is
   locked and **Start recording** is enabled.
2. Select **Start recording**, wait for the countdown, perform a supported sign
   such as PACK, and select **Stop recording**. Check the translated message and
   confidence. Try ALLERGY and check the allergy alert.
3. Select **Play translation aloud** and verify the ElevenLabs audio matches the
   displayed text. Browser autoplay restrictions may require pressing play.
4. Record staff saying `yes`, `no`, and `please repeat` separately. Confirm the
   transcript and corresponding SgSL video for each.
5. Try `I do not have it`, `I don't have it`, and `not okay`; each should show NO.
   Try `can you check?` and `yes or no`; neither should show YES or another video.
6. With speech service access temporarily unavailable, record a reply and confirm
   an error appears. Restore access and use **Retry transcription** without
   re-recording. Check that normal UI refreshes do not repeat successful requests.
7. Repeat the same sign twice, double-click recording controls, and let a capture
   reach its automatic frame limit. A stale Stop action must not start a capture.
   Also test signer loss and a recording too short to classify.

Camera quality, model accuracy, browser playback, and real ElevenLabs requests
require manual testing. Existing non-camera automated tests can be run with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s recognition -p "test_*.py"
.\.venv\Scripts\python.exe -B -m unittest test_recognition_bridge
```

## Local recognition tools

`recognition/predict.py` is an optional standalone OpenCV debugging tool:

```powershell
.\.venv\Scripts\python.exe recognition/predict.py
```

Dataset collection remains separate in `recognition/collect_data.py`.

## WebRTC network configuration

The inline camera uses a public STUN server by default. For restrictive NATs or
firewalls, configure a TURN relay in your local `.env`:

```dotenv
TURN_URL=turn:your-turn-server.example:3478
TURN_USERNAME=your-turn-username
TURN_CREDENTIAL=your-turn-credential
```

All three TURN settings are required to enable the relay. Use scoped or temporary
credentials for public deployments: WebRTC clients receive the credentials needed
to connect. Never commit real TURN credentials.
