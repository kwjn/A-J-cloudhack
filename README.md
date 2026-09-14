# Hawker Hands

Run the Streamlit application from the repository root:

```bash
recognition/.venv/bin/python -m streamlit run app.py
```

This is the only process required for the deployed app. Browser camera capture
and SgSL inference both run through `streamlit-webrtc` inside Streamlit. The
legacy convenience command below now launches that same Streamlit-only process:

```bash
recognition/.venv/bin/python run_app.py
```

### Local recognition tools

`recognition/predict.py` remains available as a standalone OpenCV debugging
tool; it is not part of the deployed startup path:

```bash
recognition/.venv/bin/python recognition/predict.py
```

Record labelled dataset samples separately with `recognition/collect_data.py`.

## WebRTC network configuration

The inline signing camera uses a public STUN server by default. For visitors
behind strict NATs, corporate firewalls, or restrictive mobile networks, copy
`.env.example` to `.env` and configure a TURN relay:

```dotenv
TURN_URL=turn:your-turn-server.example:3478
TURN_USERNAME=your-username
TURN_CREDENTIAL=your-credential
```

All three TURN values are required before the app adds the relay to its WebRTC
configuration. Never commit real TURN credentials. Prefer scoped or temporary
credentials for a public deployment because WebRTC clients necessarily receive
the credentials needed to connect to the relay.

For a hackathon deployment, [Open Relay Project](https://www.metered.ca/tools/openrelay/)
advertises a limited free TURN allowance. [Twilio Network Traversal Service](https://www.twilio.com/docs/stun-turn/api)
is another option and issues temporary TURN credentials through its API;
Twilio's TURN traffic is usage-priced, although trial credit may be available.
