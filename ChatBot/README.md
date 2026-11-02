# Orbit chatbot

## Run locally

1. Revoke any token that has been pasted into a chat, committed, or shared. Create a replacement Hugging Face token with **Inference Providers** permission in [Hugging Face token settings](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained).
2. Start the Python backend in a PowerShell terminal from this folder:

   ```powershell
   $secureToken = Read-Host "New Hugging Face token" -AsSecureString
   $credential = New-Object System.Net.NetworkCredential -ArgumentList '', $secureToken
   $env:HF_TOKEN = $credential.Password
   $secureToken.Dispose()
   python app.py
   ```

3. Keep the Python terminal running, then open `Index.html` with VS Code Live Server. Chat requests will be sent to the Python backend at `http://127.0.0.1:8000`.

Alternatively, double-click `Run-Chatbot.bat` to serve both the page and API from Python without Live Server.

The token stays on the Python server and is not sent to the browser. The backend binds to localhost and permits API calls from local HTTP development origins (`localhost`, `127.0.0.1`, or `::1`) so Live Server can connect. The default model is `meta-llama/Llama-3.1-8B-Instruct`; to use another model available through Hugging Face Inference Providers, set `HF_MODEL` before starting the server. Set `PORT` to change the local port.

`GET /api/health` reports whether a token is configured without returning it. `POST /api/chat` accepts a JSON `messages` array with `user` and `assistant` roles and calls Hugging Face's OpenAI-compatible chat-completions API.

If chat returns HTTP 402, the request reached Hugging Face but the account has no inference credits available. Check [billing](https://huggingface.co/settings/billing) and [Inference Providers usage](https://huggingface.co/settings/inference-providers/overview); add credits, use eligible included credits, or configure a provider key in Hugging Face settings. Changing code or replacing the token does not add credits.

## Notes

- No third-party Python packages are needed; the backend uses the Python standard library.
- `.env.example` documents the settings only; the app does not load `.env` files. Set `HF_TOKEN` in PowerShell as shown above.
- Do not open `Index.html` directly as a `file://` URL. Browsers block its API request; use VS Code Live Server or the Python-hosted page.
- Keep API keys private and rotate them if exposed. `.env` is ignored by Git.
