# FinRAG-Indic demonstration guide

## Start the application

From the repository root:

```bash
uv sync --dev --locked
cp .env.example .env
```

Configure `.env` locally with an account-available Gemini model and API key.
Never commit `.env`.

Terminal 1:

```bash
uv run uvicorn finrag_indic.api.app:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
uv run streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The sidebar should report the
backend and retrieval index as ready.

## Temporary Cloudflare Quick Tunnel

Install `cloudflared` once:

```bash
brew install cloudflare/cloudflare/cloudflared
```

From the repository root, launch the complete temporary demo:

```bash
./scripts/run_public_demo.sh
```

Share only the printed `https://*.trycloudflare.com` URL. The script exposes
Streamlit publicly while FastAPI remains local, verifies both services, writes
runtime logs to ignored `.demo-logs/`, and stops child services on `Ctrl-C`.
The Quick Tunnel is temporary: its URL changes, it is not durable deployment,
and the host Mac must remain powered on with the launcher running.

## Suggested walkthrough

1. Select **English** and ask: `What is simple interest?`
2. Select **हिन्दी** and ask: `अगर UPI लेन-देन असफल हो जाए तो मुझे क्या करना चाहिए?`
3. Select **தமிழ்** and ask: `முதலீட்டில் பல்வகைப்படுத்தல் என்றால் என்ன?`
4. Open **Sources** below an answer to show document title, page, URL, and
   retrieved passage.
5. Demonstrate abbreviation normalization with: `What is SI?` The original
   question is sent to Gemini while retrieval expands SI to Simple Interest.
6. Demonstrate conservative abstention with:
   `What is the exact Bitcoin to INR exchange rate today?`
7. Demonstrate helpful personalized abstention with:
   `Kya mujhe ₹1,00,000 ka loan lena chahiye? Meri monthly income 1000 repay hai.`
   The assistant should not approve or reject the loan, should explain the
   missing loan and affordability information, and should ask whether ₹1,000
   means monthly income or monthly repayment capacity.

Answers are general financial literacy, not personalized advice. Do not enter
PINs, OTPs, passwords, card numbers, or account details.

For latency inspection, the API response includes `retrieval_ms`,
`generation_ms`, `correction_ms`, and `total_ms`. A nonzero correction latency
means the first generated answer needed one bounded citation-repair request.

## Troubleshooting

- `GET http://127.0.0.1:8000/health` checks liveness.
- `GET http://127.0.0.1:8000/system/status` reports provider configuration,
  model, embedding revision, and index availability without exposing secrets.
- If the hosted provider is unavailable, the UI shows an error; do not present
  deterministic mock output as a live Gemini result.
- If the index is missing, confirm `vectors.faiss`, `metadata.json`, and
  `chunks.jsonl` exist under `data/indexes/multilingual-genuine-v2`.
- If the tunnel has no URL, confirm `cloudflared` is installed and inspect
  `.demo-logs/cloudflared.log`.
- If the first response is slow, wait for lazy BGE-M3 loading to complete.

Stop local services with `Ctrl-C` in their terminals. Stop the Quick Tunnel
with `Ctrl-C` in the launcher terminal; it also cleans up services started by
the script.
