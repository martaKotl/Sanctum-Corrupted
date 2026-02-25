import os
import json
import re
import requests
import subprocess
import time
from datetime import datetime
from typing import Tuple
try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleRequest
except Exception:
    service_account = None
    GoogleRequest = None
from typing import Optional, List


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_ENDPOINT = os.getenv('GEMINI_API_ENDPOINT')
GEMINI_MODEL = os.getenv('GEMINI_MODEL') or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
LOG_ENABLED = str(os.getenv('GEMINI_LOG', '0')).lower() in ('1', 'true', 'yes')
LOG_PATH = os.getenv('GEMINI_LOG_PATH', 'gemini_calls.log')


def _log_call(prompt: str, resp_text: str, resp_json: dict | None = None):
    if not LOG_ENABLED:
        return
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            ts = datetime.utcnow().isoformat()
            f.write(f'[{ts}] MODEL={GEMINI_MODEL} ENDPOINT={GEMINI_API_ENDPOINT}\n')
            f.write('PROMPT:\n')
            f.write(prompt + '\n')
            f.write('RESPONSE_TEXT:\n')
            f.write(str(resp_text) + '\n')
            if resp_json is not None:
                try:
                    f.write('RESPONSE_JSON:\n')
                    f.write(json.dumps(resp_json) + '\n')
                except Exception:
                    f.write('RESPONSE_JSON: <unserializable>\n')
            f.write('---\n')
    except Exception:
        pass

if (not GEMINI_API_ENDPOINT) and GEMINI_API_KEY and GEMINI_API_KEY.startswith('AIza'):
    model = GEMINI_MODEL or 'gemini-2.5-flash'
    GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"


class GeminiError(Exception):
    pass


def _build_headers():
    headers = {'Content-Type': 'application/json'}
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('AIza'):
        headers['Authorization'] = f'Bearer {GEMINI_API_KEY}'
    return headers


def _get_vertex_access_token() -> str:
    keyfile = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if keyfile and service_account and GoogleRequest:
        creds = service_account.Credentials.from_service_account_file(
            keyfile, scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        creds.refresh(GoogleRequest())
        if creds and creds.token:
            return creds.token

    try:
        res = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, check=True)
        token = res.stdout.strip()
        if token:
            return token
    except Exception:
        pass

    raise GeminiError('Unable to obtain Vertex access token; set GOOGLE_APPLICATION_CREDENTIALS or login with gcloud')


def _extract_text_from_response(resp_json: dict) -> Optional[str]:
    # Try several common response shapes (best-effort)
    # 1) Google generative response: output[0].content[0].text
    try:
        out = resp_json.get('output')
        if isinstance(out, list) and out:
            content = out[0].get('content')
            if isinstance(content, list) and content:
                t = content[0].get('text')
                if t:
                    return t
    except Exception:
        pass

    # 2) candidates -> text
    try:
        cand = resp_json.get('candidates')
        if isinstance(cand, list) and cand:
            first = cand[0]
            # content may be a dict with parts -> text
            if 'content' in first:
                cont = first['content']
                if isinstance(cont, dict):
                    parts = cont.get('parts')
                    if isinstance(parts, list) and parts:
                        p0 = parts[0]
                        if isinstance(p0, dict) and 'text' in p0 and isinstance(p0['text'], str):
                            return p0['text']
                    # sometimes content itself may have 'text'
                    if 'text' in cont and isinstance(cont['text'], str):
                        return cont['text']
                elif isinstance(cont, str):
                    return cont
            if 'text' in first and isinstance(first['text'], str):
                return first['text']
    except Exception:
        pass

    # 3) choices (OpenAI-like)
    try:
        choices = resp_json.get('choices')
        if isinstance(choices, list) and choices:
            c = choices[0]
            if 'message' in c and isinstance(c['message'], dict):
                return c['message'].get('content')
            if 'text' in c:
                return c['text']
    except Exception:
        pass

    # 4) simple fields
    for k in ('text', 'response', 'output_text'):
        if k in resp_json and isinstance(resp_json[k], str):
            return resp_json[k]

    # 5) Vertex/AI Platform predictions
    try:
        preds = resp_json.get('predictions')
        if isinstance(preds, list) and preds:
            first = preds[0]
            if isinstance(first, dict):
                for k in ('content', 'generated_text', 'text'):
                    if k in first and isinstance(first[k], str):
                        return first[k]
            if isinstance(first, str):
                return first
    except Exception:
        pass

    return None


def query_llm(prompt: str, timeout: float = 60.0) -> str:
    """Send prompt to configured Gemini/LLM endpoint and return text output.

    Requires environment variables:
    - GEMINI_API_ENDPOINT : full HTTPS URL to POST the request to
    - GEMINI_API_KEY : API key or bearer token

    The function attempts to parse multiple common response formats.
    """
    if not GEMINI_API_ENDPOINT:
        raise GeminiError('GEMINI_API_ENDPOINT is not set')

    # Build payloads for common endpoints.
    payload = None
    if 'generativelanguage.googleapis.com' in (GEMINI_API_ENDPOINT or '') and ':generateContent' in (GEMINI_API_ENDPOINT or ''):
        # v1beta generateContent expects system_instruction + contents array (chat-like)
        payload = {
            'system_instruction': {'parts': [{'text': ''}]},
            'contents': [
                {'role': 'user', 'parts': [{'text': prompt}]}
            ]
        }
    elif 'generativelanguage.googleapis.com' in (GEMINI_API_ENDPOINT or ''):
        # older Generative Language v1 shapes
        payload = {
            'prompt': {'text': prompt},
            'maxOutputTokens': 512,
        }
    elif 'aiplatform.googleapis.com' in (GEMINI_API_ENDPOINT or ''):
        # Vertex AI predict/generate: send as instances with optional parameters
        payload = {
            'instances': [
                {
                    'content': prompt
                }
            ],
            'parameters': {
                'maxOutputTokens': 512
            }
        }
    else:
        payload = {
            'prompt': prompt,
            'max_tokens': 512,
        }

    headers = _build_headers()

    # If calling Vertex AI endpoints, add a fresh access token
    if 'aiplatform.googleapis.com' in (GEMINI_API_ENDPOINT or ''):
        token = _get_vertex_access_token()
        headers['Authorization'] = f'Bearer {token}'

    retries = int(os.getenv('GEMINI_RETRIES', '2'))
    backoff_base = float(os.getenv('GEMINI_BACKOFF_BASE', '1.0'))

    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(GEMINI_API_ENDPOINT, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                sleep = backoff_base * (2 ** attempt)
                time.sleep(sleep)
                continue
            raise GeminiError(f'Network error calling Gemini endpoint: {e}')

        if r.status_code >= 500 and attempt < retries:
            try:
                time.sleep(backoff_base * (2 ** attempt))
            except Exception:
                pass
            continue
        if r.status_code >= 400:
            raise GeminiError(f'LLM returned HTTP {r.status_code}: {r.text}')

    if r.status_code >= 400:
        raise GeminiError(f'LLM returned HTTP {r.status_code}: {r.text}')

    resp_json = None
    try:
        resp_json = r.json()
    except ValueError:
        resp_text = r.text
        _log_call(prompt, resp_text, None)
        return resp_text

    text = _extract_text_from_response(resp_json)
    if text is None:
        resp_text = json.dumps(resp_json)
        _log_call(prompt, resp_text, resp_json)
        return resp_text
    _log_call(prompt, text, resp_json)
    return text


def choose_action_from_llm(enemy_state_text: str, available_actions: List[int], timeout: float = 5.0) -> int:
    """High-level helper: ask the LLM to pick one action index from available_actions.

    The LLM is instructed to reply with a single integer (the index in available_actions)
    or the literal index number corresponding to the enemy hand. The returned int is
    the chosen action index (relative to enemy.hand). If the LLM returns non-integer
    text, this function will try to parse an integer. It raises GeminiError on problems.
    """
    examples_text = (
        "EXAMPLE 1 (Defensive):\n"
        "Enemy:\n  name=Goblin\n  health=5/10\n  mana=2/3\n  Hand:\n   0: Bandages (mana=1) -- heal 3 HP\n   1: Dodge (mana=1) -- grant Dodge effect\n  Available action indices: 0,1,10\n"
        "Player:\n  health=2/10\n  mana=1/3\n"
        "Correct reply: {\"choice\": 0}\n\n"
        "EXAMPLE 2 (Offensive):\n"
        "Enemy:\n  name=Orc\n  health=2/10\n  mana=2/3\n  Hand:\n   0: Heavy Strike (mana=2) -- deal 6 damage\n   1: Shield Up (mana=1) -- gain 5 shield\n  Available action indices: 0,1,10\n"
        "Player:\n  health=8/10\n  mana=2/3\n"
        "Correct reply: {\"choice\": 0}\n\n"
    )

    prompt = (
        "You are an AI game agent deciding which card an enemy should play.\n"
        "For guidance, here are examples showing the required JSON-only output format:\n\n"
        + examples_text + "\n"
        "Now decide for the following state (note: index 10 means end-turn / skip):\n"
        "Enemy state:\n" + enemy_state_text + "\n"
        "Available action indices:\n" + ",".join(map(str, available_actions)) + "\n"
        "Return exactly one JSON object, for example: {\"choice\": 1}. Do not output any additional text or explanation.\n"
    )

    raw = query_llm(prompt, timeout=timeout)
    # Clean common wrappers (Markdown code fences, ```json ... ```) before parsing
    raw = _clean_llm_output(raw)

    # Try to parse JSON first
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict):
            for key in ('choice', 'index'):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, int):
                        return val
                    # sometimes model returns numeric string
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
        # if parsed JSON but doesn't contain expected fields, fall through to fallback
    except Exception:
        pass

    # Fallback: extract first integer found in the raw output
    import re
    m = re.search(r"-?\d+", raw)
    if m:
        try:
            return int(m.group(0))
        except Exception as e:
            raise GeminiError(f'Failed to parse integer from LLM output: {e}')

    raise GeminiError(f'LLM did not return a valid JSON choice or integer: {raw}')


def _clean_llm_output(raw: str) -> str:
    """Strip common wrappers and extract JSON-like content from LLM output.

    Handles triple-backtick fenced blocks (optionally labeled ````json```) and
    inline backticks. Falls back to extracting the first {...} substring.
    """
    if not raw:
        return raw
    s = raw.strip()

    # 1) fenced code block ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        return inner

    # 2) inline code `...` that contains JSON
    m2 = re.search(r"`([^`]+)`", s)
    if m2 and m2.group(1).strip().startswith('{'):
        return m2.group(1).strip()

    # 3) extract the first {...} block (best-effort)
    first = s.find('{')
    last = s.rfind('}')
    if first != -1 and last != -1 and last > first:
        return s[first:last+1].strip()

    return s


def choose_actions_from_llm(enemy_state_text: str, available_actions: List[int], timeout: float = 60.0) -> List[int]:
    """Ask the LLM to decide an ordered sequence of action indices for the enemy's entire turn.

    The LLM is instructed to reply with a JSON object containing a key `choices` whose value
    is a list of integers (e.g. {"choices": [0, 2, 10]}). Index 10 means end-turn.
    Returns a list of integers. Falls back to extracting all integers from the raw output.
    """
    examples_text = (
        "EXAMPLE 1 (Two-card turn):\n"
        "Enemy:\n  name=Goblin\n  health=5/10\n  mana=3/3\n  Hand:\n   0: Bandages (mana=1) -- HP +3\n   1: Strike (mana=2) -- Deal 3 damage\n  Available action indices: 0,1,10\n"
        "Player:\n  health=2/10\n  mana=1/3\n"
        "Correct reply: {\"choices\": [1, 0, 10]}\n\n"
        "EXAMPLE 2 (End-turn only):\n"
        "Enemy:\n  name=Slime\n  health=2/10\n  mana=0/3\n  Hand:\n   0: Shield Up (mana=2) -- shield +3 for one turn\n  Available action indices: 10\n"
        "Player:\n  health=8/10\n  mana=2/3\n"
        "Correct reply: {\"choices\": [10]}\n\n"
    )

    prompt = (
        "You are an AI game agent deciding which cards an enemy should play for its entire turn.\n"
        "Return exactly one JSON object with a single key `choices` whose value is an array of integers.\n"
        "Index 10 means end-turn; you may include 10 at the end to stop playing. Do not output any text besides the JSON.\n\n"
        + examples_text + "\n"
        "Now decide for the following state (note: index 10 means end-turn / skip):\n"
        "Enemy state:\n" + enemy_state_text + "\n"
        "Available action indices:\n" + ",".join(map(str, available_actions)) + "\n"
        "Return exactly one JSON object, for example: {\"choices\": [1,2,10]}. Do not output any additional text or explanation.\n"
    )

    raw = query_llm(prompt, timeout=timeout)

    # Try JSON parse first
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict):
            for key in ('choices', 'choice_sequence', 'sequence', 'plays'):
                if key in obj and isinstance(obj[key], list):
                    items = []
                    for v in obj[key]:
                        if isinstance(v, int):
                            items.append(v)
                        elif isinstance(v, str) and v.isdigit():
                            items.append(int(v))
                    if items:
                        return items
            # Single-choice legacy support
            for key in ('choice', 'index'):
                if key in obj:
                    v = obj[key]
                    if isinstance(v, int):
                        return [v]
                    if isinstance(v, str) and v.isdigit():
                        return [int(v)]
    except Exception:
        pass

    # Fallback: extract all integers from the raw output in order
    import re
    nums = re.findall(r"-?\d+", raw)
    if nums:
        return [int(n) for n in nums]

    raise GeminiError(f'LLM did not return valid JSON choices or integer list: {raw}')
