from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time
import torch
import ast
from typing import Optional
from requests.exceptions import RequestException
import codecs


from pitpo import evaluator
from pitpo import buffer
from pitpo import config as config_lib
from pitpo import grpo_trainer
import requests
import json
import http.client
import os



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            # Timing: LLM generation (includes batch)
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time,
                    original_prompt=prompt.code
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract a function body from the model's raw output:
    - Preferred: walk back from the last `return` to the nearest `def` and keep only that function body.
    - Fallback: legacy logic that keeps everything after the first `def`.
    - Cleanup: strip markdown fences, unclosed/paired triple-quoted blocks, Note/assistant text, import lines, etc.
    """
    lines = sample.splitlines()

    def _postprocess_body_lines(body_lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        in_triple: bool = False
        for ln in body_lines:
            raw = ln
            s = raw.strip()
            # Skip markdown code fences
            if s.startswith('```'):
                continue
            # Skip / strip triple-quoted docstring blocks
            if "\"\"\"" in raw or "'''" in raw:
                cnt = raw.count("\"\"\"") + raw.count("'''")
                # Toggle the in-block flag and skip this line
                in_triple = not in_triple if (cnt % 2) == 1 else in_triple
                continue
            if in_triple:
                continue
            # Skip comments and explanatory lines
            low = s.lower()
            if s.startswith('#') or low.startswith('note:') or low.startswith('assistant'):
                continue
            # Skip import lines (should not appear inside a function body)
            if s.startswith('import '):
                continue
            # Repair patterns like 'return dvassistant'
            if s.startswith('return') and 'assistant' in s:
                # Truncate before the 'assistant' marker
                cut = s.lower().find('assistant')
                s = s[:cut].rstrip()
            # Normalize indentation: preserve existing indent, otherwise pad to 4 spaces
            if s and not raw.startswith('    '):
                raw = '    ' + s
            elif s:
                raw = raw  # raw = raw  # keep existing indent
            else:
                raw = ''
            cleaned.append(raw)
        # Drop leading/trailing blank lines
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return cleaned

    # 1) Preferred: walk back from the last `return` to the nearest `def`
    last_ret_idx: int | None = None
    for i, ln in enumerate(lines):
        st = ln.strip()
        if 'return' in st and not st.startswith('#'):
            last_ret_idx = i
    if last_ret_idx is not None:
        def_idx: int | None = None
        for j in range(last_ret_idx, -1, -1):
            if lines[j].lstrip().startswith('def '):
                def_idx = j
                break
        if def_idx is not None and def_idx < last_ret_idx:
            body_lines = lines[def_idx + 1:last_ret_idx + 1]
            body_lines = _postprocess_body_lines(body_lines)
            if body_lines:
                return "\n".join(body_lines) + "\n"

    # 2) Fallback: legacy logic — keep everything after the first `def`
    func_body_lineno = None
    for lineno, line in enumerate(lines):
        if line.lstrip().startswith('def '):
            func_body_lineno = lineno
            break
    if func_body_lineno is not None:
        body_lines = lines[func_body_lineno + 1:]
        body_lines = _postprocess_body_lines(body_lines)
        if body_lines:
            return "\n".join(body_lines) + "\n"

    # 3) If everything fails, return the original text (AST validation downstream will filter)
    return sample


def _is_valid_generated_body(body: str) -> bool:
    """Validate generated function body.
    Criteria:
    - Non-empty after stripping
    - Contains at least one 'return'
    - AST-parseable when wrapped in a dummy function
    - Has at least one Return node (with or without value)
    """
    if not isinstance(body, str):
        return False
    stripped = body.strip()
    if not stripped:
        return False
    if 'return' not in stripped:
        return False
    # Ensure lines are indented
    lines = [ln if ln.startswith('    ') or not ln.strip() else ('    ' + ln) for ln in body.splitlines()]
    wrapped = 'def __tmp__():\n' + '\n'.join(lines)
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    class _RetVisitor(ast.NodeVisitor):
        def __init__(self):
            self.has_return = False
        def visit_Return(self, node):
            self.has_return = True
    v = _RetVisitor()
    v.visit(tree)
    return v.has_return


# Decode escape sequences (e.g. "\\n" -> real newline) in the raw model output before printing
def _unescape_raw(text: str) -> str:
    """Attempt to convert escaped sequences like '\\n', '\\t', unicode escapes etc. into real characters."""
    if not isinstance(text, str):
        return text
    try:
        # Try unicode escape decoding which converts backslash escapes to actual characters
        return codecs.decode(text, 'unicode_escape')
    except Exception:
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except Exception:
            return text


def _log_raw_response(raw: str, source: str = 'LLM') -> None:
    """Print the raw response from the model with escapes rendered as real characters."""
    try:
        unescaped = _unescape_raw(raw)
    except Exception:
        unescaped = raw
    print(f"[RAW OUTPUT - {source}]\n{unescaped}\n" + "-"*80, flush=True)


class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        # Default URL (kept for backwards compatibility)
        default_url = "http://127.0.0.1:5000/completions"
        # Allow override via environment variable (so port can be set without code changes)
        env_url = os.environ.get("LLM_SERVER_URL")
        # Initialize with the default; the URL is updated dynamically based on config before each request
        url = env_url if env_url else default_url
        print('url========================================================:', url)
        instruction_prompt = (
            # "You are a Python code generator for equation synthesis.\n"
            # "Return only the function body for the function annotated with @equation.evolve in the prompt below.\n"
            # "Requirements:\n"
            # "- Output only Python code lines, no def line, no decorators, no comments, no markdown.\n"
            # "- Each line must be indented with 4 spaces to fit under the existing def line.\n"
            # "- Use only variables from the function signature (e.g., x, v, ..., params). Access parameters as params[i].\n"
            # "- Vectorized computation compatible with numpy/torch broadcasting; avoid explicit for loops where possible.\n"
            # "- Add a small eps to denominators to avoid division by zero; avoid invalid domains for log/exp/sqrt.\n"
            # "- Must contain at least one return that returns the predicted value with correct shape.\n"
            # "- Output code only. No explanations.\n"
        )
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        # vLLM OpenAI-compatible model name (override via env var)
        self._model_name = os.environ.get("LLM_MODEL_NAME") or os.environ.get("VLLM_MODEL_NAME")
        
        # GRPO support
        self._grpo_trainer = None
        self._use_fine_tuned_model = False

        # vLLM lifecycle management
        self._vllm_pid: Optional[int] = None
        self._vllm_gpu: Optional[str] = None
        self._vllm_port: Optional[int] = None

    def set_grpo_trainer(self, trainer):
        """Set the GRPO trainer instance."""
        self._grpo_trainer = trainer
        print("GRPO trainer set for LocalLLM")

    # ------------------------------------------------------------------ #
    #  vLLM lifecycle (EvoTune-style: Python owns the PID)                #
    # ------------------------------------------------------------------ #

    def start_vllm_server(self, model_path: str, gpu: str, port: int, timeout: int = 300) -> bool:
        """Launch a vLLM server, wait until healthy, store PID.

        This is the **initial** startup called by pipeline.main().
        After GRPO training, use ``restart_vllm_with_model()`` instead.
        """
        self._vllm_gpu = str(gpu)
        self._vllm_port = int(port)
        self._start_vllm_process(model_path)
        return self._wait_vllm_ready(timeout=timeout)

    def restart_vllm_with_model(self, model_path: str) -> bool:
        """Kill current vLLM, restart with *model_path*, wait until ready."""
        if self._vllm_port is None or self._vllm_gpu is None:
            print("[vLLM] No vLLM config set, skipping restart")
            return False
        print(f"[vLLM] === Restarting vLLM with model: {model_path} ===")
        self._kill_vllm()
        time.sleep(3)
        self._start_vllm_process(model_path)
        return self._wait_vllm_ready()

    def cleanup_vllm(self):
        """Kill the vLLM server on shutdown."""
        self._kill_vllm()

    def _start_vllm_process(self, model_path: str):
        import subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self._vllm_gpu)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(self._vllm_port),
            "--dtype", "auto",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
        ]
        print(f"[vLLM] Starting server: GPU={self._vllm_gpu} port={self._vllm_port} model={model_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._vllm_pid = proc.pid
        print(f"[vLLM] Launched with PID {self._vllm_pid}")

    def _kill_vllm(self):
        import signal
        if self._vllm_pid is None:
            print("[vLLM] No PID recorded, nothing to kill")
            return
        try:
            os.kill(self._vllm_pid, signal.SIGTERM)
            print(f"[vLLM] Sent SIGTERM to PID {self._vllm_pid}")
            for _ in range(30):
                time.sleep(1)
                try:
                    os.kill(self._vllm_pid, 0)
                except OSError:
                    print(f"[vLLM] Process {self._vllm_pid} terminated")
                    self._vllm_pid = None
                    return
            os.kill(self._vllm_pid, signal.SIGKILL)
            print(f"[vLLM] Sent SIGKILL to PID {self._vllm_pid}")
            time.sleep(2)
        except ProcessLookupError:
            print(f"[vLLM] Process {self._vllm_pid} already gone")
        self._vllm_pid = None

    def _wait_vllm_ready(self, timeout: int = 300):
        import requests as _req
        url = f"http://127.0.0.1:{self._vllm_port}/health"
        waited = 0
        while waited < timeout:
            try:
                r = _req.get(url, timeout=2)
                if r.status_code == 200:
                    print(f"[vLLM] Server ready after {waited}s")
                    return True
            except Exception:
                pass
            time.sleep(3)
            waited += 3
            if waited % 15 == 0:
                print(f"[vLLM] Waiting... {waited}s")
        print(f"[vLLM] Server NOT ready after {timeout}s")
        return False


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`.

        All inference goes through the vLLM HTTP server. After GRPO training the
        merged model is loaded by a restarted vLLM server (EvoTune-style).
        """
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)
    

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # Refresh the URL based on config (lets the launcher inject a port)
        if hasattr(config, 'llm_server_url') and isinstance(config.llm_server_url, str):
            if self._url != config.llm_server_url:
                print(f"[INF] Switch local LLM URL -> {config.llm_server_url}")
                self._url = config.llm_server_url
        # Use original prompt without additional instructions
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    if isinstance(response, list):
                        for i, res in enumerate(response):
                            pass
                        for res in response:
                            all_samples.append(res)
                    else:
                        all_samples.append(response)
                else:
                    for i in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # Print the raw model output and render escape characters as real newlines
                for idx, raw in enumerate(all_samples):
                    _log_raw_response(raw, f'LOCAL_LLM_{idx+1}')

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                # filter invalid bodies
                filtered = []
                for b in all_samples:
                    if _is_valid_generated_body(b):
                        filtered.append(b)
                    else:
                        print('[WARN] Invalid generated body from LOCAL server; dropped one sample.')
                all_samples = filtered
                 
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        # Use original prompt without additional instructions
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com", timeout=30)
                    payload = json.dumps({
                        "max_tokens": 512,
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "frequency_penalty": 0.2,
                        "presence_penalty": 0.1,
                        # Limit stop sequences to at most 4 and avoid truncating code with 'def ' or '\n\n'
                        "stop": ["Okay", "I need", "Let me", "Looking"],
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    # "You are a Python code generator for equation synthesis.\n"
                                    # "Return only the function body for the function annotated with @equation.evolve in the prompt below.\n"
                                    # "Requirements:\n"
                                    # "- Output only Python code lines, no def line, no decorators, no comments, no markdown.\n"
                                    # "- Each line must be indented with 4 spaces to fit under the existing def line.\n"
                                    # "- Use only variables from the function signature (e.g., x, v, ..., params). Access parameters as params[i].\n"
                                    # "- Vectorized computation compatible with numpy/torch broadcasting; avoid explicit for loops where possible.\n"
                                    # "- Add a small eps to denominators to avoid division by zero; avoid invalid domains for log/exp/sqrt.\n"
                                    # "- Must contain at least one return that returns the predicted value with correct shape.\n"
                                    # "- Output code only. No explanations.\n"
                                )
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    raw_text = res.read().decode("utf-8", errors='ignore')
                    if res.status != 200:
                        print(f"[ERROR] API HTTP {res.status}: {raw_text[:200]}")
                        continue
                    data = json.loads(raw_text)
                    choices = data.get('choices') or []
                    if not choices:
                        print('[WARN] API returned no choices; retrying...')
                        continue
                    response = choices[0].get('message', {}).get('content', '')

                    # Print the raw output with escapes decoded
                    _log_raw_response(response, 'API')
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    if not _is_valid_generated_body(response):
                        print('[WARN] Invalid generated body from API; dropped one sample.')
                        continue
                     
                    all_samples.append(response)
                    break

                except Exception as e:
                    print(f"[ERROR] API sampling exception: {e}")
                    continue
        
        return all_samples
    

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        # Prepend the canonical instruction prefix
        full_prompt = (self._instruction_prompt + "\n\n" + content) if self._instruction_prompt else content

        # vLLM OpenAI-compatible endpoint (/v1/completions or /v1/chat/completions)
        if self._is_openai_compat_url(self._url):
            return self._do_request_openai(full_prompt, repeat_prompt, self._url)
        
        data = {
            'prompt': full_prompt,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': 1.0,  # Increase temperature for more randomness
                'top_k': 50,  # Increase top_k for more vocabulary choices
                'top_p': 0.9,  # Increase top_p for more diverse sampling
                'max_new_tokens': 1024,  # Set to 1024 per request
                'add_special_tokens': False,
                'skip_special_tokens': True,
                'repetition_penalty': 1.1,  # Lower repetition penalty for more variety
                # Remove 'def ' and '\n\n' to avoid truncating valid code
                'stop_sequences': ['Okay', 'I need', 'Let me', 'Looking', 'The function', 'Understanding', 'However', 'But', 'So', 'Now', 'First', 'Let\'s', 'Given', 'The problem'],
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self._url, data=json.dumps(data), headers=headers, timeout=30)
        except RequestException as e:
            raise RuntimeError(f"Local LLM request failed: {e}")

        # If we are talking to vLLM but the URL still points at /completions, fall back to /v1/completions
        if response.status_code == 404 and not self._is_openai_compat_url(self._url):
            fallback_url = self._build_openai_url(self._url)
            try:
                content_resp = self._do_request_openai(full_prompt, repeat_prompt, fallback_url)
                # Remember the working vLLM URL to avoid repeated 404s
                self._url = fallback_url
                return content_resp
            except Exception as e:
                raise RuntimeError(f"Local LLM HTTP 404; fallback to vLLM failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"Local LLM HTTP {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("Local LLM returned non-JSON response")
        if "content" not in payload:
            raise RuntimeError("Local LLM response missing 'content'")
        content_resp = payload["content"]
        return content_resp if self._batch_inference else content_resp[0]

    def _is_openai_compat_url(self, url: str) -> bool:
        return "/v1/" in url

    def _build_openai_url(self, url: str) -> str:
        base = url.rstrip("/")
        if base.endswith("/completions"):
            base = base[: -len("/completions")]
        return base + "/v1/completions"

    def _do_request_openai(self, full_prompt: str, repeat_prompt: int, url: str):
        headers = {'Content-Type': 'application/json'}

        is_chat = url.rstrip("/").endswith("/v1/chat/completions")
        if is_chat:
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.95,
                "n": repeat_prompt,
            }
        else:
            payload = {
                "prompt": full_prompt,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.95,
                "n": repeat_prompt,
            }

        if self._model_name:
            payload["model"] = self._model_name

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        except RequestException as e:
            raise RuntimeError(f"vLLM request failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("vLLM returned non-JSON response")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM response missing choices")

        if is_chat:
            contents = [c.get("message", {}).get("content", "") for c in choices]
        else:
            contents = [c.get("text", "") for c in choices]

        if not any(contents):
            raise RuntimeError("vLLM response contains empty content")

        return contents if repeat_prompt > 1 else contents[0]


# Create alias for backward compatibility
GRPOLocalLLM = LocalLLM

