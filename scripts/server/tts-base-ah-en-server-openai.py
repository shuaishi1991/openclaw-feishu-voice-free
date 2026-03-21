#!/usr/bin/env python3
"""
Qwen-TTS 服务端（嗯/啊 关键词插片版）- 兼容 OpenAI TTS API 接口
提供与 OpenAI /v1/audio/speech 兼容的接口，内部使用 Qwen3-TTS 模型；input 中与启动时
--keyword 匹配的片段会替换为预置音频并做 RMS 音量匹配，其余逻辑与 scripts/tools/tts-base-ah-en.py 一致。

Usage:
  python scripts/server/tts-base-ah-en-server-openai.py --port 8000 \\
    --clone /path/to/girl_1.pt \\
    [--keyword ... 可省略，使用下方默认 --keyword]

  默认与 --clone 一致，可显式传入 --keyword 覆盖。

API (兼容 OpenAI):
  POST /v1/audio/speech
  {
    "model": "gpt-4o-mini-tts",  // 可选，会被忽略（使用本地 Qwen3-TTS）
    "input": "Text to generate",
    "instructions": "自定义风格",  // 可选；省略或空字符串时使用服务端 --default-instructions（默认：口语化私人对话口吻）
    "response_format": "mp3",     // 可选，默认 mp3
    "speed": 1.0                  // 可选，暂不支持（Qwen3-TTS 不支持）
  }

Response:
  返回音频文件（MP3 格式），Content-Type: audio/mpeg
  响应头 X-Audio-Duration-Ms: 音频时长（毫秒）

健康检查:
  GET /v1/models
  GET /
"""

import argparse
import io
import json
import re
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# 进程内全局（启动时初始化）
model = None
ref_dict = None
device = None
keyword_info = []

# 请求体未提供 instructions 时的默认 instruct（可用 --default-instructions 覆盖）
DEFAULT_INSTRUCTIONS = "口语化私人对话口吻"

# 未传 --keyword 时的默认插片（KEYWORD, AUDIO_PATH, COUNT）；COUNT 为音频内分段槽位数
_DEFAULT_KEYWORD_SPECS = [
    (
        "啊...",
        "/root/.openclaw/workspace/skills/openclaw-feishu-voice-free/voice_embedings/ah-en/Rainnight_ah_12.mp3",
        "12",
    ),
    (
        "嗯...",
        "/root/.openclaw/workspace/skills/openclaw-feishu-voice-free/voice_embedings/ah-en/Rainnight_en_3.mp3",
        "3",
    ),
]


def load_keyword_info(keyword_args):
    """keyword_args: list of (keyword, audio_path, count_str)"""
    info = []
    for kw_data in keyword_args:
        keyword, audio_path, count_str = kw_data[0], kw_data[1], kw_data[2]
        try:
            count = int(count_str)
        except ValueError:
            print(f"❌ COUNT must be an integer for keyword '{keyword}', got '{count_str}'", file=sys.stderr)
            sys.exit(1)
        path = Path(audio_path)
        if not path.exists():
            print(f"❌ Audio file not found for keyword '{keyword}': {audio_path}", file=sys.stderr)
            sys.exit(1)
        try:
            full_audio = AudioSegment.from_file(path)
        except Exception as e:
            print(f"❌ Failed to load audio for keyword '{keyword}': {e}", file=sys.stderr)
            sys.exit(1)
        audio_len_ms = len(full_audio)
        len_per_keyword_ms = int(audio_len_ms / count)
        info.append({
            "keyword": keyword,
            "audio": full_audio,
            "count": count,
            "length_per_keyword_ms": len_per_keyword_ms,
            "audio_len_ms": audio_len_ms,
            "rms": full_audio.rms,
        })
        print(
            f"✓ Keyword '{keyword}': {audio_len_ms}ms, {count} slots, ~{len_per_keyword_ms / 1000:.2f}s/slot, RMS {full_audio.rms:.0f}",
            file=sys.stderr,
        )
    return sorted(info, key=lambda k: len(k["keyword"]), reverse=True)


def split_sequences(text, kws):
    pattern = "(" + "|".join(re.escape(k["keyword"]) for k in kws) + ")"
    parts = re.split(pattern, text)
    sequences = []
    kw_set = {k["keyword"] for k in kws}
    for part in parts:
        if not part:
            continue
        sequences.append((part in kw_set, part))
    return sequences


def synthesize_ah_en(text, language, instruct, response_format):
    """
    返回 (audio_bytes, duration_ms) 或抛出异常。
    text: 已清理的最终文稿
    """
    global model, ref_dict, keyword_info

    sequences = split_sequences(text, keyword_info)
    if not sequences:
        raise ValueError("No speakable content after split")

    prompt_item = VoiceClonePromptItem(
        ref_spk_embedding=ref_dict["ref_spk_embedding"],
        ref_text=ref_dict["ref_text"],
        ref_code=ref_dict["ref_code"],
        x_vector_only_mode=False,
        icl_mode=True,
    )

    cached_text_segments = []
    text_non_kw_count = 0
    avg_tts_rms = 0.0

    for is_kw, content in sequences:
        if is_kw:
            continue
        stripped = content.strip()
        if not stripped:
            cached_text_segments.append(None)
            continue
        print(f'[TTS ah-en] Generating: "{stripped[:80]}{"..." if len(stripped) > 80 else ""}"', file=sys.stderr)
        wavs, sr = model.generate_voice_clone(
            text=stripped,
            voice_clone_prompt=[prompt_item],
            language=language,
            instruct=instruct if instruct else None,
        )
        waveform = wavs[0]
        tts_audio = AudioSegment(
            (waveform * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        cached_text_segments.append((waveform, sr, tts_audio.rms))
        avg_tts_rms += tts_audio.rms
        text_non_kw_count += 1

    if text_non_kw_count > 0:
        avg_tts_rms = avg_tts_rms / text_non_kw_count
        print(f"[TTS ah-en] Average TTS RMS: {avg_tts_rms:.1f}, keyword gain matched", file=sys.stderr)
    else:
        avg_tts_rms = None
        print("[TTS ah-en] No text segments, keywords only", file=sys.stderr)

    final_audio = AudioSegment.empty()
    current_keyword_pos = {k["keyword"]: 0 for k in keyword_info}
    text_cache_idx = 0

    for is_kw, content in sequences:
        if is_kw:
            info = next(k for k in keyword_info if k["keyword"] == content)
            needed_ms = info["length_per_keyword_ms"]
            current_pos = current_keyword_pos[content]
            kw_segment = info["audio"][current_pos : current_pos + needed_ms]
            if avg_tts_rms is not None and info["rms"]:
                ratio = avg_tts_rms / info["rms"]
                ratio = max(0.5, min(ratio, 3.0))
                kw_segment = kw_segment.apply_gain(20 * np.log10(ratio))
            final_audio += kw_segment
            current_keyword_pos[content] += needed_ms
            current_keyword_pos[content] %= info["audio_len_ms"]
        else:
            cached = cached_text_segments[text_cache_idx]
            text_cache_idx += 1
            if cached is None:
                continue
            waveform, sr, _ = cached
            tts_segment = AudioSegment(
                (waveform * 32767).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            final_audio += tts_segment

    return export_audiosegment(final_audio, response_format)


def export_audiosegment(segment, response_format):
    """经临时 WAV 再转目标格式；返回 (bytes, duration_ms)。"""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    try:
        segment.export(temp_wav.name, format="wav")
        audio = AudioSegment.from_wav(temp_wav.name)
        duration_ms = len(audio)
        fmt = response_format if response_format in ("mp3", "wav", "opus", "aac", "flac") else "mp3"
        out = io.BytesIO()
        if fmt == "mp3":
            audio.export(out, format="mp3")
        elif fmt == "wav":
            audio.export(out, format="wav")
        elif fmt == "opus":
            audio.export(out, format="opus")
        elif fmt == "aac":
            audio.export(out, format="aac")
        elif fmt == "flac":
            audio.export(out, format="flac")
        else:
            audio.export(out, format="mp3")
        return out.getvalue(), duration_ms
    finally:
        try:
            Path(temp_wav.name).unlink()
        except OSError:
            pass


def get_content_type(fmt):
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }.get(fmt, "audio/mpeg")


def clean_text(text):
    cleaned = re.sub(r"\([^)]*\)", "", text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"（[^）]*）", "", cleaned)
    cleaned = re.sub(r"【[^】]*】", "", cleaned)
    cleaned = re.sub(r"［[^］]*］", "", cleaned)
    emoji_pattern = re.compile(
        "["
        "\u2639-\u263a"
        "\U0001F600-\U0001F64F"
        "\U0001F910-\U0001F92F"
        "\U0001F970-\U0001F97A"
        "\U0000FE0F"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", cleaned)
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text


class AhEnOpenAITTSHandler(BaseHTTPRequestHandler):
    default_instructions = DEFAULT_INSTRUCTIONS

    def detect_language(self, text):
        """简单的中英文检测"""
        # 如果包含中文字符，返回 Chinese
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return "Chinese"
        # 默认返回 English（可以根据需要扩展）
        return "English"

    def log_message(self, format, *args):
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def handle_one_request(self):
        try:
            return super().handle_one_request()
        except BrokenPipeError:
            print("[TTS ah-en] Client disconnected (BrokenPipeError)", file=sys.stderr)
        except OSError as e:
            if getattr(e, "errno", None) == 32:
                print("[TTS ah-en] Client disconnected (broken pipe)", file=sys.stderr)
            else:
                print(f"[TTS ah-en] Connection error: {e}", file=sys.stderr)
        except Exception:
            import traceback
            traceback.print_exc()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/audio/speech":
            self.handle_speech_request()
        else:
            self.send_error(404, f"Not found: {path}")

    def handle_speech_request(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "Missing request body")
            return
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
        except Exception as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        text = data.get("input")
        if not text:
            self.send_error(400, "Missing required field: input")
            return

        original_text = text
        text = clean_text(text)
        if text != original_text:
            print(f'[TTS ah-en] Cleaned: "{original_text[:60]}..." → "{text[:60]}..."', file=sys.stderr)
        if not text:
            text = original_text

        instructions = data.get("instructions")
        if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
            instructions = self.default_instructions
        response_format = data.get("response_format", "mp3")
        language = self.detect_language(text)

        try:
            start = time.time()
            audio_data, duration_ms = synthesize_ah_en(text, language, instructions, response_format)
            elapsed = time.time() - start
            print(f"[TTS ah-en] Done in {elapsed:.2f}s", file=sys.stderr)
            try:
                self.send_response(200)
                self.send_header("Content-Type", get_content_type(response_format))
                self.send_header("X-Audio-Duration-Ms", str(duration_ms))
                self.send_header("Content-Length", str(len(audio_data)))
                self.end_headers()
                self.wfile.write(audio_data)
                self.wfile.flush()
            except BrokenPipeError:
                print("[TTS ah-en] Client disconnected before response", file=sys.stderr)
            except OSError as e:
                if getattr(e, "errno", None) == 32:
                    print("[TTS ah-en] Client disconnected before response", file=sys.stderr)
                else:
                    raise
        except BrokenPipeError:
            print("[TTS ah-en] Client disconnected during generation", file=sys.stderr)
        except OSError as e:
            if getattr(e, "errno", None) == 32:
                print("[TTS ah-en] Client disconnected during generation", file=sys.stderr)
            else:
                raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                err = json.dumps({"error": {"message": str(e), "type": "server_error", "code": "tts_generation_failed"}})
                self.send_error(500, err)
            except (BrokenPipeError, OSError):
                print("[TTS ah-en] Cannot send error: client gone", file=sys.stderr)

    def do_GET(self):
        path = urlparse(self.path).path
        global model, device
        if path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "gpt-4o-mini-tts",
                                "object": "model",
                                "created": 1234567890,
                                "owned_by": "openclaw-qwen-tts-ah-en",
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        elif path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "ok",
                        "model_loaded": model is not None,
                        "device": device,
                        "api": "openai-compatible-ah-en",
                        "keywords": [k["keyword"] for k in keyword_info],
                        "endpoint": "/v1/audio/speech",
                    }
                ).encode("utf-8")
            )
        else:
            self.send_error(404, f"Not found: {path}")


def main():
    global model, ref_dict, device, keyword_info

    parser = argparse.ArgumentParser(
        description="Qwen-TTS ah/en keyword server — OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default 8000)")
    parser.add_argument(
        "--model",
        default="/root/.openclaw/models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model path",
    )
    parser.add_argument(
        "--clone",
        default="/root/.openclaw/workspace/skills/openclaw-feishu-voice-free/voice_embedings/huopo_kexin.pt",
        help="Cloned voice .pt",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        action="append",
        nargs=3,
        metavar=("KEYWORD", "AUDIO", "COUNT"),
        help='Repeatable，省略时使用内置默认（Rainnight ah/en 插片）；格式同 tts-base-ah-en.py',
    )
    parser.add_argument(
        "--default-instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="当请求 JSON 未提供或为空字符串 instructions 时，作为 instruct 传入模型（默认：口语化私人对话口吻）",
    )
    args = parser.parse_args()

    AhEnOpenAITTSHandler.default_instructions = args.default_instructions

    keyword_specs = args.keyword if args.keyword else _DEFAULT_KEYWORD_SPECS
    if not args.keyword:
        print("ℹ️  未提供 --keyword，使用默认 Rainnight ah/en 插片", file=sys.stderr)

    keyword_info = load_keyword_info(keyword_specs)

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("⚠️  flash_attn not installed, using eager attention", file=sys.stderr)
            attn_impl = "eager"
        if device == "cpu":
            attn_impl = "eager"

        print(f"📦 Loading Qwen3-TTS from {args.model}...", file=sys.stderr)
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print(f"✓ Model on {device} ({attn_impl})", file=sys.stderr)
    except Exception as e:
        print(f"❌ Model load failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    clone_path = Path(args.clone)
    if not clone_path.is_absolute():
        skill_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [
            skill_dir / clone_path,
            skill_dir / "voice_embedings" / clone_path,
            skill_dir / "voice_embedings" / clone_path.name,
        ]
        clone_resolved = next((p for p in candidates if p.exists()), clone_path)
    else:
        clone_resolved = clone_path

    if not clone_resolved.exists():
        print(f"❌ Clone not found: {args.clone}", file=sys.stderr)
        sys.exit(1)

    ref_dict = torch.load(clone_resolved, map_location=device)
    print(f"✓ Clone: {clone_resolved}", file=sys.stderr)

    server = HTTPServer(("0.0.0.0", args.port), AhEnOpenAITTSHandler)
    print(f"🚀 ah-en OpenAI TTS on http://0.0.0.0:{args.port}", file=sys.stderr)
    print("   POST /v1/audio/speech", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
