#!/usr/bin/env python3
"""
Qwen-TTS 服务端 - 兼容 OpenAI TTS API 接口
提供与 OpenAI /v1/audio/speech 兼容的接口，内部使用 Qwen3-TTS 模型

Usage:
  python scripts/tts-base-server-openai.py --port 8000 [--model /path/to/model] [--clone /path/to/clone.pt] \\
    [--default-instructions TEXT]

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
  响应头 X-Audio-Duration-Ms: 音频时长（毫秒），供上游（如飞书上传）填写 duration 使用

健康检查:
  GET /v1/models
  GET /
"""

import argparse
import sys
import json
import tempfile
from pathlib import Path
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from qwen_tts import Qwen3TTSModel
from qwen_tts import VoiceClonePromptItem
import soundfile as sf
from pydub import AudioSegment
import time
import io

# 全局变量
model = None
ref_dict = None
device = None

# 请求体未提供 instructions 时的默认 instruct（可用 --default-instructions 覆盖）
DEFAULT_INSTRUCTIONS = "口语化私人对话口吻"


class OpenAICompatibleTTSHandler(BaseHTTPRequestHandler):
    """兼容 OpenAI TTS API 的请求处理器"""

    default_instructions = DEFAULT_INSTRUCTIONS

    def log_message(self, format, *args):
        """重写日志方法，使用 stderr"""
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def handle_one_request(self):
        """重写以捕获所有异常，避免 BrokenPipeError 导致服务器崩溃"""
        try:
            return super().handle_one_request()
        except BrokenPipeError:
            # 客户端断开连接，这是正常情况（可能是超时）
            print(f'[TTS] Client disconnected (BrokenPipeError)', file=sys.stderr)
        except OSError as e:
            # 网络错误，可能是客户端断开
            if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                print(f'[TTS] Client disconnected (OSError: Broken pipe)', file=sys.stderr)
            else:
                print(f'[TTS] Connection error: {e}', file=sys.stderr)
        except Exception as e:
            # 其他异常，记录完整信息
            import traceback
            traceback.print_exc()

    def do_POST(self):
        """处理 POST 请求"""
        path = urlparse(self.path).path

        if path == "/v1/audio/speech":
            self.handle_speech_request()
        else:
            self.send_error(404, f"Not found: {path}")

    def handle_speech_request(self):
        """处理 /v1/audio/speech 请求"""
        global model, ref_dict

        # 读取请求体
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_error(400, "Missing request body")
            return

        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
        except Exception as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        # 解析 OpenAI 格式的参数
        text = data.get('input')
        if not text:
            self.send_error(400, "Missing required field: input")
            return

        # 清理文本：去掉括号内的内容（内心描述、场景描述等）
        original_text = text
        text = self.clean_text(text)
        if text != original_text:
            print(f'[TTS] Cleaned text: "{original_text[:60]}..." → "{text[:60]}..."', file=sys.stderr)

        # 如果清理后文本为空，使用原文本
        if not text:
            print(f'[TTS] Warning: Text is empty after cleaning, using original text', file=sys.stderr)
            text = original_text

        # 可选参数（缺省或空串用服务端 default_instructions）
        instructions = data.get("instructions")
        if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
            instructions = self.default_instructions
        response_format = data.get('response_format', 'mp3')
        speed = data.get('speed', 1.0)  # 暂不支持，保留用于未来扩展

        # 语言检测（简单启发式，可以根据需要改进）
        language = self.detect_language(text)

        try:
            start = time.time()

            # 使用预加载的模型和音色
            prompt_item = VoiceClonePromptItem(
                ref_spk_embedding=ref_dict["ref_spk_embedding"],
                ref_text=ref_dict["ref_text"],
                ref_code=ref_dict["ref_code"],
                x_vector_only_mode=False,
                icl_mode=True
            )

            # 生成语音
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=[prompt_item],
                language=language,
                instruct=instructions if instructions else None,
            )

            end = time.time()
            print(f'[TTS] Generated in {end - start:.2f}s: "{text[:50]}{"..." if len(text) > 50 else ""}"', file=sys.stderr)

            # 转换为请求的格式
            audio_data, duration_ms = self.convert_audio_format(wavs[0], sr, response_format)

            # 发送响应（捕获 BrokenPipeError，客户端可能已断开）
            try:
                self.send_response(200)
                self.send_header('Content-Type', self.get_content_type(response_format))
                self.send_header('X-Audio-Duration-Ms', str(duration_ms))
                self.send_header('Content-Length', str(len(audio_data)))
                self.end_headers()
                self.wfile.write(audio_data)
                self.wfile.flush()
            except BrokenPipeError:
                # 客户端已断开连接，这是正常的（可能是超时），不需要报错
                print(f'[TTS] Client disconnected before response sent (likely timeout)', file=sys.stderr)
                return
            except OSError as e:
                # 其他网络错误
                if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                    print(f'[TTS] Client disconnected before response sent', file=sys.stderr)
                    return
                raise

        except BrokenPipeError:
            # 在生成过程中客户端断开，正常情况
            print(f'[TTS] Client disconnected during generation (likely timeout)', file=sys.stderr)
            return
        except OSError as e:
            # 网络错误，可能是客户端断开
            if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                print(f'[TTS] Client disconnected (likely timeout)', file=sys.stderr)
                return
            # 其他 OSError，继续抛出
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 只有在连接仍然有效时才发送错误响应
            try:
                error_response = {
                    "error": {
                        "message": f"Generation failed: {e}",
                        "type": "server_error",
                        "code": "tts_generation_failed"
                    }
                }
                # 检查是否已经发送了响应头
                if not hasattr(self, '_headers_sent') or not self._headers_sent:
                    self.send_error(500, json.dumps(error_response))
            except (BrokenPipeError, OSError):
                # 客户端已断开，无法发送错误响应
                print(f'[TTS] Cannot send error response: client disconnected', file=sys.stderr)
            return

    def clean_text(self, text):
        """
        清理文本：去掉括号包裹的内心描述、场景描述和常用脸型 Emoji
        示例: "(看向窗外) 你好😊 (微笑)" → "你好"
        全角与中括号: "（微笑）【提示】［注］好的" → "好的"
        仅剥黄脸/经典表情等脸型码位；心形、动物、旗帜等非脸型符号不剥除。
        """
        import re
        # 去掉所有 (...) 圆括号内容
        cleaned = re.sub(r'\([^)]*\)', '', text)
        # 去掉所有 [...] 方括号内容
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        # 去掉所有 {...} 花括号内容
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
        # 去掉全角圆括号（…）内容
        cleaned = re.sub(r'（[^）]*）', '', cleaned)
        # 去掉中式方括号【…】内容
        cleaned = re.sub(r'【[^】]*】', '', cleaned)
        # 去掉全角方括号［…］内容
        cleaned = re.sub(r'［[^］]*］', '', cleaned)

        # 仅去掉常用脸型 Emoji（不误伤汉字；心形/物体/旗等不在此列）
        emoji_pattern = re.compile(
            "["
            "\u2639-\u263a"  # BMP 经典表情 ☹ ☺
            "\U0001F600-\U0001F64F"  # Emoji 表情主块
            "\U0001F910-\U0001F92F"  # 补充表情（捂脸、拉链嘴等）
            "\U0001F970-\U0001F97A"  # 常见新表情（如 🥰 🥺）
            "\U0000FE0F"  # 变体选择符-16（常与 Emoji 连用）
            "]+",
            flags=re.UNICODE
        )
        cleaned = emoji_pattern.sub('', cleaned)
        # Emoji 序列拆完后常见的零宽/控制残留（不误删 CJK 标点）
        cleaned = re.sub(r'[\u200B-\u200D\uFEFF]+', '', cleaned)

        # 去掉多余空格和换行
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else text

    def detect_language(self, text):
        """简单的中英文检测"""
        # 如果包含中文字符，返回 Chinese
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "Chinese"
        # 默认返回 English（可以根据需要扩展）
        return "English"

    def convert_audio_format(self, audio_data, sample_rate, format):
        """转换音频格式，返回 (bytes, duration_ms)。duration_ms 与 pydub 段长度一致。"""
        # 创建临时 WAV 文件
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()

        try:
            # 保存为 WAV
            sf.write(temp_wav.name, audio_data, sample_rate)

            # 根据格式转换
            audio = AudioSegment.from_wav(temp_wav.name)
            duration_ms = len(audio)

            if format == "mp3":
                output = io.BytesIO()
                audio.export(output, format="mp3")
                return output.getvalue(), duration_ms
            elif format == "wav":
                output = io.BytesIO()
                audio.export(output, format="wav")
                return output.getvalue(), duration_ms
            elif format == "opus":
                output = io.BytesIO()
                audio.export(output, format="opus")
                return output.getvalue(), duration_ms
            elif format == "aac":
                output = io.BytesIO()
                audio.export(output, format="aac")
                return output.getvalue(), duration_ms
            elif format == "flac":
                output = io.BytesIO()
                audio.export(output, format="flac")
                return output.getvalue(), duration_ms
            else:
                # 默认返回 MP3
                output = io.BytesIO()
                audio.export(output, format="mp3")
                return output.getvalue(), duration_ms
        finally:
            # 清理临时文件
            try:
                Path(temp_wav.name).unlink()
            except:
                pass

    def get_content_type(self, format):
        """获取 Content-Type"""
        content_types = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
        }
        return content_types.get(format, "audio/mpeg")

    def do_GET(self):
        """处理 GET 请求（健康检查和模型列表）"""
        path = urlparse(self.path).path

        if path == "/v1/models":
            # 返回模型列表（兼容 OpenAI API）
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "object": "list",
                "data": [
                    {
                        "id": "gpt-4o-mini-tts",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "openclaw-qwen-tts"
                    }
                ]
            })
            self.wfile.write(response.encode('utf-8'))
        elif path == "/":
            # 健康检查
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "ok",
                "model_loaded": model is not None,
                "device": device,
                "api": "openai-compatible",
                "endpoint": "/v1/audio/speech"
            })
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404, f"Not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-TTS server with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--model",
        default="/root/.openclaw/models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model path or name"
    )
    parser.add_argument(
        "--clone",
        default="/root/.openclaw/workspace/skills/openclaw-feishu-voice-free/voice_embedings/huopo_kexin.pt",
        help="Path to cloned voice embedding file (.pt format). If not specified, uses default voice."
    )
    parser.add_argument(
        "--default-instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="当请求 JSON 未提供或为空字符串 instructions 时，作为 instruct 传入模型（默认：口语化私人对话口吻）",
    )
    args = parser.parse_args()

    OpenAICompatibleTTSHandler.default_instructions = args.default_instructions

    global model, ref_dict, device

    # 加载模型
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        # Try Flash attention if available (CUDA only)
        attn_impl = "flash_attention_2"
        try:
            import flash_attn
        except ImportError:
            print(f"⚠️  flash_attn not installed, falling back to eager attention", file=sys.stderr)
            attn_impl = "eager"

        if device == "cpu":
            attn_impl = "eager"

        print(f"📦 Loading Qwen3-TTS model from {args.model}...", file=sys.stderr)
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print(f"✓ Model loaded on {device} with {attn_impl}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 加载音色克隆文件
    ref_dict = None
    if args.clone:
        clone_path = Path(args.clone)
        # 支持相对路径
        clone_path_resolved = None
        if not clone_path.is_absolute():
            skill_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                skill_dir / clone_path,
                skill_dir / "voice_embedings" / clone_path,
                skill_dir / "voice_embedings" / clone_path.name,
            ]
            for p in possible_paths:
                if p.exists():
                    clone_path_resolved = p
                    break
            if clone_path_resolved is None:
                clone_path_resolved = clone_path
        else:
            clone_path_resolved = clone_path

        if not clone_path_resolved.exists():
            print(f"❌ Cloned embedding not found: {args.clone}", file=sys.stderr)
            if not Path(args.clone).is_absolute():
                print(f"   Tried paths: {[str(p) for p in possible_paths]}", file=sys.stderr)
            sys.exit(1)

        ref_dict = torch.load(clone_path_resolved, map_location=device)
        print(f"✓ Loaded cloned voice: {clone_path_resolved}", file=sys.stderr)
    else:
        print(f"⚠️  No clone file specified, using default voice", file=sys.stderr)
        # 创建一个默认的 ref_dict（如果需要）
        # 这里可以根据实际情况调整

    # 启动服务器
    server = HTTPServer(('0.0.0.0', args.port), OpenAICompatibleTTSHandler)
    print(f"🚀 OpenAI-compatible TTS server running on http://0.0.0.0:{args.port}", file=sys.stderr)
    print(f"   API endpoint: POST /v1/audio/speech", file=sys.stderr)
    print(f"   Health check: GET /", file=sys.stderr)
    print(f"   Models list: GET /v1/models", file=sys.stderr)
    print(f"   Example request:", file=sys.stderr)
    print(f"""   curl -X POST http://localhost:{args.port}/v1/audio/speech \\
     -H "Content-Type: application/json" \\
     -d '{{"model": "gpt-4o-mini-tts", "input": "Hello, world!"}}' \\
     --output output.mp3""", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
