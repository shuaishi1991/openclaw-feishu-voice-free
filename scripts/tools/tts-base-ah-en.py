#!/usr/bin/env python3
"""
分段调用版本：每个文本段单独调用TTS生成，然后插入关键词音频，每个片段音量自动匹配

Usage:
  python tts-base-ah-en.py \
    --prompt "嗯...那我跟着哥哥一起好不好...啊...好深呀...嗯..." \
    --keyword "啊..." /path/to/ahs.mp3 15 \
    --keyword "嗯..." /path/to/ens.mp3 3 \
    --clone /path/to/girl_1.pt \
    --model /path/to/Qwen3-TTS-12Hz-1.7B-Base \
    -l Chinese \
    -i "high moaning" \
    -o output.wav
"""

import argparse
import sys
import os
import re
from pathlib import Path
import torch
import soundfile as sf
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel
from qwen_tts import VoiceClonePromptItem
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Qwen-TTS + keyword audio replacement: split segments, generate once, adjust keyword volume, concat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", dest="prompt", required=True, help="Full text with keywords to replace")
    parser.add_argument("-k", "--keyword", action="append", nargs=3, metavar=("KEYWORD", "AUDIO", "COUNT"),
                            help="Keyword phrase, full audio file, number of keywords in this audio")
    parser.add_argument("-o", "--output", required=True, help="Output audio path (auto convert to mp3 if suffix is .mp3)")
    parser.add_argument("--mp3", action="store_true", help="Force convert output to MP3 (default: auto-detect by output suffix)")
    parser.add_argument("--clone", help="Cloned voice embedding (.pt) from Qwen-TTS (required)", default=None)
    parser.add_argument("--model", default="/root/.openclaw/models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base", help="Model path or name")
    parser.add_argument("-l", "--language", default="Chinese", help="Language (default: Chinese)")
    parser.add_argument("-i", "--instruct", default="excited moaning", help="Voice instruction (default: excited moaning)")
    parser.add_argument("--ms-per-char", type=float, default=50, help="Average milliseconds per character doesn't matter anymore, doesn't used")
    parser.add_argument("--device", help="Device (default: auto detect cuda if available)")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Using device: {device}", file=sys.stderr)

    # Check keywords and load audio
    keyword_info = []
    if not args.keyword or len(args.keyword) == 0:
        print(f"❌ Please provide at least one keyword with --keyword KEYWORD AUDIO COUNT", file=sys.stderr)
        sys.exit(1)
    for kw_data in args.keyword:
        keyword = kw_data[0]
        audio_path = kw_data[1]
        try:
            count = int(kw_data[2])
        except ValueError:
            print(f"❌ COUNT must be an integer for keyword '{keyword}', got '{kw_data[2]}'", file=sys.stderr)
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
        keyword_info.append({
            "keyword": keyword,
            "audio": full_audio,
            "count": count,
            "length_per_keyword_ms": len_per_keyword_ms,
            "audio_len_ms": audio_len_ms,
            "rms": full_audio.rms,  # for volume matching later
        })
        print(f"✓ Loaded keyword '{keyword}': {audio_len_ms}ms total, {count} keywords, ~{len_per_keyword_ms/1000:.2f}s per keyword, RMS: {full_audio.rms:.0f}", file=sys.stderr)

    # Sort keywords by length descending - longer match first
    keyword_info = sorted(keyword_info, key=lambda k: len(k["keyword"]), reverse=True)

    # Split text into (is_keyword, content) sequences
    sequences = []
    remaining = args.prompt
    pattern = '(' + '|'.join(re.escape(k["keyword"]) for k in keyword_info) + ')'
    parts = re.split(pattern, remaining)

    keyword_count = 0
    text_segment_count = 0
    for part in parts:
        if not part:
            continue
        is_kw = part in [k["keyword"] for k in keyword_info]
        sequences.append( (is_kw, part) )
        if is_kw:
            keyword_count += 1
        else:
            text_segment_count += 1

    print(f"ℹ️ Found {keyword_count} keywords, {text_segment_count} text segments", file=sys.stderr)

    # Load model
    try:
        print(f"🔄 Loading TTS model: {args.model}", file=sys.stderr)
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print(f"✓ Model loaded on {device}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load cloned embedding - exactly same as tts-base.py
    ref_dict = None
    if args.clone:
        clone_path = Path(args.clone)
        if not clone_path.exists():
            print(f"❌ Cloned embedding not found: {args.clone}", file=sys.stderr)
            sys.exit(1)
        ref_dict = torch.load(args.clone, map_location=device)
        print(f"✓ Loaded cloned voice ref_dict: {args.clone}", file=sys.stderr)

    if ref_dict is not None:
        prompt_item = VoiceClonePromptItem(
                ref_spk_embedding=ref_dict["ref_spk_embedding"],
                ref_text=ref_dict["ref_text"],
                ref_code=ref_dict["ref_code"],
                x_vector_only_mode=False,
                icl_mode=True
            )

    # First pass: generate all text segments once, cache them
    print(f"🔄 First pass: generating all text segments...", file=sys.stderr)
    cached_text_segments = []  # (waveform, sample_rate, rms) or None for keyword
    
    text_cache_idx = 0
    avg_tts_rms = 0
    for (is_kw, content) in sequences:
        if is_kw:
            # cached_text_segments.append(None)
            continue
        stripped = content.strip()
        if not stripped:
            # cached_text_segments.append(None)
            continue
        print(f"🔊 Generating TTS: '{stripped}'", file=sys.stderr)

        # Generate - exactly same clone mode as tts-base.py
        if ref_dict is not None:
            wavs, sr = model.generate_voice_clone(
                text=stripped,
                voice_clone_prompt=[prompt_item],
                language=args.language,
                instruct=args.instruct if args.instruct else None,
                # icl_mode=False
            )

            waveform = wavs[0]

            tts_audio = AudioSegment(
                (waveform * 32767).astype(np.int16).tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )

            cached_text_segments.append( (waveform, sr, tts_audio.rms) )
            text_cache_idx += 1
            avg_tts_rms += tts_audio.rms
    
    avg_tts_rms = avg_tts_rms / text_cache_idx
    
    if text_cache_idx > 0:
        print(f"ℹ️ Average TTS RMS: {avg_tts_rms}, adjusting keywords to match...", file=sys.stderr)
    else:
        print(f"ℹ️ No text segments, just assembling keywords", file=sys.stderr)

    # Second pass: assemble final audio with adjusted keyword volume
    print(f"🔄 Second pass: assembling final audio with volume adjusted...", file=sys.stderr)
    final_audio = AudioSegment.empty()
    current_keyword_pos = {k["keyword"]: 0 for k in keyword_info}
    text_cache_idx = 0
    for (is_kw, content) in sequences:
        if is_kw:
            # Insert keyword with adjusted volume
            info = next(k for k in keyword_info if k["keyword"] == content)
            needed_ms = info["length_per_keyword_ms"]
            current_pos = current_keyword_pos[content]
            kw_segment = info["audio"][current_pos : current_pos + needed_ms]
            if avg_tts_rms is not None:
                kw_rms = info["rms"]
                ratio = avg_tts_rms / kw_rms
                ratio = max(0.5, min(ratio, 3.0)) # don't adjust too extreme
                kw_segment = kw_segment.apply_gain(20 * np.log10(ratio))
            final_audio += kw_segment
            current_keyword_pos[content] += needed_ms
            current_keyword_pos[content] %= info["audio_len_ms"]
        else:
            # Add cached text segment
            cached = cached_text_segments[text_cache_idx]
            waveform, sr, _ = cached
            tts_segment = AudioSegment(
                (waveform * 32767).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1
            )
            final_audio += tts_segment
            text_cache_idx += 1

    # Export final audio - auto convert to MP3 if needed
    final_path = Path(args.output).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    final_audio.export(
        final_path,
        format="mp3",  # 压缩格式，可选：mp3/aac/ogg等
        bitrate="128k"  # 语音场景推荐32k，足够清晰且体积小
    )

    # need_mp3 = args.mp3 or (final_path.suffix.lower() == '.mp3')
    # if need_mp3:
    #     # Export temp wav then convert
    #     temp_wav = final_path.with_suffix('.tmp.wav')
    #     final_audio.export(temp_wav, format="wav")
    #     import subprocess
    #     try:
    #         print(f"🔄 Converting to MP3: {final_path}", file=sys.stderr)
    #         subprocess.run([
    #             'ffmpeg', '-y',
    #             '-i', str(temp_wav),
    #             '-codec:a', 'libmp3lame',
    #             '-V', '2',
    #             str(final_path),
    #         ], check=True, capture_output=True)
    #         temp_wav.unlink()
    #         print(f"✅ MP3 saved: {final_path}", file=sys.stderr)
    #     except subprocess.CalledProcessError as e:
    #         print(f"⚠️  ffmpeg failed, keeping original wav: {temp_wav}", file=sys.stderr)
    #         final_path = temp_wav
    #     print(f"✅ Final audio saved: {final_path}", file=sys.stderr)
    # else:
    #     final_audio.export(final_path, format="wav")
    #     print(f"✅ WAV saved: {final_path}", file=sys.stderr)

    # # Print final path for integration
    # print(str(final_path), file=sys.stderr)

    print(f"✅ Cloned audio saved: {final_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
