# TurboFieldfare local runner.
# Local addition, not upstream. `just` with no arguments lists everything.

model    := "scratch/gemma4-heretic.gturbo"
model-id := "gemma-4-26b-a4b-it-heretic"
base     := "scratch/gemma4.gturbo"
port     := "8080"
context  := "16384"
bin      := ".build/release"
# 32 over the default 16: measured +23% decode (median 13.6 -> 16.7 tok/s,
# interleaved 16/32/16/32, warmup discarded, non-overlapping ranges).
slots    := "32"

default:
    @just --list --unsorted

# Build every product in release mode.
build:
    swift build -c release

# Serve the abliterated model on an OpenAI-compatible endpoint.
serve: _solo
    {{bin}}/TurboFieldfareServer \
      --model {{model}} \
      --port {{port}} --max-context {{context}} \
      --model-id {{model-id}} --expert-cache-slots {{slots}}

# Serve the stock pinned model instead.
serve-base: _solo
    {{bin}}/TurboFieldfareServer \
      --model {{base}} \
      --port {{port}} --max-context {{context}} \
      --model-id gemma-4-26b-a4b-it --expert-cache-slots {{slots}}

# One-shot chat, templated (raw --prompt gives garbage on an instruct model).
ask prompt: _solo
    #!/usr/bin/env bash
    set -euo pipefail
    f=$(mktemp -t tf-msg).json
    trap 'rm -f "$f"' EXIT
    printf '[{"role":"user","content":%s}]' "$(jq -Rn --arg s {{quote(prompt)}} '$s')" > "$f"
    {{bin}}/TurboFieldfareCLI --model {{model}} --messages-file "$f" --max-new 512

# Same, against the stock model.
ask-base prompt: _solo
    #!/usr/bin/env bash
    set -euo pipefail
    f=$(mktemp -t tf-msg).json
    trap 'rm -f "$f"' EXIT
    printf '[{"role":"user","content":%s}]' "$(jq -Rn --arg s {{quote(prompt)}} '$s')" > "$f"
    {{bin}}/TurboFieldfareCLI --model {{base}} --messages-file "$f" --max-new 512

# Send a prompt to an already-running server.
curl prompt:
    @curl -s http://127.0.0.1:{{port}}/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "$(jq -n --arg m {{model-id}} --arg c {{quote(prompt)}} \
            '{model:$m, messages:[{role:"user",content:$c}], max_completion_tokens:512}')" \
      | jq -r '.choices[0].message.content // .'

# Is the server up, and what is it serving?
status:
    @curl -s --max-time 5 http://127.0.0.1:{{port}}/health || echo '{"status":"down"}'
    @curl -s --max-time 5 http://127.0.0.1:{{port}}/v1/models | jq -c '.data[0].id' 2>/dev/null || true

# Stop the server (and anything else holding the model).
stop:
    -@pkill -f TurboFieldfareServer && echo "stopped server" || echo "no server running"

# Verify an installed model against its receipt.
verify:
    {{bin}}/TurboFieldfareRepack --verify-install --input-gturbo {{model}}

test:
    Scripts/test.sh

# Fail early if something else already holds the model or the port. The
# runtime supports exactly one local-model process at a time.
_solo:
    #!/usr/bin/env bash
    if pgrep -f 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|swiftpm-testing-helper' >/dev/null; then
      echo "error: another TurboFieldfare process is running — 'just stop' first" >&2
      pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI' >&2
      exit 1
    fi
    if pgrep -x ollama >/dev/null; then
      echo "warning: ollama is running; it competes for memory and GPU" >&2
    fi
