# openWakeWord Trainer

An all-in-one Docker container for training custom [openWakeWord](https://github.com/dscripka/openWakeWord) models, with a web UI. Type a wake word, record your own voice in the browser, click train, test it live with your microphone, and download a `.tflite` file for Home Assistant.

Nothing external is needed: [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) runs inside the container to speak your wake word in thousands of blended voices, and the training pipeline, feature extraction, TFLite conversion and web UI are all included. Built for Unraid (no Docker Compose), works anywhere Docker does.

## Features

- **Guided web UI**: pronunciation check, recording, training with live progress, live testing, downloads.
- **Kokoro built in**: 54 voices mixed into random blends, varied speed and punctuation, American and British pronunciations. Custom phoneme overrides for names Kokoro reads wrong.
- **Record your voice in the browser**: hands-free mode records 10 clips in a row and stops each one when you stop talking. Also records "other speech" and room noise, or upload audio files.
- **Tuning loop**: the live test shows the score in real time. When it misses you or triggers falsely, one click saves that audio into your recordings, and retraining reuses all the cached voices so it's quick.
- **Hard negatives generated automatically**: phonetically similar phrases ("hey marcus", "bay jarvis", "jarvis" on its own) from the CMU pronouncing dictionary, plus your own list.
- **Realistic augmentation**: 270 real room impulse responses, 2.5 hours of AudioSet background audio, EQ, pitch shifts, noise, level changes, and your own background recordings.
- **Honest evaluation**: false activations per hour on openWakeWord's 11 hour validation set, recall on held-out voices and held-out recordings of you, a per-clip score list, and a recommended threshold.
- **Exports** `.tflite` (Home Assistant openWakeWord add-on) and `.onnx`, both verified numerically against the trained model.
- **NVIDIA GPU** for Kokoro, feature extraction and training, with automatic CPU fallback.

## Install on Unraid

1. *(Optional, recommended)* Install the **Nvidia Driver** plugin from Community Applications and reboot if it asks.
2. Open the Unraid terminal and fetch the template:
   ```bash
   wget -O /boot/config/plugins/dockerMan/templates-user/my-openwakeword-trainer.xml https://raw.githubusercontent.com/adman234/openwakeword-trainer/main/unraid/openwakeword-trainer.xml
   ```
3. Go to **Docker › Add Container**, pick **openwakeword-trainer** from the *Template* dropdown, check the paths and click **Apply**.
   - No GPU? Remove `--runtime=nvidia` from **Extra Parameters** (switch to *Advanced View*) before applying.
4. Click the container icon › **WebUI**, or browse to `https://YOUR-SERVER-IP:8443`. Accept the self-signed certificate warning once. HTTPS is required because browsers only allow the microphone on secure pages.

<details>
<summary>Adding it by hand instead of with the template</summary>

| Field | Value |
| --- | --- |
| Repository | `ghcr.io/adman234/openwakeword-trainer:latest` |
| Network | `bridge` |
| Extra Parameters | `--runtime=nvidia --restart=unless-stopped` (drop `--runtime=nvidia` for CPU only) |
| Port | container `8443` (HTTPS web UI, needed for the microphone) |
| Port | container `8080` (plain HTTP, for a reverse proxy) |
| Path | container `/data` → `/mnt/user/appdata/openwakeword-trainer` |
| Path | container `/output` → `/mnt/user/appdata/openwakeword-trainer/output` |
| Variable | `NVIDIA_VISIBLE_DEVICES` = `all` |
| Variable | `PUID` = `99`, `PGID` = `100` |

</details>

## Run with plain Docker

```bash
docker run -d --name openwakeword-trainer --gpus all --restart unless-stopped \
  -p 8443:8443 -p 8080:8080 \
  -v /srv/openwakeword-trainer:/data \
  -v /srv/openwakeword-trainer/output:/output \
  ghcr.io/adman234/openwakeword-trainer:latest
```

Leave out `--gpus all` to run on the CPU.

## Training a wake word

1. **New wake word**: type the phrase, press *Hear it*, pick a quality preset.
2. **Pronunciation**: check how Kokoro says it with different voices. If it's wrong, edit the phonemes and replay until it sounds right. Add phrases that should *not* wake it (names in your house, catchphrases from shows you watch).
3. **Record** *(optional, but it helps a lot)*: 20 to 50 clips of the wake word from the people who will use it, at different distances and speeds. A few clips of other speech and 30 seconds of normal room noise help too.
4. **Train**: one click. The first run downloads about 19 GB of shared training data, which every later wake word reuses.
5. **Test**: press *Start listening* and talk. Adjust the threshold, and use *It missed me* or *False trigger* to save problem audio. Train again to fix them.
6. **Download** the `.tflite` and use it in Home Assistant:
   1. Copy it into `/share/openwakeword` on Home Assistant (Samba share or File editor add-on). Create the folder if it doesn't exist.
   2. Restart the **openWakeWord** add-on.
   3. **Settings › Voice assistants**, open your assistant and choose the new wake word.

Finished models are also copied to `/output` automatically. ESPHome devices that run wake word detection on the device itself use microWakeWord, which is a different model format.

### Presets

| Preset | Generated clips | Steps | Time with an RTX 4070 class GPU |
| --- | --- | --- | --- |
| Quick test | 2,000 + 2,000 | 8,000 | about 10 minutes |
| Standard | 10,000 + 10,000 | 25,000 | about 35 minutes |
| Maximum | 25,000 + 25,000 | 50,000 | about 1.5 hours |

Most of the time goes into Kokoro generating clips, and those are cached: retraining after adding recordings or changing training settings skips straight to the fast parts. On CPU only, expect several times longer (Kokoro is around 0.2 s per clip on a modern desktop CPU).

Everything is adjustable under *Advanced settings*: clip counts, steps, model size, target false activations per hour, negative weighting, how much weight your recordings get, and which voices are used.

## How it works

1. **Voices**: Kokoro speaks the phrase with a random blend of one to three of its 54 voices, a random speed and punctuation, then trims the silence. American and British G2P are used to match the dominant voice.
2. **Hard negatives**: similar-sounding phrases are built by swapping words for ones within one or two phoneme edits (CMU dictionary), plus partial phrases, your custom phrases and everyday sentences. Kokoro speaks those too.
3. **Augmentation**: each clip is placed so the wake word ends up to 200 ms before the end of a 2 second window (matching how streaming detection sees it), then gets reverb from real rooms, background audio at -5 to 15 dB SNR, EQ, pitch shift, coloured noise, band-stop filtering and a random level. Your recordings are augmented many times over.
4. **Features**: openWakeWord's melspectrogram and Google speech embedding models turn each clip into 16 × 96 features.
5. **Training**: a small fully connected network, trained openWakeWord style against 2,000 hours of pre-computed ACAV100M negative speech features, with hard example mining, a ramping negative class weight, three learning rate phases and checkpoint averaging.
6. **Evaluation and export**: false activations per hour on 11 hours of held-out audio, recall on held-out voices and recordings, a threshold table, then ONNX and TFLite export with numerical checks.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HTTPS_PORT` | `8443` | HTTPS web UI (self-signed certificate) |
| `HTTP_PORT` | `8080` | Plain HTTP web UI, for reverse proxies |
| `PUID` / `PGID` | unset (`99` / `100` in the Unraid template) | Run as this user so files stay manageable |
| `CERT_HOSTS` | empty | Extra IPs or hostnames for the self-signed certificate, comma separated |
| `HF_TOKEN` | empty | Optional Hugging Face token if downloads get rate limited |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPU to use |
| `PREVIEW_DEVICE` | `cpu` | Device for pronunciation previews in the web UI (`cuda` to use the GPU) |

### What's in `/data`

```
/data
├── projects/<wake_word>/
│   ├── project.json
│   ├── recordings/{positive,negative,background}/   your clips
│   ├── work/                                         generated clips and cached features (safe to delete)
│   └── runs/<timestamp>/                             .tflite, .onnx and report.json
├── downloads/        shared training data (about 19 GB)
├── background/       drop your own background audio here to use it for every wake word
└── certs/            self-signed HTTPS certificate (delete to regenerate)
```

## GPUs

The image uses PyTorch 2.8 with CUDA 12.8, which supports NVIDIA cards from Turing (RTX 20xx, GTX 16xx) through Blackwell (RTX 50xx). If the GPU can't be used, the training log says so and everything falls back to the CPU. Your Unraid NVIDIA driver needs to be version 570 or newer for CUDA 12.8.

For older Pascal cards (GTX 10xx, Quadro P series), build the image with CUDA 12.6:

```bash
docker build --build-arg TORCH_CUDA=cu126 -t openwakeword-trainer:cu126 .
```

## Troubleshooting

- **"Microphone unavailable"**: open the HTTPS address (port 8443), not HTTP. If you use a reverse proxy, it needs a valid certificate and WebSocket support for the live test.
- **No GPU detected**: check the Nvidia Driver plugin shows your card, that `--runtime=nvidia` is in Extra Parameters, and that `NVIDIA_VISIBLE_DEVICES` is set.
- **Too many false activations**: use the live test's *False trigger* button a few times, add the phrases that triggered it under Pronunciation, retrain, or raise the threshold. The *Maximum* preset also helps.
- **Misses your voice**: record more clips of the people who will use it, at the distance they'll speak from, and retrain. Lowering the threshold also helps.
- **Download stalls or fails**: it resumes where it left off. Press *Download everything now* on the Setup page again, or set `HF_TOKEN`.

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka: model architecture, feature models, training recipe and the pre-computed ACAV100M and validation features.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, with the [misaki](https://github.com/hexgrad/misaki) G2P.
- [MIT environmental impulse responses](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) and [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) for augmentation.

The downloaded datasets have their own licenses; check them before any commercial use of trained models.
