import queue
import webrtcvad
import whisper
import sounddevice as sd
import numpy as np
import json

with open('data/intent_terms.json') as intent_data:
    intent = json.load(intent_data)
    intent_data.close()
    # Import library of intents, store in variable intent.

with open('data/objects_terms.json') as objects_data:
    objects = json.load(objects_data)
    objects_data.close()
    # Import library of objects, store in variable objects.

def normalize(text):
    return text.lower().strip()
    # Converts all processed text to lowercase.
    # Also removes whitespace around the text. So "   Drop    " becomes "drop"
    # Can be upgraded later using re package

def find_match(text, term_dict):
    text = normalize(text)
    for key, phrases in term_dict.items():
        for phrase in phrases:
            if phrase in text:
                return key
    return None
    # normalizes text using the above function
    # searches for the phrases that will activate the key
    # for instance, a person might say "pick up", and that phrase activates the key, "grab"
    # Also loops through and gauges if the phrase appears in the spoken text
    # If the phrase is recognized, the key is registered and pushed through
    # If no phrase is recognized, the function does nothing.


def announce_action(intent, obj):
    if intent and obj:
        print(f"ACTION: {intent.upper()} → {obj.upper()}")
    elif intent:
        print(f"INTENT DETECTED: {intent.upper()} (no object specified)")
    elif obj:
        print(f"OBJECT DETECTED: {obj.upper()} (no intent specified)")
    # Takes intents and objects from find_match function
    # Only announces registered commands from the JSON library
    # If both intent and objects are detected, the code outputs the action of the intent on the object
    # If only one is detected, it will specify which intent/object it heard.
    # THIS FUNCTION WILL BE REPLACED BY ROBOT MOVEMENT LATER

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000 # Audio sample rate (Whisper + WebRTC VAD both expect 16 kHz PCM audio)
FRAME_MS = 20                     # MUST be 10, 20, or 30
                                # Duration (in ms) of each audio frame processed by VAD
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) # Number of samples per frame

MIN_AUDIO_SECONDS = 0.8           # minimum speech before Whisper
SILENCE_FRAMES_END = 15           # ~300 ms silence, stops the listening
PADDING_SECONDS = 0.2               # Adds .2 seconds of silence for listening accuracy

DEVICE_INDEX = 1                 # Registers the default microphone from the PC, adjust if needed
# ---------------------------------------

print(sd.query_devices())
print("Using device:", DEVICE_INDEX)
    # Prints the audio devices available and which device will be used

vad = webrtcvad.Vad(2) # Initialize WebRTC Voice Activity Detection with medium aggressiveness (0–3) to detect speech in audio frames
model = whisper.load_model("small") # Loads OpenAI's Whisper "small" model.
# Transcribes speech to text. Small is useful for accuracy and speed for lower computing power, like a laptop.

speech_frames = [] # Buffer to store audio frames that contain speech.
silence_counter = 0 # Tracks consecutive silent frames which determines when speech has ended.
speech_queue = queue.Queue() # Queue for passing completed speech segments to another process/thread

# ---------------- AUDIO CALLBACK ----------------
def callback(indata, _frames, _time_info, status):
    global silence_counter

    if status:
        return # Exits function if there is an error.

    audio_float = indata[:, 0] # Extract mono audio

    # Convert to int16 PCM for VAD
    audio_int16 = (audio_float * 32768).astype(np.int16)

    # REQUIRED: exact frame size check
    if len(audio_int16) != FRAME_SIZE:
        return

    try: # Run voice activity detection on the audio frame
        is_speech = vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
    except webrtcvad.Error:
        return # Skip current frame if VAD fails

    if is_speech: # If speech is detected, reset the silence counter and store the frame for further use.
        silence_counter = 0
        speech_frames.append(audio_float.copy())
    else: # If no speech is detected, add to the silence counter.
        silence_counter += 1

    # Speech ended → push to queue
    if silence_counter > SILENCE_FRAMES_END and speech_frames: # If the silence counter exceeds the speech frames, process this loop.
        audio_np = np.concatenate(speech_frames) # Combine frames
        speech_frames.clear() # Reset buffer
        silence_counter = 0 # Reset silence counter
        speech_queue.put(audio_np) # Send speech segment for further processing.

# ---------------- MAIN LOOP ----------------
# Open a microphone input that continuously produces audio frames.
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE_INDEX,
    blocksize=FRAME_SIZE,
    callback=callback
):
    print("Listening...")

    # Main processing loop, takes completed speech segments from the queue.
    while True:
        try:
            # Retrieve a speech segment produced by the callback
            audio_np = speech_queue.get(timeout=0.1)
        except queue.Empty:
            # If there isn't a speech segment, keep waiting in this loop.
            continue

        # Enforce minimum duration / skips audio that is too short
        if len(audio_np) < SAMPLE_RATE * MIN_AUDIO_SECONDS:
            continue

        # Pad with silence (critical for Whisper, reduces hallucinations and cutoffs during transcription)
        pad = np.zeros(int(PADDING_SECONDS * SAMPLE_RATE))
        audio_np = np.concatenate([pad, audio_np, pad]).astype(np.float32)

        # ---------------- WHISPER ----------------
        # Transcribe the speech using Whisper with mostly default parameters.
        result = model.transcribe(
            audio_np,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            fp16=False
        )

        text = result["text"].strip() # Cleans the output

        # Hard filter hallucinations
        if len(text) < 3:
            continue
        if sum(c.isdigit() for c in text) > len(text) * 0.3:
            continue

        print("→", text) # Print the speech

        # Match the transcription against the intents and objects.
        intent_match = find_match(text, intent)
        object_match = find_match(text, objects)

        # Trigger the corresponding action/announcement
        announce_action(intent_match, object_match)
