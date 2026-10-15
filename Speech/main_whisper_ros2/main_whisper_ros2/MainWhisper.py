import json
import queue

import numpy as np
import rclpy
import sounddevice as sd
import webrtcvad
from ament_index_python.packages import get_package_share_directory
from moonshine_voice import Transcriber, get_model_for_language
from rclpy.node import Node
from std_msgs.msg import String


SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
MIN_AUDIO_SECONDS = 0.8
SILENCE_FRAMES_END = 15
MAX_SPEECH_FRAMES = 750
PADDING_SECONDS = 0.2


def normalize(text):
    return text.lower().strip()


def find_match(text, term_dict):
    text = normalize(text)
    for key, phrases in term_dict.items():
        for phrase in phrases:
            if phrase in text:
                return key
    return None


class MainWhisperNode(Node):
    def __init__(self):
        super().__init__("main_whisper")

        self.declare_parameter("device_index", 1)
        device_index = self.get_parameter("device_index").value

        share_dir = get_package_share_directory("main_whisper_ros2")
        with open(f"{share_dir}/data/intent_terms.json", encoding="utf-8") as intent_file:
            self.intent_terms = json.load(intent_file)
        with open(f"{share_dir}/data/objects_terms.json", encoding="utf-8") as objects_file:
            self.object_terms = json.load(objects_file)

        self.transcript_publisher = self.create_publisher(
            String, "voice/transcript", 10
        )
        self.command_publisher = self.create_publisher(String, "voice/command", 10)

        self.vad = webrtcvad.Vad(2)
        model_path, model_arch = get_model_for_language("en")
        self.model = Transcriber(model_path=model_path, model_arch=model_arch)

        self.speech_frames = []
        self.silence_counter = 0
        self.speech_queue = queue.Queue()
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
            blocksize=FRAME_SIZE,
            callback=self._audio_callback,
        )
        self.input_stream.start()
        self.create_timer(0.05, self._process_audio)
        self.get_logger().info(f"Listening on audio device {device_index}")

    def _audio_callback(self, indata, _frames, _time_info, status):
        if status:
            return

        audio_float = indata[:, 0]
        audio_int16 = (audio_float * 32768).astype(np.int16)
        if len(audio_int16) != FRAME_SIZE:
            return

        try:
            is_speech = self.vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
        except webrtcvad.error:
            return

        if is_speech:
            self.silence_counter = 0
            self.speech_frames.append(audio_float.copy())
        else:
            self.silence_counter += 1

        if (
            self.silence_counter > SILENCE_FRAMES_END
            or len(self.speech_frames) >= MAX_SPEECH_FRAMES
        ):
            if self.speech_frames:
                audio = np.concatenate(self.speech_frames)
                self.speech_frames.clear()
                self.silence_counter = 0
                self.speech_queue.put(audio)

    def _process_audio(self):
        try:
            audio = self.speech_queue.get_nowait()
        except queue.Empty:
            return

        if len(audio) < SAMPLE_RATE * MIN_AUDIO_SECONDS:
            return

        padding = np.zeros(int(PADDING_SECONDS * SAMPLE_RATE))
        audio = np.concatenate([padding, audio, padding]).astype(np.float32)
        transcript = self.model.transcribe_without_streaming(
            audio,
            sample_rate=SAMPLE_RATE,
            flags=0,
        )
        text = " ".join(line.text for line in transcript.lines).strip()

        if len(text) < 3 or sum(char.isdigit() for char in text) > len(text) * 0.3:
            return

        self.get_logger().info(f"Heard: {text}")
        transcript_message = String()
        transcript_message.data = text
        self.transcript_publisher.publish(transcript_message)

        intent = find_match(text, self.intent_terms)
        obj = find_match(text, self.object_terms)
        if intent and obj:
            self.get_logger().info(f"ACTION: {intent.upper()} -> {obj.upper()}")
        elif intent:
            self.get_logger().info(f"INTENT DETECTED: {intent.upper()} (no object specified)")
        elif obj:
            self.get_logger().info(f"OBJECT DETECTED: {obj.upper()} (no intent specified)")

        if intent or obj:
            command_message = String()
            command_message.data = json.dumps({"intent": intent, "object": obj})
            self.command_publisher.publish(command_message)

    def destroy_node(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MainWhisperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
