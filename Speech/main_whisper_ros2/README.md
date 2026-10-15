# MainWhisper ROS 2 package

This `ament_python` package wraps the existing microphone/VAD/Moonshine pipeline
in a ROS 2 node. It includes the intent and object phrase libraries.

## Dependencies

ROS 2 dependencies: `rclpy`, `std_msgs`, and `ament_index_python`.

Python dependencies are listed in `requirements.txt`: NumPy, WebRTC VAD,
SoundDevice, and Moonshine Voice. Install these into the same Python environment
used by ROS 2:

```bash
python -m pip install -r requirements.txt
```

## Build and run

Place `main_whisper_ros2` in a ROS 2 workspace's `src` directory, install the
Python dependencies, then build and source the workspace. On Linux:

```bash
colcon build --packages-select main_whisper_ros2
source install/setup.bash
ros2 run main_whisper_ros2 main_whisper
```

On Windows, run the build and setup commands from a ROS 2-enabled command
prompt:

```bat
colcon build --packages-select main_whisper_ros2
call install\setup.bat
ros2 run main_whisper_ros2 main_whisper
```

By default, the node opens audio device index `1`, matching the original script.
Override it if needed:

```bash
ros2 run main_whisper_ros2 main_whisper --ros-args -p device_index:=0
```

## Topics

- `voice/transcript` (`std_msgs/msg/String`): each accepted transcript.
- `voice/command` (`std_msgs/msg/String`): JSON with `intent` and `object`
  fields when either library has a match, for example
  `{"intent": "grab", "object": "pen"}`. An unmatched field is `null`.
