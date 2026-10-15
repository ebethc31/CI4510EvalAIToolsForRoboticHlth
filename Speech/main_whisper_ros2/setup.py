from setuptools import find_packages, setup

package_name = "main_whisper_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/data",
            [
                "main_whisper_ros2/data/intent_terms.json",
                "main_whisper_ros2/data/objects_terms.json",
            ],
        ),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "webrtcvad",
        "sounddevice",
        "moonshine-voice",
    ],
    zip_safe=True,
    maintainer="Package Maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS 2 voice command node using Moonshine Voice and WebRTC VAD.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "main_whisper = main_whisper_ros2.MainWhisper:main",
        ],
    },
)
