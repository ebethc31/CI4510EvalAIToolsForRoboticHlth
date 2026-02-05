import cv2


def find_available_cameras(max_tested=10):
    """Test camera indices from 0 to max_tested-1 and return available ones."""
    available_cameras = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get camera info
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera {i}: Available - Resolution: {int(width)}x{int(height)}")
                available_cameras.append(i)
            cap.release()

    return available_cameras


if __name__ == "__main__":
    print("Scanning for available cameras...")
    cameras = find_available_cameras()

    if cameras:
        print(f"\nFound {len(cameras)} camera(s) at index/indices: {cameras}")
    else:
        print("\nNo cameras found.")
