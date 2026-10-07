import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

CAMERA_INDEX = 0

# YOLO face model.
MODEL_NAME = "yolov11s-face.pt"

# Hat size relative to face width.
HAT_WIDTH_SCALE = 2.0

# How far above the detected face the hat should be placed.
HAT_VERTICAL_OFFSET = 40

# Supported hat formats.
HAT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ============================================================
# Utility functions
# ============================================================

def load_hats(hat_directory):
    """
    Load all hat images from the hats directory.

    PNG files are preferred because they can contain transparency.
    """

    hat_directory = Path(hat_directory)

    if not hat_directory.exists():
        print(f"Hat directory does not exist: {hat_directory}")
        return []

    hats = []

    for path in hat_directory.iterdir():

        if path.suffix.lower() not in HAT_EXTENSIONS:
            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            print(f"Could not load hat: {path}")
            continue

        # Make sure the image has 4 channels.
        if len(image.shape) == 2:
            # Grayscale -> BGRA
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGRA
            )

        elif image.shape[2] == 3:

            alpha = np.full(
                (image.shape[0], image.shape[1]),
                255,
                dtype=np.uint8
            )

            image = np.dstack(
                (image, alpha)
            )

        hats.append({
            "name": path.name,
            "image": image
        })

    return hats


def resize_hat(hat, target_width):
    """
    Resize a hat while preserving its aspect ratio.
    """

    original_height, original_width = hat.shape[:2]

    if original_width <= 0 or original_height <= 0:
        return None

    scale = target_width / original_width

    target_height = max(
        1,
        int(original_height * scale)
    )

    resized = cv2.resize(
        hat,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def overlay_transparent(background, overlay, x, y):
    """
    Overlay a BGRA image onto a BGR image.

    The overlay is clipped to the camera frame, so hats can
    safely extend beyond any edge of the screen.
    """

    x = int(x)
    y = int(y)

    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    # Calculate the visible intersection.
    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(
        x + overlay_width,
        bg_width
    )

    y2 = min(
        y + overlay_height,
        bg_height
    )

    # Hat is completely outside the frame.
    if x1 >= x2 or y1 >= y2:
        return background

    # Coordinates inside the hat image.
    overlay_x1 = int(x1 - x)
    overlay_y1 = int(y1 - y)

    overlay_x2 = int(
        overlay_x1 + (x2 - x1)
    )

    overlay_y2 = int(
        overlay_y1 + (y2 - y1)
    )

    # Safety clamp.
    overlay_x1 = max(
        0,
        min(overlay_x1, overlay_width)
    )

    overlay_x2 = max(
        0,
        min(overlay_x2, overlay_width)
    )

    overlay_y1 = max(
        0,
        min(overlay_y1, overlay_height)
    )

    overlay_y2 = max(
        0,
        min(overlay_y2, overlay_height)
    )

    # Make sure we still have something to draw.
    if (
        overlay_x1 >= overlay_x2
        or overlay_y1 >= overlay_y2
    ):
        return background

    # Crop the visible portion of the hat.
    overlay_crop = overlay[
        overlay_y1:overlay_y2,
        overlay_x1:overlay_x2
    ]

    # Split BGR and alpha.
    overlay_bgr = overlay_crop[:, :, :3]

    alpha = (
        overlay_crop[:, :, 3]
        .astype(np.float32)
        / 255.0
    )

    # Add channel dimension for broadcasting.
    alpha = alpha[:, :, np.newaxis]

    # Corresponding region in webcam frame.
    background_crop = background[
        y1:y2,
        x1:x2
    ]

    # Alpha blend.
    blended = (
        overlay_bgr.astype(np.float32) * alpha
        + background_crop.astype(np.float32)
        * (1.0 - alpha)
    )

    background[
        y1:y2,
        x1:x2
    ] = blended.astype(np.uint8)

    return background


def get_hat_position(
    face_x,
    face_y,
    face_width,
    hat_width,
    hat_height
):
    """
    Calculate the position of the hat.

    The hat is horizontally centered over the face.

    The bottom of the hat sits at the top of the face.
    """

    face_center_x = (
        face_x + face_width / 2
    )

    hat_x = int(
        face_center_x - hat_width / 2
    )

    hat_y = int(
        face_y - hat_height
    )

    return hat_x, hat_y


def get_face_center(x1, y1, x2, y2):
    """
    Calculate the center point of a face.
    """

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def find_matching_face(
    center_x,
    center_y,
    previous_faces,
    max_distance
):
    """
    Find the previous face closest to the current face.

    Returns the ID of the matching face, or None if no
    suitable match is found.
    """

    best_id = None
    best_distance = float("inf")

    for face_id, data in previous_faces.items():

        old_x = data["center_x"]
        old_y = data["center_y"]

        distance = (
            (center_x - old_x) ** 2
            + (center_y - old_y) ** 2
        ) ** 0.5

        if (
            distance < max_distance
            and distance < best_distance
        ):
            best_distance = distance
            best_id = face_id

    return best_id


# ============================================================
# Main program
# ============================================================

def main():

    print("Loading YOLO face detector...")

    model = YOLO(MODEL_NAME)

    print("Loading hats...")

    hats = load_hats("hats")

    if not hats:

        print()
        print("ERROR: No hats found.")
        print(
            "Put PNG/JPG hat images inside "
            "the 'hats' folder."
        )

        return

    print(
        f"Loaded {len(hats)} hat(s):"
    )

    for hat in hats:
        print(
            f"  - {hat['name']}"
        )

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    print()
    print("Webcam started.")
    print("Press Q to quit.")
    print(
        "Each detected face gets a random hat."
    )

    # --------------------------------------------------------
    # Face tracking data
    #
    # Each tracked face gets its own hat.
    # --------------------------------------------------------

    tracked_faces = {}

    next_face_id = 0

    # Number of frames a face is allowed to disappear
    # before its tracking information is removed.
    MAX_MISSING_FRAMES = 10

    # Maximum movement between frames used for matching.
    # This is relative to the detected face size.
    MAX_MATCH_DISTANCE_SCALE = 1.5

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read frame "
                "from webcam."
            )

            break

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # YOLO face detection
        # ----------------------------------------------------

        results = model(
            frame,
            verbose=False
        )

        # Keep track of faces detected in this frame.
        current_face_ids = set()

        # ----------------------------------------------------
        # Process detected faces
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # --------------------------------------------
                # Get bounding box coordinates
                # --------------------------------------------

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = coordinates

                # Convert to integers.
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Clamp face coordinates.
                x1 = max(
                    0,
                    min(
                        x1,
                        frame_width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame_height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame_width
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame_height
                    )
                )

                face_width = x2 - x1
                face_height = y2 - y1

                if (
                    face_width <= 0
                    or face_height <= 0
                ):
                    continue

                # --------------------------------------------
                # Face center
                # --------------------------------------------

                center_x, center_y = (
                    get_face_center(
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                # --------------------------------------------
                # Try to match this face to a face from
                # the previous frame.
                # --------------------------------------------

                max_match_distance = (
                    face_width
                    * MAX_MATCH_DISTANCE_SCALE
                )

                matched_id = (
                    find_matching_face(
                        center_x,
                        center_y,
                        tracked_faces,
                        max_match_distance
                    )
                )

                # --------------------------------------------
                # New face
                # --------------------------------------------

                if matched_id is None:

                    face_id = next_face_id

                    next_face_id += 1

                    # Pick a RANDOM hat for this face.
                    random_hat_index = random.randrange(
                        len(hats)
                    )

                    tracked_faces[face_id] = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "hat_index": random_hat_index,
                        "missing_frames": 0
                    }

                else:

                    face_id = matched_id

                    # Update face position.
                    tracked_faces[
                        face_id
                    ]["center_x"] = center_x

                    tracked_faces[
                        face_id
                    ]["center_y"] = center_y

                    tracked_faces[
                        face_id
                    ]["missing_frames"] = 0

                current_face_ids.add(face_id)

                # --------------------------------------------
                # Get this face's assigned random hat.
                # --------------------------------------------

                hat_index = tracked_faces[
                    face_id
                ]["hat_index"]

                current_hat = hats[
                    hat_index
                ]["image"]

                # --------------------------------------------
                # Calculate hat width.
                #
                # Hat width = face width * 2.0
                # Change HAT_WIDTH_SCALE above if desired.
                # --------------------------------------------

                hat_width = int(
                    face_width
                    * HAT_WIDTH_SCALE
                )

                if hat_width <= 0:
                    continue

                # --------------------------------------------
                # Resize hat
                # --------------------------------------------

                resized_hat = resize_hat(
                    current_hat,
                    hat_width
                )

                if resized_hat is None:
                    continue

                hat_height = (
                    resized_hat.shape[0]
                )

                # --------------------------------------------
                # Calculate hat position
                # --------------------------------------------

                hat_x, hat_y = (
                    get_hat_position(
                        x1,
                        y1,
                        face_width,
                        hat_width,
                        hat_height
                    )
                )

                # Vertical offset.
                hat_y += (
                    HAT_VERTICAL_OFFSET
                    * hat_width
                    / 200
                )

                # --------------------------------------------
                # Draw hat
                # --------------------------------------------

                frame = overlay_transparent(
                    frame,
                    resized_hat,
                    hat_x,
                    hat_y
                )

                # --------------------------------------------
                # Optional face rectangle.
                #
                # Uncomment if you want to see the
                # detected face boxes.
                # --------------------------------------------

                # cv2.rectangle(
                #     frame,
                #     (x1, y1),
                #     (x2, y2),
                #     (0, 255, 0),
                #     2
                # )

        # ----------------------------------------------------
        # Update tracking information.
        #
        # Faces that disappear temporarily are kept for a
        # few frames so their hat does not immediately change.
        # ----------------------------------------------------

        faces_to_remove = []

        for face_id in tracked_faces:

            if face_id not in current_face_ids:

                tracked_faces[
                    face_id
                ]["missing_frames"] += 1

                if (
                    tracked_faces[
                        face_id
                    ]["missing_frames"]
                    > MAX_MISSING_FRAMES
                ):

                    faces_to_remove.append(
                        face_id
                    )

        for face_id in faces_to_remove:

            del tracked_faces[
                face_id
            ]

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "YOLO Face Hats",
            frame
        )

        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
