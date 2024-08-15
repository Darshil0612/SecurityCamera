import cv2
import depthai as dai
import numpy as np
import time
import blobconverter

# Constants
FOLLOW_THRESHOLD = 10  # Threshold duration for follow alert (in seconds)
NN_WIDTH, NN_HEIGHT = 300, 300
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
HIST_THRESHOLD = 0.7

# Dictionary to store the start time and ID for each person being tracked
follow_start_times = {}
unique_id_counter = 0  # Counter for assigning unique IDs
person_images = {}  # Dictionary to store saved images for each unique person


def calculate_iou(bbox1, bbox2):
    """Calculate the Intersection Over Union (IOU) between two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    # Calculate intersection
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    # Calculate the area of both bounding boxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def get_color_histogram(image, bbox):
    """Get a color histogram for the detected person to use as a descriptor."""
    x1, y1, x2, y2 = bbox
    person_roi = image[y1:y2, x1:x2]
    hist = cv2.calcHist([person_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compare_histograms(hist1, hist2):
    """Compare two color histograms using the correlation method."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def assign_unique_id(new_bbox, new_hist, tracked_people, frame):
    """Assign a unique ID to a detected person based on IOU and color histogram."""
    global unique_id_counter

    best_match_id = None
    highest_iou = 0
    highest_hist_match = 0

    for person_id, person_data in tracked_people.items():
        existing_bbox = person_data['bbox']
        existing_hist = person_data['hist']

        # Calculate IOU
        iou = calculate_iou(new_bbox, existing_bbox)
        hist_match = compare_histograms(new_hist, existing_hist)

        # Choose the person with the highest IOU and histogram match
        if iou > highest_iou and hist_match > HIST_THRESHOLD:
            highest_iou = iou
            highest_hist_match = hist_match
            best_match_id = person_id

    # Assign a new ID if no good match is found
    if highest_iou < IOU_THRESHOLD or highest_hist_match < HIST_THRESHOLD:
        unique_id_counter += 1
        return unique_id_counter

    return best_match_id


def trigger_alert(track_id, follow_duration, frame, bbox):
    """Trigger an alert when a person has been following for too long."""
    print(f"Alert: Person {track_id} has been following for {int(follow_duration)} seconds!")

    # Save the image of the detected person if not already saved
    if track_id not in person_images:
        x1, y1, x2, y2 = bbox
        person_image = frame[y1:y2, x1:x2]
        image_filename = f"person_{track_id}.jpg"
        cv2.imwrite(image_filename, person_image)
        person_images[track_id] = image_filename
        print(f"Captured image: {image_filename}")


def detect_and_track(frame, detections, tracked_people):
    """Detect and track persons with improved re-identification using IOU and histograms."""
    global follow_start_times
    current_time = time.time()

    tracked_ids = set()

    for det in detections:
        bbox = det[:4].astype(int)
        score = det[4]

        # Calculate color histogram for the new detection
        new_hist = get_color_histogram(frame, bbox)

        # Assign a unique ID to the detected person based on IOU and histogram
        track_id = assign_unique_id(bbox, new_hist, tracked_people, frame)

        # Draw bounding box and ID
        color = (0, 255, 0) if track_id not in follow_start_times else (0, 0, 255)
        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(frame, f'ID: {track_id} Score: {score:.2f}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Track the person and calculate follow time
        if track_id not in follow_start_times:
            follow_start_times[track_id] = current_time
        else:
            follow_duration = current_time - follow_start_times[track_id]
            cv2.putText(frame, f'Follow Time: {int(follow_duration)}s',
                        (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if follow_duration > FOLLOW_THRESHOLD:
                trigger_alert(track_id, follow_duration, frame, bbox)

        tracked_people[track_id] = {'bbox': bbox, 'hist': new_hist}
        tracked_ids.add(track_id)

    # Remove IDs that are no longer tracked
    follow_start_times = {k: v for k, v in follow_start_times.items() if k in tracked_ids}
    return tracked_people


def main():
    """Main function to run the OakCamera with person detection."""
    # Define a pipeline
    pipeline = dai.Pipeline()

    # Define a neural network that will detect faces
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", zoo_type="depthai", shaves=6))
    detection_nn.input.setBlocking(False)

    # Define camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(60)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Define manip
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)  # Ensure resize matches NN size
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    manip.inputConfig.setWaitForMessage(False)

    # Create outputs
    xout_cam = pipeline.create(dai.node.XLinkOut)
    xout_cam.setStreamName("cam")

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    cam.preview.link(manip.inputImage)
    cam.preview.link(xout_cam.input)
    manip.out.link(detection_nn.input)
    detection_nn.out.link(xout_nn.input)

    with dai.Device(pipeline) as device:
        # Output queues
        q_cam = device.getOutputQueue("cam", 4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        start_time = time.time()
        counter = 0
        fps = 0
        tracked_people = {}

        while True:
            in_frame = q_cam.get()
            in_nn = q_nn.get()

            frame = in_frame.getCvFrame()

            # Extract the 'detection_out' data
            detection_data = np.array(in_nn.getLayerFp16("detection_out")).reshape(-1, 7)

            dets = []
            for det in detection_data:
                image_id, label, conf, x_min, y_min, x_max, y_max = det
                if conf > CONFIDENCE_THRESHOLD:
                    x1, y1 = int(x_min * VIDEO_WIDTH), int(y_min * VIDEO_HEIGHT)
                    x2, y2 = int(x_max * VIDEO_WIDTH), int(y_max * VIDEO_HEIGHT)
                    dets.append([x1, y1, x2, y2, conf])

            dets = np.array(dets)

            # Apply NMS
            if dets.shape[0] > 0:
                bboxes = dets[:, 0:4].tolist()
                scores = dets[:, -1].tolist()

                indices = cv2.dnn.NMSBoxes(
                    bboxes=bboxes,
                    scores=scores,
                    score_threshold=CONFIDENCE_THRESHOLD,
                    nms_threshold=IOU_THRESHOLD
                )

                if len(indices) > 0:
                    indices = indices.flatten()  # Flatten the list of indices
                    dets = dets[indices]

            # Track and visualize detections
            tracked_people = detect_and_track(frame, dets, tracked_people)

            # Show FPS
            label_fps = "Fps: {:.2f}".format(fps)
            (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
            cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), (255, 255, 255), -1)
            cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))

            # Show frame
            cv2.imshow("Detections", frame)

            counter += 1
            if (time.time() - start_time) > 1:
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
