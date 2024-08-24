import depthai as dai
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

# Load the R3D model (pre-trained on action recognition dataset)
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define preprocessing transformations for the input frames
preprocess = transforms.Compose([
    transforms.Resize((128, 171)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# Define number of frames the model expects as input
SEQUENCE_LENGTH = 16  # R3D expects a sequence of 16 frames

# OAK-D Pipeline for RGB Camera
pipeline = dai.Pipeline()

# Setup RGB camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Updated to avoid the deprecation warning
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setVideoSize(1080, 720)

# Create XLink output node for RGB video
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

# Start the OAK-D device with the pipeline
device_oak = dai.Device(pipeline)
video_stream = device_oak.getOutputQueue(name="video", maxSize=4, blocking=False)

# Function to preprocess and create a sequence of frames
def preprocess_frames(frame_list):
    processed_frames = []
    for frame in frame_list:
        # Convert the frame from NumPy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Apply the preprocessing transformations
        processed_frame = preprocess(pil_image)
        processed_frames.append(processed_frame)
    
    return torch.stack(processed_frames).unsqueeze(0).to(device)

# Function to run violence detection using R3D model
def detect_violence(sequence):
    with torch.no_grad():
        outputs = model(sequence)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    violence_class = 1  # Index for "violence" (needs fine-tuning on custom dataset)
    return probabilities[0, violence_class].item() > 0.5  # Assuming violence threshold

# Initialize frame buffer for sequence
frame_buffer = []

# Video capture loop
while True:
    # Get frame from the OAK-D video stream
    video_packet = video_stream.get()
    frame = video_packet.getCvFrame()

    # Convert frame to RGB format for consistency
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add current frame to the frame buffer
    frame_buffer.append(frame_rgb)

    # Maintain a sequence length of 16 frames
    if len(frame_buffer) == SEQUENCE_LENGTH:
        # Detect violence on the current sequence of frames
        frame_sequence = preprocess_frames(frame_buffer)
        is_violent = detect_violence(frame_sequence)

        # Display the result on the frame
        label = "Violence Detected" if is_violent else "No Violence"
        color = (0, 0, 255) if is_violent else (0, 255, 0)

        # Overlay label on the frame
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Violence Detection", frame)

        # Clear the buffer for the next sequence
        frame_buffer = []

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
