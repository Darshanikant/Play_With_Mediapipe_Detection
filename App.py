# import the libraries
import cv2
import time
import mediapipe as mp
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configure Streamlit app
st.set_page_config(page_title="🎥 Mediapipe Library detaction 🚀",
                   layout="wide", page_icon="🤖")

# Add background image via CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.freepik.com/premium-photo/human-brain-plain-background-with-copy-space_143463-5271.jpg");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }
        .main-heading {
            font-size: 2.5rem;
            color: #ffffff;
            text-shadow: 2px 2px 5px #000000;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# Main heading
st.markdown('<h1 class="main-heading">🎥 Play with Mediapipe Detection🚀</h1>',
            unsafe_allow_html=True)
st.sidebar.title("🌟 **MediaPipe Multi-Feature App** 🌟")
st.sidebar.markdown("""
### 👋 Welcome to the **MediaPipe Magic App**!
Explore **AI-powered features** for real-time detection and transformation. 🎉

#### 🚀 **Features Include:**
- 🖼️ **Face & Hand Landmark Detection**  
- 🎭 **Face Transformations**:  
  - ✨ Transformation 1  
  - 🌈 Transformation 2  
  - 🔮 Transformation 3  
  - 🌀 Transformation 4  
- 👱 **Face Tracking**  
- 🏃‍ **Body Pose Tracking**

---

### 🛠️ **How to Use:**
1. **🔍 Select a Feature**: Face/Hand Detection, Transformations, Pose, or Meditation.  
2. **📤 Upload Your File** or **📹 Start the Webcam**.  
3. **✨ Click "Process"** to see the magic happen.  
4. **💾 Save Results** or try another feature!

---

### 📝 **Credits**:
- **Developed by Darshanikanta**  
- **© All rights reserved.**
""")



# Dropdown selector for app features
option = st.selectbox(
    "🌟 **Choose a Feature to Explore:**",
    [
        "None (App Details)",
        "Face & Hand Landmark Detection 🤲",
        "Hand Tracking 🤲",
        "Face Transformation 1 ✨",
        "Face Transformation 2 🌈",
        "Face Transformation 3 🔮",
        "Face Transformation 4 🌀",
        "Face Tracking 👱",
        "Body Pose Tracking 🏃‍",
        "Pose Tracking 🙋‍"
    ],
    index=0,
    help="Select a feature to explore real-time AI-powered detection and transformation."
)

# Display message based on selected option
if option == "None (App Details)":
    st.subheader("🧐 **Discover the App:**")
    st.write("""
    Welcome to the **MediaPipe Multi-Feature App**!  
    Unlock real-time AI-powered detection and transformation with advanced features:
    """)

    st.markdown("""
    #### 🌟 **What You Can Do:**
    - Detect **Face & Hand Landmarks** 🤲
    - Explore creative **Face Transformations** ✨🌈🔮🌀
    - Track your **Body Movements** 🏃
    - Track Your **Face** 👱
    - Track **Body Pose from Image**🙋‍
    """)

    st.success("Ready to begin? **Select a feature** from the dropdown above and explore the magic of MediaPipe! 🚀")
    
# Set option for Face & Hand Landmark Detection
if option == "Face & Hand Landmark Detection 🤲":
    # Add Start and Stop buttons
    start_detection = st.button("🚀 Start Detection")
    stop_detection = st.button("🛑 Stop Detection")

    if start_detection:
        # Grabbing the Holistic Model from Mediapipe and initializing the model
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initializing the drawing utils for drawing the facial landmarks on the image
        mp_drawing = mp.solutions.drawing_utils

        # Open webcam video stream
        capture = cv2.VideoCapture(0)

        # Initializing time variables for FPS calculation
        previousTime = 0

        st.write("⏳ Starting webcam detection... Press **Stop Detection** to terminate.")

        # Create a placeholder for dynamic image updates
        stframe = st.empty()

        # Real-time detection loop
        while capture.isOpened():
            # Read frames from the webcam
            ret, frame = capture.read()
            if not ret:
                st.write("⚠️ Unable to read from webcam. Please check your camera connection.")
                break

            # Resize frame for better visualization
            frame = cv2.resize(frame, (800, 600))

            # Convert frame from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Improve performance
            results = holistic_model.process(image)  # Make predictions
            image.flags.writeable = True  # Allow modifications

            # Convert frame back to BGR for OpenCV rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw facial landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
                )

            # Draw right-hand landmarks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Draw left-hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            # Display FPS on the frame
            cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Render the frame in a Streamlit window (one live frame)
            stframe.image(image, channels="BGR", use_column_width=True)

            # Break loop if Stop Detection is pressed
            if stop_detection:
                st.write("🛑 Detection stopped!")
                break

        # Release resources after stopping detection
        capture.release()
        cv2.destroyAllWindows()

        # Message to confirm termination
        st.write("✅ Webcam detection stopped successfully.")
    st.write("🔍 Detecting face and hand landmarks in real-time!")
elif option == "Hand Tracking 🤲":
    # Add Start and Stop buttons
    start_tracking = st.button("🚀 Start Hand Tracking")
    stop_tracking = st.button("🛑 Stop Hand Tracking")

    if start_tracking:
    
        # Set up MediaPipe
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        # Set up video capture
        cap = cv2.VideoCapture(0)  # Change the index if you want to use a different camera

        # Set up MediaPipe Hands
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
            stframe = st.empty()
            
            while cap.isOpened():
                # Read frame from camera
                success, frame = cap.read()
                if not success:
                    st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                    break

                # Convert the BGR image to RGB and process it with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Draw hand landmarks on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Show the frame with hand landmarks
                stframe.image(frame, channels="BGR", use_column_width=True)

                # Stop hand tracking if Stop button is pressed
                if stop_tracking:
                    st.write("🛑 Hand Tracking stopped!")
                    break

        # Release resources after stopping tracking
        cap.release()
        cv2.destroyAllWindows()

        # Message to confirm termination
        st.write("✅ Hand Tracking stopped successfully.")

if option == "Face Transformation 1 ✨":
    
    # Add Start and Stop buttons
    start_transformation = st.button("🚀 Start Transformation")
    stop_transformation = st.button("🛑 Stop Transformation")

    if start_transformation:
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_holistic = mp.solutions.holistic
        
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Start video capture
        cap = cv2.VideoCapture(0)
        st.write("⏳ Starting **Face Transformation 1**... Press **Stop Transformation** to terminate.")

        # Create a placeholder for dynamic image updates
        stframe = st.empty()

        # Use MediaPipe FaceMesh and Holistic model for Hand Landmark
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic_model:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                    break

                # Process the image for face and hand landmarks
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(image)
                results_hands = holistic_model.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw face landmarks
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                        )

                # Draw hand landmarks
                if results_hands.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results_hands.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results_hands.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results_hands.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Display the processed frame in Streamlit (one live frame)
                stframe.image(image, channels="BGR", use_column_width=True)

                # Stop loop if Stop Transformation is pressed
                if stop_transformation:
                    st.write("🛑 Transformation stopped!")
                    break

        # Release resources and cleanup
        cap.release()
        cv2.destroyAllWindows()
        st.write("✅ Face Transformation 1 terminated successfully.")

    
elif option == "Face Transformation 2 🌈":
    # Add Start and Stop buttons
    start_transformation = st.button("🚀 Start Transformation")
    stop_transformation = st.button("🛑 Stop Transformation")

    if start_transformation:
       
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

        # Set the transformation matrix (example: scaling)
        transformation_matrix = np.array([[1.5, 0, 0],
                                          [0, 1.5, 0],
                                          [0, 0, 1]])

        def transform_3d_face(image, landmarks):
            # Perform 3D face transformation
            transformed_landmarks = np.matmul(landmarks, transformation_matrix.T)
            transformed_image = image.copy()

            for i in range(transformed_landmarks.shape[0]):
                x, y, _ = transformed_landmarks[i]
                x = int(x * image.shape[1])
                y = int(y * image.shape[0])

                transformed_landmarks[i] = [x, y, _]
                cv2.circle(transformed_image, (x, y), 1, (255, 0, 0), -1)

            return transformed_image

        # Start video capture
        cap = cv2.VideoCapture(0)
        st.write("⏳ Starting **Face Transformation 2**... Press **Stop Transformation** to terminate.")
        
        # Real-time frame-by-frame loop
        stframe = st.empty()
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                break

            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe face mesh
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert face landmarks to a NumPy array
                    landmarks = np.zeros((468, 3), dtype=np.float32)
                    for i, landmark in enumerate(face_landmarks.landmark):
                        landmarks[i] = [landmark.x, landmark.y, landmark.z]

                    # Perform 3D face transformation
                    transformed_image = transform_3d_face(image, landmarks)

                    # Draw the face mesh on the image
                    mp_drawing.draw_landmarks(
                        transformed_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

            # Display the transformed image in Streamlit (one live frame)
            stframe.image(transformed_image, channels="BGR", use_column_width=True)

            # Stop loop if Stop Transformation is pressed
            if stop_transformation:
                st.write("🛑 Transformation stopped!")
                break

        # Release resources and cleanup
        cap.release()
        cv2.destroyAllWindows()
        st.write("✅ Face Transformation 2 terminated successfully.")

elif option == "Face Transformation 3 🔮":
    # Add Start and Stop buttons
    start_transformation = st.button("🚀 Start Transformation")
    stop_transformation = st.button("🛑 Stop Transformation")

    if start_transformation:
        
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh

        # Initialize MediaPipe face mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

        # Define function to get bounding box coordinates
        def get_face_bbox(landmarks, image_shape):
            x_coordinates = [landmark[0] for landmark in landmarks]
            y_coordinates = [landmark[1] for landmark in landmarks]

            xmin = int(min(x_coordinates) * image_shape[1])
            ymin = int(min(y_coordinates) * image_shape[0])
            xmax = int(max(x_coordinates) * image_shape[1])
            ymax = int(max(y_coordinates) * image_shape[0])

            return xmin, ymin, xmax, ymax

        # Define transformation function
        def transform_3d_face(image, landmarks, replacement_face):
            transformed_image = image.copy()

            xmin, ymin, xmax, ymax = get_face_bbox(landmarks, image.shape[:2])
            resized_replacement_face = cv2.resize(replacement_face, (xmax - xmin, ymax - ymin))
            mask = cv2.cvtColor(resized_replacement_face, cv2.COLOR_BGR2GRAY) / 255.0
            replacement_face_rgb = resized_replacement_face[:, :, :3]

            roi = transformed_image[ymin:ymax, xmin:xmax]
            roi = roi * (1 - mask[:, :, np.newaxis]) + replacement_face_rgb * mask[:, :, np.newaxis]
            transformed_image[ymin:ymax, xmin:xmax] = roi

            return transformed_image

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Capture replacement face image
        st.write("📸 Capturing **Replacement Face** from webcam...")
        ret, replacement_face = cap.read()
        if not ret:
            st.error("⚠️ Failed to capture the replacement face image. Please try again.")
            cap.release()
            exit()

        st.write("✅ Replacement face image captured successfully!")

        # Start real-time transformation
        stframe = st.empty()
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                break

            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe face mesh
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

                    # Perform 3D face transformation
                    transformed_image = transform_3d_face(image, landmarks, replacement_face)

                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        transformed_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

            # Display the transformed image in Streamlit
            stframe.image(transformed_image, channels="BGR", use_column_width=True)

            # Stop if the stop button is pressed
            if stop_transformation:
                st.write("🛑 Transformation stopped!")
                break

        cap.release()
        cv2.destroyAllWindows()
        st.write("✅ Face Transformation 3 terminated successfully.")
elif option == "Face Transformation 4 🌀":
    # Add Start and Stop buttons
    start_transformation = st.button("🚀 Start Transformation")
    stop_transformation = st.button("🛑 Stop Transformation")

    if start_transformation:
        

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Initialize MediaPipe pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        # Default color range for clothing detection
        lower_color = np.array([0, 0, 0])  # Default lower range for black
        upper_color = np.array([50, 50, 50])  # Default upper range for dark shades
        new_color = np.array([0, 255, 0])  # Default new color (green)

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                break

            # Flip the image for a mirror effect
            image = cv2.flip(image, 1)

            # Convert the BGR image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect pose landmarks
            results = pose.process(image_rgb)

            # Process detected pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    drawing_spec,
                    drawing_spec
                )

                # Create a mask for clothing color
                mask = cv2.inRange(image, lower_color, upper_color)

                # Apply the new color
                colored_mask = cv2.bitwise_and(image, image, mask=mask)
                colored_mask[mask > 0] = new_color

                # Combine with the original image
                result = cv2.bitwise_or(image, colored_mask)
            else:
                result = image

            # Display the result in Streamlit
            stframe.image(result, channels="BGR", use_column_width=True)

            # Stop if the stop button is pressed
            if stop_transformation:
                st.write("🛑 Transformation stopped!")
                break

        cap.release()
        cv2.destroyAllWindows()

# Start button for Face Tracking
if option == "Face Tracking 👱":
    # Add Start and Stop buttons
    start_tracking = st.button("🚀 Start Face Tracking")
    stop_tracking = st.button("🛑 Stop Face Tracking")

    if start_tracking:
        # Initialize MediaPipe Face Detection and Drawing Utils
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            st.warning("⚠️ Unable to access the webcam. Please check your webcam connection.")
            st.stop()

        # Start Face Detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            # Placeholder to update frame dynamically
            stframe = st.empty()

            st.write("⏳ Starting face tracking... Press **Stop Face Tracking** to terminate.")

            # Real-time detection loop
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    st.warning("⚠️ Unable to read the webcam frame.")
                    break

                # Convert BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                # Convert image back to BGR for OpenCV rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Draw the face detections
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(image, detection)

                # Display the resulting image in the Streamlit app
                stframe.image(image, channels="BGR", use_column_width=True)

                # Stop face tracking if the stop button is pressed
                if stop_tracking:
                    st.write("🛑 Face Tracking stopped!")
                    break

        # Release webcam resources and close windows
        cap.release()
        cv2.destroyAllWindows()
        st.write("✅ Face Tracking stopped successfully.")
elif option == "Body Pose Tracking 🏃‍":
    st.write("🏃‍ Tracking body pose in real-time!")

    # Add Start and Stop buttons
    start_tracking = st.button("🚀 Start Pose Tracking")
    stop_tracking = st.button("🛑 Stop Pose Tracking")

    if start_tracking:
        
        # Initialize MediaPipe Pose
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Initialize video capture
        cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide the path to a video file

        # Initialize BlazePose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            stframe = st.empty()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("⚠️ Unable to access the camera. Please check your webcam.")
                    break

                # Flip the frame horizontally for a mirror effect
                frame = cv2.flip(frame, 1)

                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image with BlazePose
                results = pose.process(image)

                # Render the landmarks on the frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the resulting frame
                stframe.image(frame, channels="BGR", use_column_width=True)

                # Stop the pose tracking if the stop button is pressed
                if stop_tracking:
                    st.write("🛑 Pose Tracking stopped!")
                    break

        cap.release()
        cv2.destroyAllWindows()
        st.write("✅ Pose Tracking stopped successfully.")

elif option == "Pose Tracking 🙋‍":
    # Add file uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # Step 1: Create a PoseLandmarker object.
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        detector = mp_pose.Pose(
            static_image_mode=True,  # Use static image mode for pose detection
            model_complexity=2,
            enable_segmentation=True
        )

        # Step 2: Read the uploaded image
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Step 3: Convert the image to RGB format and process it.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(image_rgb)

        # Step 4: Process the detection result and visualize it.
        annotated_image = image.copy()

        if results.pose_landmarks:
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Step 5: Display the annotated image in Streamlit with custom width (approx. 2-3 cm)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), channels="RGB", width=300)  # Width in pixels (approx. 2-3 cm)

        st.write("✅ Pose Tracking completed on the uploaded image.")


# Footer with copyright notice
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: #ffffff;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            font-size: 0.9rem;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
        }
    </style>
    <div class="footer">
        © 2024 | Developed by <b>Darshanikanta</b>
    </div>
""", unsafe_allow_html=True)