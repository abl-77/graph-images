import face_recognition
import cv2
import matplotlib.pyplot as plt

def extract_face(image_path, output_size=(32, 32)):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print("No face detected in:", image_path)
        return None

    # Use first detected face
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]

    # Resize for graph processing
    face_image_resized = cv2.resize(face_image, output_size)

    return face_image_resized

# Test on one image
cropped_face = extract_face("data/real/fake/10.S.B.M.png")

if cropped_face is not None:
    plt.imshow(cropped_face)
    plt.title("Detected Face Region")
    plt.axis("off")
    plt.show()
