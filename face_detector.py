import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            self.reference = self.detect_face(image)
            if self.reference is None:
                return None
            return self.reference

        ref_rect = self.reference["rect"]
        ref_image = self.reference["image"]
        search_window = self.get_search_window(ref_rect)
        search_image = image[search_window[1]:search_window[1] + search_window[3],
                            search_window[0]:search_window[0] + search_window[2]]
        result = cv2.matchTemplate(search_image, ref_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        detected_rect = {
            "x": search_window[0] + max_loc[0],
            "y": search_window[1] + max_loc[1],
            "w": ref_rect[2],
            "h": ref_rect[3]
        }

        if max_val < self.tm_threshold:
            self.reference = self.detect_face(image)
            if self.reference is None:
                return None
        else:
            self.reference["rect"] = detected_rect
            self.reference["image"] = self.align_face(image, detected_rect)

        return self.reference



    
    def get_search_window(self, rect):
        search_window = [
            max(rect[0] - self.tm_window_size, 0),
            max(rect[1] - self.tm_window_size, 0),
            rect[2] + 2 * self.tm_window_size,
            rect[3] + 2 * self.tm_window_size
        ]
        return search_window



    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        print(image.shape)
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

def main():
    # Create an instance of FaceDetector with desired parameters.

    ## optimal parametrs set here: with lower threshold, the crop face fizzes out and throws an error. the higher threshodl helps in better tracking the face.
    face_detector = FaceDetector(tm_window_size=40, tm_threshold=0.9, aligned_image_size=224)

    # Open the video capture using the system camera (index 0).
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_info = face_detector.track_face(frame)

        if face_info is not None:
            face_rect = face_info["rect"]
            cv2.rectangle(frame, (face_rect[0], face_rect[1]),
                          (face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]),
                          (0, 255, 0), 2)

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()