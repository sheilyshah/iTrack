"""
main.py

Capture data from a video source,
process the data using detect.py
Take appropriate actions with that data.
"""

import sys
import numpy as np
import cv2
import detect

class iTrackClient():
    """iTrack Client."""

    def __init__(self):
        # capture variables
        self.capture = None
        self.capture_running = False

        # detection variables
        self.face_detector, self.eye_detector, self.detector = detect.init_cv()

        self.previous_right_keypoints = None
        self.previous_left_keypoints = None
        self.previous_right_blob_area = None
        self.previous_left_blob_area = None


        self.start_capture()



        



    def start_capture(self):
        """Starts video capture."""

        if not self.capture_running:
            self.capture = cv2.VideoCapture(0)
            if self.capture is None:
                sys.exit('Video capture failed.')


            self.capture_running = True
            

            while(self.capture_running):
                # Capture frame-by-frame
                ret, frame = self.capture.read()
                self.process_frame(frame)
            self.stop_capture()


    def stop_capture(self):
        """Stop video capture.
        
        TODO: currently this is inaccessible. with a gui, this is useful
        """
        self.capture.release()

    
    def display_image(self, img):
        cv2.imshow("preview", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.capture_running = False



    def get_keypoints(self, frame, frame_gray, threshold, previous_keypoint, previous_area):

        keypoints = detect.process_eye(frame_gray, threshold, self.detector,
                                        prevArea=previous_area)
        if keypoints:
            previous_keypoint = keypoints
            previous_area = keypoints[0].size
        else:
            keypoints = previous_keypoint
        return keypoints, previous_keypoint, previous_area

        
    def process_frame(self, frame):
        """Process frame."""

        if frame is None:
            return
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if frame_gray is None:
            return

        face_frame, face_frame_gray, left_eye_estimated_position, right_eye_estimated_position, _, _ = detect.detect_face(
            frame,
            frame_gray,
            self.face_detector
        )
      
        if face_frame is None:
            return
        
        left_eye_frame, right_eye_frame, left_eye_frame_gray, right_eye_frame_gray = detect.detect_eyes(
            face_frame,
            face_frame_gray,
            left_eye_estimated_position,
            right_eye_estimated_position,
            self.eye_detector
        )

        if right_eye_frame is not None:
            right_eye_threshold = 42
            right_keypoints, self.previous_right_keypoints, self.previous_right_blob_area = self.get_keypoints(
                right_eye_frame,
                right_eye_frame_gray,
                right_eye_threshold,
                previous_area=self.previous_right_blob_area,
                previous_keypoint=self.previous_right_keypoints
            )
            detect.draw_blobs(right_eye_frame, right_keypoints)

            right_eye_frame = np.require(right_eye_frame, np.uint8, 'C')
            self.display_image(right_eye_frame)

        if left_eye_frame is not None:
            left_eye_threshold = 42
            left_keypoints, self.previous_left_keypoints, self.previous_left_blob_area = self.get_keypoints(
                left_eye_frame, left_eye_frame_gray, left_eye_threshold,
                previous_area=self.previous_left_blob_area,
                previous_keypoint=self.previous_left_keypoints
            )
            detect.draw_blobs(left_eye_frame, left_keypoints)

            left_eye_frame = np.require(left_eye_frame, np.uint8, 'C')
            # self.display_image(left_eye_frame)

        # self.display_image(frame)
            



if __name__ == "__main__":
    client = iTrackClient()




