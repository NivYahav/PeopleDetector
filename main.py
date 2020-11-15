import cv2
import multiprocessing
import datetime
import argparse

def read(q, video):
    """"
    function to read video and stack its frames in a q structure
    takes 2 parameters:

    q: Queue for storing the frames

    video: video path

    """

    print("Reading Video...")

    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            q.put(frame)

        else:
            break
    print("Done!")


def detector(q, q1, Blur=False):
    """
    function to detect peoples with the given frames.
    this function takes 3 parameters:

    q: unpacking the q from the reader function

    q2: store the new frames with the detections

    Blur: this feature designed to blur the detected people

    """

    # Define body classifier cascade
    body_classifier = cv2.CascadeClassifier(r'C:\Users\nivy1\virtual_env\Lib\site-packages\cv2\data\haarcascade_fullbody.xml')

    print("Detecting...")
    # Unpack the Queue till it's empty
    while not q.empty():

        frame = q.get()
        # Returns the detected bounding boxes coordinates
        bodies = body_classifier.detectMultiScale(frame, 1.05, 5)
        # Draw rectangle
        for (x, y, w, h) in bodies:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Blurring region of interest
            if Blur:

                # Create 2 image copies
                blurred = frame.copy()
                roi = frame.copy()
                # Use the coordinates from bounding boxes to discribe the ROI area
                roi = roi[y:y + h, x:x + w]
                # Averaging method
                blurred_roi = cv2.blur(roi, (7,7))
                # Use the second copy to replace each box with the blurred one
                blurred[y:y + h, x:x + w] = blurred_roi
                q1.put(blurred)
            else:
                q1.put(frame)

    print("Writing predictions to video...")
    video_name = "PeopleDetector.avi"

    # Video encoding
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # FMP4,#XVID
    # Video features
    video = cv2.VideoWriter(video_name, fourcc, 180, (1280, 720))

    # Put the date and time variable over the video frame
    dt = str(datetime.datetime.now())
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    # Dequeue the predictions frames into a video
    while not q1.empty():
        frame = q1.get()
        # Write current date and time on each frame
        frame = cv2.putText(frame, dt, (10, 100), font, 1, (0, 0, 255), 4, cv2.LINE_8)
        video.write(frame)
    print("Done!")


def displayer(video):
    """
    displayer is a simple function to display videos

    takes a single parameter.
    video: indicates the camera instances which can be a video path or
           live web camera

    """

    cap = cv2.VideoCapture(video)

    print("Playing...")
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            cv2.imshow(video, frame)

            if cv2.waitKey(25) & 0xFF == 27:
                break
        else:
            break
    # Shutting down instance
    cap.release()
    cv2.destroyAllWindows()
    
    print("Done!")

def main():

    parser = argparse.ArgumentParser(description='Blocks Communication.')

    parser.add_argument('-i', "--input", type=str, help="input file video", default="People.mp4")

    parser.add_argument('-b', '--Blur', type=bool, help="Blur detections", default=False)

    parser.add_argument('-o', '--output', type=str, help="output video name", default='PeopleDetector.avi')

    args = parser.parse_args()
    # Define Queues
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    read(q1, args.input)
    detector(q1, q2, args.Blur)
    displayer(args.output)


if __name__ == '__main__':

    main()
