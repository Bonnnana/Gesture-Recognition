import cv2
import mediapipe as mp
import threading
import time


def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1

    return cnt


def play_video(video_path, stop_event, pause_event):
    clip = cv2.VideoCapture(video_path)
    while clip.isOpened() and not stop_event.is_set():
        if not pause_event.is_set():
            ret, frame = clip.read()
            if not ret:
                break
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1 / 30)
        else:
            time.sleep(0.1)
    clip.release()
    cv2.destroyWindow("Video")


def main():
    cap = cv2.VideoCapture(0)
    drawing = mp.solutions.drawing_utils
    hands = mp.solutions.hands
    hand_obj = hands.Hands(max_num_hands=1)

    video1 = 'video1.mp4'
    video2 = 'video2.mp4'

    start_init = False
    prev = -1
    playing = False
    paused = False
    current_video = None

    stop_event = threading.Event()
    pause_event = threading.Event()

    while True:
        end_time = time.time()
        ret, frm = cap.read()
        frm = cv2.flip(frm, 1)

        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keyPoints)

            if not (prev == cnt):
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif (end_time - start_time) > 0.2:
                    if cnt == 1 and current_video != video1:
                        if playing:
                            stop_event.set()
                            stop_event = threading.Event()
                        current_video = video1
                        playing = True
                        paused = False
                        pause_event.clear()
                        threading.Thread(target=play_video, args=(video1, stop_event, pause_event)).start()
                    elif cnt == 2 and current_video != video2:
                        if playing:
                            stop_event.set()
                            stop_event = threading.Event()
                        current_video = video2
                        playing = True
                        paused = False
                        pause_event.clear()
                        threading.Thread(target=play_video, args=(video2, stop_event, pause_event)).start()
                    elif cnt == 3:
                        if playing:
                            stop_event.set()
                            stop_event = threading.Event()
                        playing = False
                        paused = False
                        current_video = None
                    elif cnt == 5:
                        if playing:
                            if paused:
                                pause_event.clear()
                            else:
                                pause_event.set()
                            paused = not paused
                    prev = cnt
                    start_init = False

            drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27:
            if playing:
                stop_event.set()
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == "__main__":
    main()
