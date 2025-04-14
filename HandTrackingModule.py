import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        if len(self.lmList) == 0:
            return []
        # Thumb
        fingers.append(1 if self.lmList[4][1] > self.lmList[3][1] else 0)
        # Other fingers
        for id in range(1, 5):
            fingers.append(1 if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2] else 0)
        return fingers







































