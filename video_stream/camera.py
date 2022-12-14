from __future__ import division
import cv2
import argparse
from models import *
from utils.utils import *
from utils.datasets import *
from oauth2client import tools
import torch
from torch.autograd import Variable
import cv2
import pathlib
from datetime import date, datetime

# FOR TELLO
from djitellopy import tello
from simple_pid import PID
from video_stream import pidTello


class Donatello(object):
    def __init__(self):
        # Donatello obj
        self.donatello = tello.Tello()

        try:
            self.donatello.connect()
            self.donatello.streamon()
            print(f'\nConnected succesfully!\n')
        except Exception as error:
            print(f'\nError trying to connect:\n{error}\n')

        # Image size (width, height)
        self.w, self.h = 1080, 720
        # self.w, self.h = 1360, 240


        # Desired range
        # self.fbRange = [40000, 80000]
        self.fbRange = [200, 400]

        # Error
        self.pError = 0

        ''' SIMPLE PID '''

        # Simple PID for Person + Color
        # Pid object for yaw
        self.pidForYaw = PID(0.18, 0.00005, 0.01, setpoint=self.w // 2)
        # yawSpeed limits
        self.pidForYaw.output_limits = (-100, 100)
        # Pid object for Z axis
        self.pidForZaxis = PID(0.8, 0.0001, 0.0001, setpoint=self.h // 2)
        # updownSpeed limits
        self.pidForZaxis.output_limits = (-100, 100)
        # Pid object for X axis
        self.pidForXaxis = PID(0.001, 0.00001, 0.0001, setpoint=(
            self.fbRange[1] + self.fbRange[0])//2)
        # forwardspeed limits
        self.pidForXaxis.output_limits = (-80, 80)


        # Simple PID for Person + Bicycle
        self.pidForYawBic = PID(0.12, 0.000001, 0.001, setpoint=self.w//2)    # Pid object for yaw
        self.pidForYawBic.output_limits = (-100, 100)      # yawSpeed limits

        self.pidForZaxisBic = PID(0.3, 0.00001, 0.0001, setpoint=self.h //
                        2)    # Pid object for Z axis
        self.pidForZaxisBic.output_limits = (-100, 100)      # updownSpeed limits

        self.pidForXaxisBic = PID(0.8, 0.00005, 0.001, setpoint=(
            self.fbRange[1] + self.fbRange[0])//2)    # Pid object for X axis
        self.pidForXaxisBic.output_limits = (-80, 80)      # forwardspeed limits


        # Default option menu
        self.optionMenu = 0

        ''' COLOR DETECTION '''

        # kernel window for morphological operations
        self.kernel = np.ones((5, 5), np.uint8)
        # Upper and lower limits for the color ORANGE in HSV color space
        self.lower_yellow = np.array([0, 135, 150])
        self.upper_yellow = np.array([20, 255, 255])

        # ARGUMENTS
        self.parser = argparse.ArgumentParser()
        # self.parser = tools.argparser.parse_args([])
        self.parser.add_argument("runserver", type=str,
                                 help="RUNSERVER DJANGO")
        self.parser.add_argument("--image_folder", type=str,
                                 default="../data/samples", help="path to dataset")
        self.parser.add_argument("--model_def", type=str,
                                 default="config/yolov3.cfg", help="path to model definition file")
        self.parser.add_argument("--weights_path", type=str,
                                 default="weights/yolov3.weights", help="path to weights file")
        self.parser.add_argument("--class_path", type=str,
                                 default="data/coco.names", help="path to class label file")
        self.parser.add_argument("--conf_thres", type=float,
                                 default=0.8, help="object confidence threshold")
        self.parser.add_argument("--webcam", type=int, default=1,
                                 help="Is the video processed video? 1 = Yes, 0 == no")
        self.parser.add_argument("--nms_thres", type=float, default=0.4,
                                 help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--batch_size", type=int,
                                 default=1, help="size of the batches")
        self.parser.add_argument("--n_cpu", type=int, default=0,
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--img_size", type=int, default=416,
                                 help="size of each image dimension")
        self.parser.add_argument("--directorio_video", type=str,
                                 help="Directorio al video")
        self.parser.add_argument("--checkpoint_model", type=str,
                                 help="path to checkpoint model")

        # Define opt
        self.opt = self.parser.parse_args()
        print()
        print('EL OPT ALV')
        print(self.opt)
        print()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.opt.model_def,
                             img_size=self.opt.img_size).to(self.device)

        # Load weights
        if self.opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            self.model.load_state_dict(torch.load(self.opt.weights_path))

        self.model.eval()
        self.classes = load_classes(self.opt.class_path)
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.a = []

        self.out = cv2.VideoWriter(f'output{datetime.timestamp(datetime.now())}.avi', cv2.VideoWriter_fourcc(*'MPEG'), 12, (self.w, self.h))

        # self.generare_frame_ctrl()


    def Convertir_RGB(self, img):
        # Convertir Blue, green, red a Red, green, blue
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        return img

    def Convertir_BGR(self, img):
        # Convertir red, blue, green a Blue, green, red
        r = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        b = img[:, :, 2].copy()
        img[:, :, 0] = b
        img[:, :, 1] = g
        img[:, :, 2] = r
        return img

    def colorDetection(self, img):
        cx, cy = -1, -1
        # Smooth the frame
        frameColorPicker = cv2.GaussianBlur(img, (11, 11), 0)
        # Convert to HSV color space
        hsv = cv2.cvtColor(frameColorPicker, cv2.COLOR_BGR2HSV)
        # Mask to extract just the yellow pixels
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        # morphological opening
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        # morphological closing
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        # Detect contours from the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

        if (len(cnts) > 0):
            # Contour with greatest area
            c = max(cnts, key=cv2.contourArea)
            # Radius and center pixel coordinate of the largest contour
            # ((x,y),radius) = cv2.minEnclosingCircle(c)
            x, y, w, h = cv2.boundingRect(c)

            cx = ((w)//2) + x
            cy = ((h)//2) + y

            # Draw the center
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        return cx, cy

    def __del__(self):
        # Release camera video
        self.donatello.land()

    # Generate frame with vision and control
    def generare_frame_ctrl(self):
        while True:
            # Get frames from Tello cam
            frame = self.donatello.get_frame_read().frame
            frame = cv2.resize(frame, (self.w, self.h),
                            interpolation=cv2.INTER_CUBIC)
            self.out.write(frame)

            # For color detection:
            colorCenterX, colorCenterY = self.colorDetection(frame)

            # LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
            RGBimg = self.Convertir_RGB(frame)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(self.Tensor))

            # Keyboard Control
            '''
            lr, fb, ud, yv, optionMenu = keyControl.getKeyboardInput(
                self.donatello, optionMenu, frame)
            if (optionMenu == 0):
                self.donatello.send_rc_control(lr, fb, ud, yv)
            '''

            # YOLO Person + ColorTracking
            if (self.optionMenu == 1):
                with torch.no_grad():
                    detections = self.model(imgTensor)
                    detections = non_max_suppression(
                        detections, self.opt.conf_thres, self.opt.nms_thres)

                for detection in detections:
                    if detection is not None:
                        detection = rescale_boxes(
                            detection, self.opt.img_size, RGBimg.shape[:2])
                        for x1, y1, x2, y2, conf, _, cls_pred in detection:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            color = [int(c) for c in self.colors[int(cls_pred)]]
                            # Only for person detection
                            if cls_pred == 0:
                                # print("Se detect?? {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                                # frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                                cv2.rectangle(frame, (x1, y1 + box_h),
                                            (x2, y1), color, 3)
                                # Nombre de la clase detectada
                                cv2.putText(
                                    frame, self.classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                                # Certeza de prediccion de la clase
                                cv2.putText(frame, str("%.2f" % float(
                                    conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)
                                # area = int(box_w*box_h)
                                area = int(box_h)
                                c = [int(x1+box_w//2), int(y1+box_h//2)]
                                info = [c, area]
                                # print("c: {}, {}     area: {}".format(c[0], c[1], area))
                                # Shows the center of the rectangle/face
                                cv2.circle(frame, (c[0], c[1]),
                                        5, (0, 255, 0), cv2.FILLED)

                                # Some references:
                                # Shows the center of the frame
                                cv2.circle(frame, (self.w//2, self.h//2), 5,
                                        (255, 0, 0), cv2.FILLED)
                                cv2.rectangle(frame, (self.w//4, self.h//4),
                                            (3*self.w//4, 3*self.h//4), (255, 0, 0), 2)

                                # Color detection:
                                booleanExpresion = colorCenterX > int(x1) and colorCenterX < int(
                                    x2) and colorCenterY > int(y1) and colorCenterY < int(y2)
                                # print("orangeCenter: ", colorCenterX, ", ", colorCenterY)
                                # print(booleanExpresion)
                                if booleanExpresion:
                                    print("Persona con color naranja")
                                    cv2.circle(
                                        frame, (c[0], c[1]), 15, (0, 0, 255), cv2.FILLED)
                                    pidTello.track(
                                        self.donatello, self.pidForYaw, self.pidForZaxis, self.pidForXaxis, info, self.fbRange)
                                else:
                                    print("No hay persona con color naranja")
                                    self.donatello.send_rc_control(0, 0, 0, 0)
                            else:
                                print("No se detecto persona")
                                self.donatello.send_rc_control(0, 0, 0, 0)

            if(self.optionMenu == 2):
                with torch.no_grad():
                    detections = self.model(imgTensor)
                    detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

                existsPerson = False
                existsBicycle = False
                infoPerson = []
                centerPersons = []
                areaPersons = []

                infoBicycle = []
                centerBicycles = []
                areaBicycles = []
                xValuesBic = []

                for detection in detections:
                    if detection is not None:
                        detection = rescale_boxes(detection, self.opt.img_size, RGBimg.shape[:2])
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            color = [int(c) for c in self.colors[int(cls_pred)]]
                            # Only for person detection

                            # Person
                            if cls_pred == 0:
                                existsPerson = True
                                arPer = int(box_w*box_h)
                                cenPer = [int(x1+box_w//2), int(y1+box_h//2)]
                                centerPersons.append(cenPer)
                                areaPersons.append(arPer)
                                infoPerson.append([cenPer, arPer])
                            # Bicycle
                            if cls_pred == 1:
                                existsBicycle = True
                                # areaBic = int(box_w*box_h)
                                areaBic = int(box_h)
                                cenBic = [int(x1+box_w//2), int(y1+box_h//2)]
                                centerBicycles.append(cenBic)
                                areaBicycles.append(areaBic)
                                infoBicycle.append([cenBic, areaBic])
                                xValuesBic.append([int(x1), int(x2)])


                            if cls_pred == 1 or cls_pred == 0:
                                # print("Se detect?? {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(self.classes[int(cls_pred)], x1, y1, x2, y2))
                                frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                                cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 2)
                                cv2.putText(frame, self.classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)# Nombre de la clase detectada
                                cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2) # Certeza de prediccion de la clase
                                
                                area = int(box_w*box_h)
                                c = [int(x1+box_w//2), int(y1+box_h//2)]
                                info = [c, area]
                                # print("c: {}, {}     area: {}".format(c[0], c[1], area))
                                cv2.circle(frame, (c[0], c[1]), 5, (0, 255, 0), cv2.FILLED) # Shows the center of the rectangle/face

                                # Some references:
                                cv2.circle(frame, (self.w//2, self.h//2), 5, (200, 0, 255), cv2.FILLED) # Shows the center of the frame
                                cv2.rectangle(frame, (self.w//4, self.h//4), (3*self.w//4, 3*self.h//4), (200, 0, 255), 2)
                            else:
                                print("No se detecto persona u bici")
                                # print("Se detect?? {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(self.classes[int(cls_pred)], x1, y1, x2, y2))

                        # Checks if the two objects exists in the same frame
                        isPair = existsPerson & existsBicycle
                        if isPair:
                            # Look for the biggest bicycle
                            i = areaBicycles.index(max(areaBicycles))
                            j = -1

                            for index in range(len(centerPersons)):
                                if centerPersons[index][0] > xValuesBic[i][0] and centerPersons[index][0] > xValuesBic[i][0]:
                                    j = index
                                    print("Persona y bici ALV ALV ALV")
                                    print(centerPersons[j])
                                    cv2.circle(frame, (centerBicycles[i][0], centerBicycles[i][1]), 20, (120, 0, 200), cv2.FILLED) # Shows the center of the bicycle
                                    cv2.circle(frame, (centerPersons[j][0], centerPersons[j][1]), 20, (0, 0, 255), cv2.FILLED) # Shows the center of the tracked person
                                    infoForPID = [(centerBicycles[i][0], centerBicycles[i][1]), areaBicycles[i]]
                                    pidTello.track(
                                        self.donatello, self.pidForYawBic, self.pidForZaxisBic, self.pidForXaxisBic, infoForPID, self.fbRange)
                                    print("Area Bici: ", areaBicycles[i])
                                    break
            print("\nOPTION MODE: \n", self.optionMenu)

            # Convert to BRG
            #return self.Convertir_BGR(RGBimg)
            frame = self.Convertir_BGR(RGBimg)

            # cv2.imshow('frame', self.Convertir_BGR(RGBimg))
        
            # Resize with defined width and height
            frame = cv2.resize(frame, (self.w, self.h))

            # Flip
            frame_flip = cv2.flip(frame, 1)

            # Encode image and convert it to 'jpg' format
            _, img = cv2.imencode('.jpg', frame_flip)
            
            return img.tobytes()
        

    def get_frame(self):
        # Get frame from camera
        frame = self.donatello.get_frame_read().frame
        # Resize with defined width and height
        frame = cv2.resize(frame, (self.w, self.h))
        # Flip
        frame_flip = cv2.flip(frame, 1)
        # Encode image and convert it to 'jpg' format
        _, img = cv2.imencode('.jpg', frame_flip)
        return img.tobytes()

    # Move right
    def move_right(self):
        self.donatello.move_right(35)

    # Move left
    def move_left(self):
        self.donatello.move_left(35)

    # Move forward
    def move_forward(self):
        self.donatello.move_forward(35)

    # Move backward
    def move_backward(self):
        self.donatello.move_back(35)

    # Move up
    def move_up(self):
        self.donatello.move_up(35)

    # Move down
    def move_down(self):
        self.donatello.move_down(35)