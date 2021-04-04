#IMPORTS
import cv2

#CARREGA CLASSES
class_names = []

with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]


class Remote:
    def __init__(self, qtd):
        self.qtd = qtd

    def increment(self):
        self.qtd += 1
        return self.qtd


class Cup:
    def __init__(self, qtd):
        self.qtd = qtd

    def increment(self):
        self.qtd += 1
        return self.qtd


class Cell_phone:
    def __init__(self, qtd):
        self.qtd = qtd

    def increment(self):
        self.qtd += 1
        return self.qtd

cels_pass = []
rem_pass = []
c_pass = []

remote_pass = 0
cup_pass = 0
cell_phone_pass = 0
aux_cup = 0
aux_rem = 0
aux_cell = 0

rem = Remote(remote_pass)
cup = Cup(cup_pass)
cel = Cell_phone(cell_phone_pass)

#CARREGA VIDEO
cap = cv2.VideoCapture(0)

fr_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fr_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr_fps = int(cap.get(cv2.CAP_PROP_FPS))


#CARREGA REDE NEURAL
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#SETAR PARÂMETROS NA REDE
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

#RODAR VIDEO
while (cap.isOpened()):
    #PEGAR FRAME DO VIDEO
    _, frame = cap.read()

    x, y, _ = frame.shape

    # print(frame.shape)
    division_x = int(x / 2)
    division_y = int(y / 2)

    #print(fr_height, fr_width)

    #DETECÇÂO
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    half = fr_width / 2
    print(half)

    #PERCORRER TODAS DETECÇÕES

    for (classId, score, box) in zip(classes, scores, boxes):
        label = f"{class_names[classId[0]]}"
        rec = (box[0])

        cels_pass_m = []
        rem_pass_m = []
        c_pass_m = []

        if rec > division_x:
            if label == 'cup' and aux_cup == 0:
                cup_pass = cup.increment()
                rec = (box[0])
                c_pass.append(1)
                aux_cup += 1
            elif label == 'remote' and aux_rem == 0:
                rec = (box[0])
                remote_pass = rem.increment()
                rem_pass.append(1)
                aux_rem += 1
            elif label == 'cell phone' and aux_cell == 0:
                rec = (box[0])
                cell_phone_pass = cel.increment()
                cels_pass_m.append(1)
                cels_pass.extend(cels_pass_m)
                aux_cell += 1

        print(cels_pass, rem_pass, c_pass)
        print('crossed_objects: Cels: {}, Rem: {}, Cup: {}'.format(sum(cels_pass),sum(rem_pass), sum(c_pass)))

        cv2.rectangle(frame, box, (255, 0, 0), 2)
        cv2.putText(frame, label, (box[0], box[1]-10), 0, 0.5, (255,0,0), 2)

    color = (0, 255, 0)
    thickness = 2
    cv2.imshow('frame', frame)
    cv2.line(frame, (division_y, x), (division_y, 1), color, thickness)
    cv2.putText(frame, 'remote_pass = {}'.format(remote_pass), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, 'cell_phone_pass = {}'.format(cell_phone_pass), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(frame, 'cup_pass = {}'.format(cup_pass), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()