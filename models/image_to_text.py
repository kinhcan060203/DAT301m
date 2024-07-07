import torch
import cv2
import numpy as np
from models.craft import CRAFT
import imgproc
from utils import getDetBoxes, adjustResultCoordinates, copyStateDict
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


class Image2Text(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']=False
        config['device'] = 'cuda:0'
        self.detector = Predictor(config)
        self.craft = self._load_model()
    
    def _load_model(self):
        craft = CRAFT()
        if self.args.CUDA:
            craft.load_state_dict(copyStateDict(torch.load(self.args.MODEL_PATH)))
            craft = craft.cuda()
        else:
            craft.load_state_dict(copyStateDict(torch.load(self.args.MODEL_PATH, map_location='cpu')))
        craft.eval()

        return craft

    def get_boxes(self, image):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.args.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=self.args.MAG_RATIO)
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if self.args.CUDA:
            x = x.cuda()

        with torch.no_grad():
            y, _ = self.craft(x)



        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()
        boxes = getDetBoxes(score_text, score_link, self.args.TEXT_THRESHOLD, self.args.LINK_THRESHOLD, self.args.LOW_TEXT)
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        return boxes

    def postprocess_boxes(self, bboxes):
        pos_table = []
        row_table = {}

        for i, bbox in enumerate(bboxes):
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            if i > 0 and abs(centroid_y - pos_table[i - 1][0]) < 8:
                centroid_y = pos_table[i - 1][0]
                row_table[centroid_y].append([x0, y0, x1, y1])
            else:
                row_table[centroid_y] = [[x0, y0, x1, y1]]
            
            pos_table.append([centroid_y, centroid_x])
        return row_table

    def forward(self, img_url):
        image = imgproc.loadImage(img_url)
        image_org = image.copy()
        bboxes = self.get_boxes(image)
        row_table = self.postprocess_boxes(bboxes)
        
        x_min = np.min([item for sublist in row_table.values() for item in sublist], axis=0)[[0, 2]].min()
        k=0
        results = []
        for row in row_table.values():
            row = np.array(row)
            x0_mean, y0_mean, x1_mean, y1_mean = row[:, [0, 2]].min(), row[:, [1, 3]].min(), row[:, [0, 2]].max(), row[:, [1, 3]].max()
            bbox_img = image[int(y0_mean): int(y1_mean), int(x_min): int(x1_mean)]
            img = Image.fromarray(bbox_img)
            s = self.detector.predict(img)
            # value = self.reader.readtext(bbox_img, detail=0)
            # results.append(value[0].strip() if value else '')
            results.append(s.strip() if s else '')
            # # cv2.imwrite(f"demo_comeo/{k}.png", bbox_img)
            k+=1
            cv2.rectangle(image_org, (int(x_min), int(y0_mean)), (int(x1_mean), int(y1_mean)), color=(255, 0, 0), thickness=1)
        
        cv2.imwrite("demo_result/bbox.png", image_org)
        return results


