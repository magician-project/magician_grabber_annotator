import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
import sys
import io

class SAMProcessor:
    def __init__(self, sam_checkpoint, model_type, device="cuda"):
        sys.path.append("..")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.selectedAreas = list()
        self.masks = None
        self.image = None
        self.foregroundMask  = None
        self.foregroundImage = None


    def mouse_callback(self, event, x, y, flags, param):
        global newlySelectedAreas
        masks = param["masks"]
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at coordinates (x: {x}, y: {y})")
            if len(masks) > 0:
                for i, mask in enumerate(masks):
                    segmentationFlag = mask['segmentation'][y][x]
                    if segmentationFlag:
                        print("#Area %u X/Y (%u/%u)" % (i, x, y))
                        newlySelectedAreas.append(i)

    def overlay_mask(self, image, mask, alpha=0.5, true_color=(0, 255, 0)):
        overlay = np.zeros_like(image)
        overlay[mask] = true_color
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return result

    def process_mask(self, mask, kernel_size=5, iterations=2, min_blob_area=1500):
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)

        for i in range(1, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] < min_blob_area:
                processed_mask[labels == i] = 0

        processed_mask_bool = (processed_mask > 0)
        return processed_mask_bool

    def union(self, array1, array2):
        return np.logical_or(array1, array2)

    def save_mask(self, filename, mask):
        vis_image = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
        vis_image[mask] = 255
        cv2.imwrite(filename, vis_image)

    def show_image_with_callback(self, path, image, masks):
        global newlySelectedAreas
        selectedAreas = list()

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback, {"masks": masks})

        foregroundMask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        imageToShow = image

        while True:
            if len(newlySelectedAreas) > 0:
                for newArea in newlySelectedAreas:
                    selectedAreas.append(newArea)
                    foregroundMask = self.union(foregroundMask, masks[newArea]['segmentation'])

                foregroundMask = self.process_mask(foregroundMask, kernel_size=5, iterations=2)
                imageToShow = self.overlay_mask(image, foregroundMask)

                self.save_mask("%s_foreground.png" % path, foregroundMask)
                newlySelectedAreas = []  # Flush UI input

            cv2.imshow('Image', imageToShow)
            key = cv2.waitKey(15) & 0xFF

            if key == 27:  # Press 'Esc' to exit
                break

        cv2.destroyAllWindows()

    def show_anns(self, anns, path):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        plt.imsave(path, img)
        fig = plt.figure()
        return fig


    def figtoimg2(self,fig):
        with io.BytesIO() as buff:
            plt.savefig(buff, format='raw')
            buff.seek(0)
            img = plt.imread(buff)

            #data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            
        #w, h = fig.canvas.get_width_height()
        #img = data.reshape((int(h), int(w), -1))
        return img


    def figtoimg(self,fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = data.reshape((int(h), int(w), -1))
        return img


    def select_area(self,x,y):
        if (len(self.masks)>0):
           for i,mask in enumerate(self.masks):
             segmentationFlag = mask['segmentation'][y][x]
             if (segmentationFlag):
                newArea = i
                print("#Area %u X/Y (%u/%u)" % (i,x,y))
                self.selectedAreas.append(newArea)
                self.foregroundMask  = self.union(self.foregroundMask , self.masks[newArea]['segmentation'])
                self.foregroundMask  = self.process_mask(self.foregroundMask, kernel_size=5, iterations=2)
                #self.foregroundImage = self.overlay_mask(self.image, self.foregroundMask)          #<- base on input image
                self.foregroundImage = self.overlay_mask(self.foregroundImageOriginal, self.foregroundMask,alpha=0.8) #<- base on segmentation image

          #self.save_mask("%s_foreground.png"%path ,self.foregroundMask)  

    def process_image(self, image):
        self.selectedAreas = list()
        self.foregroundMask = np.zeros((image.shape[0], image.shape[1] ), dtype=bool)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:        
          self.masks = self.mask_generator.generate(self.image)
          fig = self.show_anns(self.masks, f"tmp.png")
          self.foregroundImageOriginal = cv2.imread("tmp.png") #self.figtoimg(fig) 
          self.foregroundImage = self.foregroundImageOriginal  
        except:
          print("Failed segmenting, returning input")
          self.masks = dict()
          self.foregroundImageOriginal = self.image 
          self.foregroundImage = self.foregroundImageOriginal   

        return self.foregroundImage



if __name__ == "__main__":
  sam_processor = SAMProcessor(sam_checkpoint="sam_vit_l_0b3195.pth", model_type="vit_l", device="cuda")
 
  for index, arg in enumerate(sys.argv[1:], start=1):
        print(f"Argument {index}: {arg}")
        sam_processor.process_path(arg)




