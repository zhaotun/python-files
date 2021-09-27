"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.15
IOU_THRESHOLD = 0.45
int_box=[0,0,0,0]
int_box1=[0,0,0,0]
fps1=0.0

color_categories = [
        (0,111,255),   # pedestrian 
        (255,0,0),     # people
        (100,100,100), # bicycle
        (0,0,255),     # car
        (0,255,0),     # van
        (255,255,0),   # truck
        (0,100,100),   # tricycle
        (100,100,0),   # awning-tricycle
        (100,0,100),   # bus
        (211,0,100)    # motor
    ]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = ( line_thickness or round(0.00003 * (img.shape[0] + img.shape[1]) / 2) + 1 )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                #self.input_w = engine.get_binding_shape(binding)[-1]
                #self.input_h = engine.get_binding_shape(binding)[-2]

                self.input_w = INPUT_W
                self.input_h = INPUT_H

                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, input_image_path):
        global int_box,int_box1,fps1
        #threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            input_image_path
        )

        # Copy input image to host buffer
        #np.copyto(host_inputs[0], batch_input_image.ravel())
        np.copyto(host_inputs[0], input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        #context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess

        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w
        )

        '''
        # Draw rectangles and labels on the original image
        for i in range(len(result_boxes)):
            box1 = result_boxes[i]
            plot_one_box(
                box1,
                image_raw,
                color=color_categories[i],
                label="{}:{:.2f}".format(
                    categories[int(result_classid[i])], result_scores[i]
                ),
            )
        '''

        return end - start, result_boxes, result_scores, result_classid

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        y = np.zeros_like(x)

        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def nms_mxnet(self, boxes, scores, thresh):
        """
        greedily select boxes with high confidence and overlap with current maximum <= thresh
        rule out overlap >= thresh
        :param dets: [[x1, y1, x2, y2 score]]
        :param thresh: retain overlap < thresh
        :return: indexes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        #scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        #pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        #indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        indices = self.nms_mxnet(boxes, scores, thresh=IOU_THRESHOLD)

        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        return result_boxes, result_scores, result_classid

class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
 
    def run(self):
        self.func(*self.args)


if __name__ == "__main__":

    # load custom plugins
    PLUGIN_LIBRARY   = "build/libmyplugins.so"
    engine_file_path = "build/visdrone-yolov5-640-e300-fp32.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load visdrone labels
    categories = ["pedestrian", "people","bicycle","car","van", "truck","tricycle","awning-tricycle","bus","motor"]
    
    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    try:
        # infer a video
        # 读取视频
        #video_path = '/home/nvidia/deepstream_lpr/deepstream_lpr_app-master/deepstream-lpr-app/d1-1080p.mp4'
        #video_path = 'rtsp://192.168.1.220/test'
        video_path = 'rtsp://192.168.1.100:554/majorAV'
        cap = cv2.VideoCapture(video_path)

        
        # 保存视频
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        out = cv2.VideoWriter('result-d4.mp4', fourcc, fps, (width, height))  # 写入视频

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                
                start = time.time()
                use_time, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(frame)
                print('infer time->{:.2f}ms...'.format( use_time * 1000))
                print('infer a image cost time:', time.time()-start)
                
                # Draw rectangles and labels on the original image
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    plot_one_box(
                            box,
                            frame,
                            color_categories[int(result_classid[j])], # color
                            label="{}:{:.2f}".format(
                                categories[int(result_classid[j])], result_scores[j]
                            ),
                    )

                out.write(frame)  # 写入帧
                # Display the resulting frame
                cv2.imshow('Frame',frame)
                
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()
        #out.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
