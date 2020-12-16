"""ssd.py
This module implements the TrtSSD class.
"""
import ctypes
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import openpifpaf
import torch
import PIL
import sys 
import time
import cv2 as cv 

class PoseEstimator():

    def _load_engine(self):
        TRTbinPath = self.trt_bin_path
        with open(TRTbinPath, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    #def _create_context(self):
    def _allocate_buffers(self):
        summ = 0
        for i in range(self.engine.num_bindings):
            binding = self.engine[i]
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            summ = summ + size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return

    def __init__(self, trt_bin_path, model_input_size):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.trt_bin_path = trt_bin_path
        self.model = ''
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.model_input_size = model_input_size
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = [] 
        self.cuda_outputs = []
        self.bindings = [] 
        self._init_cuda_stuff()

    def _init_cuda_stuff(self):
        cuda.init()
        self.device = cuda.Device(0)  # enter your Gpu id here
        self.cuda_context = self.device.make_context()
        self.engine = self._load_engine()
        self._allocate_buffers()
        self.engine_context = self.engine.create_execution_context()
        self.stream = cuda.Stream()  # create a CUDA stream to run inference        

    def __del__(self):
        """ Free CUDA memories. """
        for mem  in self.cuda_inputs:
            mem.free()
        for mem in self.cuda_outputs:
            mem.free

        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.engine
        del self.bindings
        del self.host_inputs
        del self.host_outputs


    def inference(self, img):
        """Estimate human poses in the input image."""

        img = cv.resize(img, self.model_input_size)
        pil_im = PIL.Image.fromarray(img)
        preprocess = None
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False,
            pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        for images_batch, _ , __ in loader:
            np_img = images_batch.numpy() # np_img.size = (0,3,641,641)
            
        self.host_inputs[0] = np.ravel(np.zeros_like(np_img)) #np.ravel(np.zeros_like(np_img))
        
        self.cuda_context.push() 
        start_time = time.time()

        np.copyto(self.host_inputs[0], np.ravel(np_img))
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.stream.synchronize()

               
        self.engine_context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
               
        pif_names=['pif_c', 'pif_r', 'pif_b', 'pif_s']
        paf_names=['paf_c','paf_r1','paf_r2','paf_b1','paf_b2']
        other_names=['616','646','648','637','644']
        pif=[None]*4
        paf=[None]*5
        other=[None]*5
        for i in range(1, self.engine.num_bindings): #int(len(self.host_outputs)/2) +1):
            cuda.memcpy_dtoh_async(
                self.host_outputs[i - 1], self.cuda_outputs[i - 1], self.stream)

        self.stream.synchronize()
        print("--- %s seconds ---" % (time.time() - start_time))

        for i in range(1, self.engine.num_bindings): #int(len(self.host_outputs)/2) +1):
            shape = self.engine.get_binding_shape(i)
            name = self.engine.get_binding_name(i)
            total_shape = np.prod(shape)
            output = self.host_outputs[i - 1][0: total_shape]
            output = np.reshape(output, tuple(shape))
            if name in pif_names:
                index_n = pif_names.index(name)
                pif[index_n] = torch.from_numpy(output)
            elif name in paf_names:
                index_n = paf_names.index(name)
                paf[index_n] = torch.from_numpy(output)
            elif name in other_names:
                index_n = other_names.index(name)
                other[index_n] = torch.from_numpy(output)

        
        final_output = [pif, paf, other]
        self.cuda_context.pop()
        
        return final_output
        
    
