
# imports
# -------

# imports
# -------
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


# torch related imports
# ---------------------
import torch
from torch.autograd import Variable
import torch.nn as nn


# # code

# In[17]:


# wrapper function to call final objective on an image
# will perform all CV tasks
# ----------------------------------------------------
def process_image(task, img):
    
    '''
    1. takes in task & image as input - 
    
    task can be - facemask, seg
    
    '''
    
    
    if task == 'facemask':
        
        
        # 1. loading mask_nomask model
        # -----------------------------
        try:
            model_path = os.getcwd() + '/models/'
            base_name = 'eplane_mask_nomask_grayscale'
            model_mask = load_saved_model_function_cpu(model_path + base_name + '.tar')
        except:
            assert 1==2,'Error: cannot find model to run! Please check if model is in path /models/'
            model_mask = None


        # 2. call detail function
        # -----------------------
        d, boxed_image, faces_without_mask_curr_image = return_faces_in_image_opencv_dnn(img,model_mask)
        #print('total number of faces extracted: ' + str(len(d)))
        #print('total faces WITHOUT mask: ' + str(faces_without_mask_curr_image))


        # 3. final return
        # ---------------
        return boxed_image
    
    
    elif task == 'seg':
        
        # perform segmentation task on image
        # ----------------------------------
        
        # 1. loading seg model
        # --------------------
        try:
            model_path = os.getcwd() + '/models/'
            base_name = 'eplane_street_segmentation_22classes'
            model_seg = load_saved_model_function_cpu(model_path + base_name + '.tar')
            model_seg = model_seg.eval()
        except:
            assert 1==2,'Error: cannot find model to run! Please check if model is in path /models/'
            model_seg = None
        
    
        # 2. calling funtion
        # ------------------
        overlayed_image = return_streetview_segmentation(img, model_seg)
        
        # final return
        # ------------
        return overlayed_image
    
    elif task == 'depth':

        # return_perception(img_in_BGR, model)
        # use the abve function

        # 1. loading seg model
        # --------------------
        try:
            model_path = os.getcwd() + '/models/'
            base_name = 'eplane_depth_perception'
            model_dep = load_saved_model_function_cpu(model_path + base_name + '.tar')
            model_dep = model_dep.eval()
        except:
            assert 1==2,'Error: cannot find model to run! Please check if model is in path /models/'
            model_dep = None
        
    
        # 2. calling funtion
        # ------------------
        depth_image = return_perception(img, model_dep)
        
        # final return
        # ------------
        return depth_image

    
    else:
        
        # invalid task
        # ------------
        assert 1==2,"Error: Invalid task"
    


# In[18]:


# make this usable across all CV tasks like segmentation / depth perception etc
# a generice wrapper function to read video from webcam
# -----------------------------------------------------

def process_video(task, in_video):
    
    '''
    
    1. generic function that can run all CV tasks from CAM or camera
    2. in_video can be 'cam' or video_file_URL
    3. takes in task as input - 
    
    task can be - facemask, seg
    
    '''
    
     
    # lets use split this video function by camera / video in
    # its a extended approch - but let me do this to keep things seperate
    ##
    
    
    # 0. if else
    # ----------
    if in_video == 'cam':
        
        # 0. reading from CAM
        # --------------------
        cap = cv2.VideoCapture(0)
        
        # 1. frame by frame op
        # --------------------
        while(True):

            # 1. ret is True or Fales - indicates end of video in uploaded files
            # ----------------------------------------------------------------
            ret, frame = cap.read()
            

            # 2. getting FPS
            # ---------------
            fps = cap.get(5)

            # 3. main operation on frame (which is a BGR image) comes here
            # ------------------------------------------------------------
            read_image = process_image(task, frame)
            cv2.putText(read_image, 'FPS: ' + str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            cv2.imshow('frame',cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB))


            # 4. break the read
            # -----------------
            if cv2.waitKey(1) & 0xFF == ord('q'):

                # user in for breaking live feed
                # ------------------------------
                break


        # 2. outside while - when everything done, release the capture
        # ------------------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()

        # 3. final return 
        # ---------------
        # nothing
        
    
    else:
        
        # 0. user has input video file
        # ----------------------------
        cap = cv2.VideoCapture(in_video)
        
        # not taking EFFECT AT ALL
        ##########################
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,300) # width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,300) # height
        cap.set(cv2.CAP_PROP_FPS,12) # FPS
        

        # 0.1 check if file there
        # -----------------------
        if (cap.isOpened()== False): 
            
            # not proceeding further
            # -----------------------
            assert 1==2, "Error opening video stream or file. Check file URL and format."
            
        
        # 1. frame by frame op
        # --------------------
        while(cap.isOpened()):


            # 1. ret is True or Fales - indicates end of video in uploaded files
            # ----------------------------------------------------------------
            ret, frame = cap.read()
            
            # 2. processing until we reach end of video
            # -----------------------------------------
            if ret == True:
                
                                
                # 2. getting FPS
                # ---------------
                fps = cap.get(cv2.CAP_PROP_FPS) #5
                frame_h, frame_w = frame.shape[0], frame.shape[1]

                # 3. main operation on frame (which is a BGR image) comes here
                # ------------------------------------------------------------
                read_image = process_image(task,frame.astype('uint8'))
                read_image = cv2.resize(read_image,(frame_w, frame_h))
                cv2.putText(read_image, 'FPS: ' + str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                cv2.imshow('frame',cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB))
                
                # If the FPS is equal to 20, then you should wait 0,05 seconds between displaying the consecutive frames. 
                # So just put waitKey(50) after imshow() in order to have the desired speed for the playback.
                # ------------
                # ------------
                #cv2.waitKey(1)
                
             
                # 2. breaking
                # -----------
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            else:

                # breaking video in
                # as video seems to have ended
                # ----------------------------
                break
        
        
        # 2. outside while - when everything done, release the capture
        # ------------------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()

        # 3. final return 
        # ---------------
        # nothing
    
        
   
    
    


# ### 1. helper functions

# In[19]:


# GENERIC - change an torch image to numpy image
# ----------------------------------------------
def to_numpy_image(xin):
    
    try:
        xin = xin.data.numpy()
    except:
        xin = xin.numpy()
    
    xout = np.swapaxes(xin,1,2)
    xout = np.swapaxes(xout,2,3)
    
    # returns axes swapped numpy images
    # ---------------------------------
    return xout       



# GENERIC - converts numpy images to torch tensors for training
# -------------------------------------------------------------
def setup_image_tensor(xin):
    xout = np.swapaxes(xin,1,3)
    xout = np.swapaxes(xout,2,3)
    
    # returns axes swapped torch tensor
    # ---------------------------------
    xout = torch.from_numpy(xout)
    return xout.float()


# In[20]:


# function to return color map
# ----------------------------

def return_colormap(x):
    
    '''
    
    1. x is in form (m,h,w,no_channels) - numpy array
    2. returns (m,h,w,1) - with index positions
    3. returns color maps based on index positions
    
    '''
    
    # 0. main color code legend here
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # ------------------------------
    color_legend = {}
    color_legend[0] = np.array([255,0,0]) # red
    color_legend[1] = np.array([0,255,0]) # lime
    color_legend[2] = np.array([0,0,255]) # blue
    color_legend[3] = np.array([255,255,0]) # yellow
    color_legend[4] = np.array([0,255,255]) # cyan/aqua
    color_legend[5] = np.array([255,0,255]) # magenta
    color_legend[6] = np.array([192,192,192]) # silver
    color_legend[7] = np.array([255,0,0]) # olive
    color_legend[8] = np.array([0,128,0]) # green
    color_legend[9] = np.array([128,0,128]) # purple
    color_legend[10] = np.array([0,128,128]) # teal
    color_legend[11] = np.array([0,0,128]) # navy
    color_legend[12] = np.array([255,140,0]) # orange
    color_legend[13] = np.array([143,188,143]) # dark sea green
    color_legend[14] = np.array([70,130,180]) # steel blue
    color_legend[15] = np.array([255,20,147]) # pink
    color_legend[16] = np.array([245,222,179]) # wheat
    color_legend[17] = np.array([139,69,19]) # brown
    color_legend[18] = np.array([210,105,30]) # chocolate
    color_legend[19] = np.array([255,250,240]) # floral white
    color_legend[20] = np.array([105,105,105]) # dark gray
    color_legend[21] = np.array([105,105,105]) # dark gray
    color_legend[22] = np.array([105,105,105]) # dark gray
    color_legend[23] = np.array([105,105,105]) # dark gray
    
    # 1. inits
    # --------
    out = np.zeros((x.shape[0],x.shape[1],x.shape[2],3))
    
    # 2. max
    # ------
    max_out = np.argmax(x,3)    
    
    # 3. iter over each key of legend and assign values
    # -------------------------------------------------
    for keys in color_legend:
        
        # j = ((m==2).reshape(2,3,2,1) * np.array([8,8,8])) + j
        # -----------------------------------------------------
        out += (max_out == keys).reshape(max_out.shape[0], max_out.shape[1], max_out.shape[2], 1) * color_legend[keys]
    
    
    # 4. return
    # ---------
    return out.astype('uint8')
        


# In[21]:


# function to return dict of faces given a single image
# using opencv inbuilt dnn
# trained simple classifier for mask/no_mask detection
# -----------------------------------------------------

def return_faces_in_image_opencv_dnn(img, model_mask):
    
    '''
    
    0. img in a numpy array of shape (h,w,c) in RGB - THIS IS A MUST AS THIS AFFECTS MODEL ACCURACY
    1. simply processes the image > crops faces > returns dict of faces of varying size
    2. ensure prototxt_url, model_url are in code file path
    
    NOTE:
    set threshold face confidence to LOW value for dnn to pick faces with mask
    
    # alternate model in case required
    # --------------------------------
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # properties of classifier
    # ------------------------
    # mask_nomask classifier has 2 outputs of order ['without_mask_faces', 'with_mask_faces']
    # takes in 127x127x3 image as input
    # NOT optimising for GPU for now / for now will run on CPU

    
    '''
    

    
    # 0. inits
    # --------
    d = {}
    confidence_threshold = 0.2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_counter = 0
    m_h,m_w = 127,127
    faces_without_mask = 0
    
    
    # 0.1 loading model - opencv dnn
    # ------------------------------
    parent_url = os.getcwd() + '/opencv_dnn_files/'
    prototxt_url = parent_url + 'deploy.prototxt.txt'
    model_url = parent_url + 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt_url, model_url)
    
    
    # 1.
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (104,177,123) -- this is required for optimal perfomance of the model
    # input image must be in BGR format and NOT RGB
    # ----------------------------------------------------------------------
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    

    # 2.
    # pass the blob through the network and obtain the detections and predictions
    # detections is of format ( , , no_faces, )
    # ---------------------------------------------------------------------------
    net.setInput(blob)
    detections = net.forward()
    
    
    # 3. looping over detections and keeping only really confident values
    # --------------------------------------------------------------------
    for i in range(0,detections.shape[2]):

        # 1. extract the confidence associated with the prediction
        # --------------------------------------------------------
        confidence = detections[0, 0, i, 2]

        # 2. keeping detection over a certain threshold of confidence only
        # ----------------------------------------------------------------
        if confidence >= confidence_threshold:


            # proceeding only if detections are legitimate
            # --------------------------------------------
            if detections[0, 0, i, 3:7][0] < 1 and detections[0, 0, i, 3:7][1] < 1 and detections[0, 0, i, 3:7][2] < 1 and detections[0, 0, i, 3:7][3] < 1:

                # 1. incrementing face counter
                # ----------------------------
                face_counter += 1


                # 2. compute the (x, y)-coordinates of the bounding box for the object
                # --------------------------------------------------------------------
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # putting this in try since due to low confidence the bbox co-ords could be in small range
                # -----------------------------------------------------------------------------------------
                try:
                    
                    # 3. getting the face cropped for dict
                    # ------------------------------------
                    d[face_counter] = cv2.cvtColor(img[startY:endY,startX:endX,:], cv2.COLOR_BGR2RGB)
                    
                    # 4. some default bbx print settings
                    # tuple (BGR)
                    # ----------------------------------
                    #text = "{:.2f}%".format(confidence * 100)
                    text = 'ok'
                    box_color_tuple = (0, 255, 0)

                    
                    # check if face is masked or not
                    # ------------------------------
                    if model_mask != None:
                        
                        # model mask is input
                        # time to classify each and evey face for mask vs no_mask
                        # -------------------------------------------------------
                        input_face = cv2.cvtColor(cv2.resize(d[face_counter],(m_w,m_h)),cv2.COLOR_RGB2GRAY)
                        input_face = input_face.reshape(1,m_h,m_w,1)
                        input_face_tensor = Variable(setup_image_tensor(input_face)).float()
                        input_face_tensor = input_face_tensor/torch.max(input_face_tensor)
                        prediction = model_mask.eval()(input_face_tensor)
                        
                        # check if face is WITHOUT mask
                        # setting the threshold lower so as to NOT miss a face without mask
                        # may lead to false alarm - but thats ok
                        # ['without_mask_faces', 'with_mask_faces']
                        # -----------------------------------------------------------------
                        if prediction[0,0] >= 0.2:
                            
                            # with mask
                            # ---------
                            text = 'not ok'
                            #box_color_tuple = (0, 0, 255)
                            box_color_tuple = (255, 0, 0)
                            faces_without_mask += 1


                    # 4. draw the bounding box of the face along with the associated probability
                    # --------------------------------------------------------------------------
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(img, (startX, startY), (endX, endY),box_color_tuple, 2)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color_tuple, 2)
                
                except:
                    
                    pass
    
    
    # 4. out of foop loop - looping through all faces
    # return image is BGR
    # -----------------------------------------------
    final_image = img
    #final_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return d, final_image, faces_without_mask



# In[22]:


# function return street view segmentation
# ----------------------------------------
def return_streetview_segmentation(img_in, model):
    
    '''
    
    1. a simple forward pass on UNET FCN model
    2. will use custom color map function to return a nice overlay
    3. img is in RGB format
    
    '''
    
    # 0. inits
    # ---------
    orig_h, orig_w = img_in.shape[0], img_in.shape[1]
    h,w = 255, 255
    #img = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB) # not required as input is RGB
    img = cv2.resize(img_in, (w, h))
    img = img.reshape(1,h,w,3)
    
    img_trn = Variable(setup_image_tensor(img)).float()
    img_trn = img_trn/torch.max(img_trn)
    
    # 1. forward pass
    # ---------------
    model_out = model.eval()(img_trn)
    
    # 2. convrt to colormap
    # ---------------------
    x_target_model_out_colormap = return_colormap(to_numpy_image(model_out))
    
    # 3. overlap
    # ----------
    overlayed = img[0] * 0.5 + x_target_model_out_colormap[0] * 0.5
    overlayed = cv2.resize(overlayed, (orig_w, orig_h))
    
    # 4. return
    # return image is RGB
    # -------------------
    return overlayed.astype('uint8')

    
# depth perception
# ----------------
# function to return depth perception
# -----------------------------------
# function to return depth perception
# -----------------------------------

def return_perception(img_in, model):
    
    '''
    
    1. simple forward pass and regress depth
    2. still not sure how to perceive this here
    
    '''
    
    # 0. inits
    # ---------
    orig_h, orig_w = img_in.shape[0], img_in.shape[1]
    h,w = 255, 255
    img = cv2.resize(img_in, (w, h))
    img = img.reshape(1,h,w,3)
    img_trn = Variable(setup_image_tensor(img)).float()
    img_trn = img_trn/torch.max(img_trn)
    
    # 1. forward pass
    # ---------------
    model_out = model.eval()(img_trn)
    
    # 2. simple np ops
    # ----------------
    model_out_np = to_numpy_image(model_out)
    model_out_np = model_out_np[0]
    
    # 3. heatmap ops
    # normalise & ops
    # --------------
    model_out_np = model_out_np/np.max(model_out_np)
    model_out_np = (model_out_np * 255).astype('uint8')
    model_out_np = cv2.applyColorMap(model_out_np, cv2.COLORMAP_HSV)
    model_out_np = cv2.cvtColor(model_out_np, cv2.COLOR_BGR2RGB) 
    model_out_np = cv2.resize(model_out_np, (orig_w, orig_h))

    # overlay
    # -------
    overlay = (img_in.astype('uint8') * 0.5  + model_out_np * 0.5).astype('uint8')


    # 4. return
    # return image is RGB
    # -------------------
    return overlay


    
    


# ### 2. models &  related codes

# In[23]:


# a function to load a saved model
# --------------------------------

def load_saved_model_function_cpu(path):
    
    
    ''' path = /folder1/folder2/model_ae.tar format'''
    
    # 1. loading full model
    # ---------------------
    model = torch.load(path.replace('.tar','_MODEL.tar'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()))
    
    # 2. Applying state dict
    # loads to CPU
    # torch.load(checkpoint_file, map_location=‘cpu’)
    # ------------------------------------------------
    checkpoint = torch.load(path, map_location='cpu')
    
    # loading checkpoint
    # -------------------
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # loading optimizer
    # -----------------
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    # loading other stuff
    # -------------------
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    loss_mode = checkpoint['loss_mode']
    
    # just return model only
    # ----------------------
    return model
    
    


# In[24]:


# FCN mask_nomask classifier
# --------------------------

class simple_classifier(nn.Module):
    def __init__(self, len_classlabels, main_in_channels):
        super().__init__()
        
        # AIMING FOR THIS TO BE A FCNs
        ##############################
        
        '''
        
        classifier has 2 outputs of order ['without_mask_faces', 'with_mask_faces']
        
        '''
        
         # Initialising N/W here
        # ---------------------
        nw_activation_conv = nn.LeakyReLU(0.2, inplace=True) #nn.ReLU() # nn.ReLU() #nn.ReLU() #nn.LeakyReLU(0.2) # nn.Tanh() nn.Softmax2d()
        f = 3
        s = 2
        dropout_prob = 0.2
        dropout_node = nn.Dropout2d(p=dropout_prob)
      
        # 1. image encoding
        # -----------------

        # 0.
        ####
        conv0_ch = 32
        ct0 = nn.Conv2d(main_in_channels,conv0_ch,f,stride = s)
        cb0 = nn.BatchNorm2d(conv0_ch)
        ca0 = nw_activation_conv
        cl0 = [ct0,cb0,ca0,dropout_node]
        # 63
        
        # 1.
        ####
        conv1_ch = 64
        ct1 = nn.Conv2d(conv0_ch,conv1_ch,f,stride = s)
        cb1 = nn.BatchNorm2d(conv1_ch)
        ca1 = nw_activation_conv
        cl1 = [ct1,cb1,ca1,dropout_node]
        # 31
        
        # 2.
        ####
        conv2_ch = 128
        ct2 = nn.Conv2d(conv1_ch,conv2_ch,f,stride = s)
        cb2 = nn.BatchNorm2d(conv2_ch)
        ca2 = nw_activation_conv
        cl2 = [ct2,cb2,ca2,dropout_node]
        # 15
        
        # 3.
        ####
        conv3_ch = 256
        ct3 = nn.Conv2d(conv2_ch,conv3_ch,f,stride = s)
        cb3 = nn.BatchNorm2d(conv3_ch)
        ca3 = nw_activation_conv
        cl3 = [ct3,cb3,ca3,dropout_node]
        # 7

        # 4.
        ####
        conv4_ch = 512
        ct4 = nn.Conv2d(conv3_ch,conv4_ch,f,stride = s)
        cb4 = nn.BatchNorm2d(conv4_ch)
        ca4 = nw_activation_conv
        cl4 = [ct4,cb4,ca4,dropout_node]
        # 3

        # 5.
        ####
        ct5 = nn.Conv2d(conv4_ch,len_classlabels,f,stride = s)
        ca5 = nn.Sigmoid()
        cl5 = [ct5,ca5]
        # 1
        
       
        # building encoder
        # ----------------
        self.image_encoder = nn.Sequential(*cl0 + cl1 + cl2 + cl3 + cl4 + cl5)



    def forward(self, x):
        
        # simple stuff
        # ------------
        final_out = self.image_encoder(x).reshape(x.size()[0],-1)
       
        # final return
        # ------------
        return final_out
    


# In[25]:


# FCN class copied from image search notebook which worked
# generator_1_127 latent_dim, line_in_channels, design_in_channels, out_channels return_encoded_latents
# --------------------------------------------------------

class fcn_UNET_segmentation(nn.Module):
    def __init__(self, len_classlabels):
        super().__init__()
        
        # AIMING FOR THIS TO BE A FCNs
        ##############################
        # generator with RELU and discrimitor with leaky relu
        
         # Initialising N/W here
        # ---------------------
        nw_activation_conv = nn.LeakyReLU(0.2) #nn.LeakyReLU(0.2) #nn.ReLU() # nn.ReLU() #nn.ReLU() #nn.LeakyReLU(0.2) # nn.Tanh() nn.Softmax2d()
        f = 3
        s = 2
        dropout_prob = 0.1
        dropout_node = nn.Dropout2d(p=dropout_prob)
        self.main_latent_dim = 128

        # 1. image encoder
        # ----------------
        # 00.
        ####
        conv00_ch = 32
        ct00 = nn.Conv2d(3,conv00_ch,f,stride = s)
        cb00 = nn.BatchNorm2d(conv00_ch)
        ca00 = nw_activation_conv
        self.cl00 = nn.Sequential(*[ct00,cb00,ca00,dropout_node])
        # 127

        # 0.
        ####
        conv0_ch = 32
        ct0 = nn.Conv2d(conv00_ch,conv0_ch,f,stride = s)
        cb0 = nn.BatchNorm2d(conv0_ch)
        ca0 = nw_activation_conv
        self.cl0 = nn.Sequential(*[ct0,cb0,ca0,dropout_node])
        # 63
        
        # 1.
        ####
        conv1_ch = 64
        ct1 = nn.Conv2d(conv0_ch,conv1_ch,f,stride = s)
        cb1 = nn.BatchNorm2d(conv1_ch)
        ca1 = nw_activation_conv
        self.cl1 = nn.Sequential(*[ct1,cb1,ca1,dropout_node])
        # 31
        
        # 2.
        ####
        conv2_ch = 128
        ct2 = nn.Conv2d(conv1_ch,conv2_ch,f,stride = s)
        cb2 = nn.BatchNorm2d(conv2_ch)
        ca2 = nw_activation_conv
        self.cl2 = nn.Sequential(*[ct2,cb2,ca2,dropout_node])
        # 15
        
        # 3.
        ####
        conv3_ch = 256
        ct3 = nn.Conv2d(conv2_ch,conv3_ch,f,stride = s)
        cb3 = nn.BatchNorm2d(conv3_ch)
        ca3 = nw_activation_conv
        self.cl3 = nn.Sequential(*[ct3,cb3,ca3,dropout_node])
        # 7
        
        # 4.
        ####
        conv4_ch = 512
        ct4 = nn.Conv2d(conv3_ch,conv4_ch,f,stride = s)
        cb4 = nn.BatchNorm2d(conv4_ch)
        ca4 = nw_activation_conv
        self.cl4 = nn.Sequential(*[ct4,cb4,ca4,dropout_node])
        # 3

        # 5.
        ####
        ct5 = nn.Conv2d(conv4_ch,self.main_latent_dim,f,stride = s)
        ca5 = nw_activation_conv
        self.cl5 = nn.Sequential(*[ct5,ca5,dropout_node])
        # 1


        #################################################
        #################################################
        #################################################

        # GETTING INTO UPCONS
        # -------------------
        # Upconv layer 1
        ###
        t1 = nn.ConvTranspose2d(self.main_latent_dim,conv4_ch,f,stride = s)
        b1 = nn.BatchNorm2d(conv4_ch)
        a1 = nw_activation_conv
        self.ul1 = nn.Sequential(*[t1,b1,a1,dropout_node])
        # 3x3
        
        # Upconv layer 2
        ###
        t2 = nn.ConvTranspose2d(conv4_ch*2,conv3_ch,f,stride = s)
        b2 = nn.BatchNorm2d(conv3_ch)
        a2 = nw_activation_conv
        self.ul2 = nn.Sequential(*[t2,b2,a2,dropout_node])
        # 7
        
        # Upconv layer 3
        ###
        t3 = nn.ConvTranspose2d(conv3_ch*2,conv2_ch,f,stride = s)
        b3 = nn.BatchNorm2d(conv2_ch)
        a3 = nw_activation_conv
        self.ul3 = nn.Sequential(*[t3,b3,a3,dropout_node])
        # 15
        
        # Upconv layer 4
        ###
        t4 = nn.ConvTranspose2d(conv2_ch*2,conv1_ch,f,stride = s)
        b4 = nn.BatchNorm2d(conv1_ch)
        a4 = nw_activation_conv
        self.ul4 = nn.Sequential(*[t4,b4,a4,dropout_node])
        # 31
        
        # Upconv layer 5
        ###
        t5 = nn.ConvTranspose2d(conv1_ch*2,conv0_ch,f,stride = s)
        b5 = nn.BatchNorm2d(conv0_ch)
        a5 = nw_activation_conv
        self.ul5 = nn.Sequential(*[t5,b5,a5,dropout_node])
        # 63
        
        
        # Upconv layer 6
        ###
        t6 = nn.ConvTranspose2d(conv0_ch*2,conv00_ch,f,stride = s)
        b6 = nn.BatchNorm2d(conv00_ch)
        a6 = nw_activation_conv
        self.ul6 = nn.Sequential(*[t6,b6,a6,dropout_node])
        # 63
        

        # Upconv layer 6
        ###
        t7 = nn.ConvTranspose2d(conv00_ch*2,len_classlabels,f,stride = s)
        a7 = nn.Sigmoid()
        self.ul7 = nn.Sequential(*[t7,a7])
        # 127

       

    def forward(self, x):
        
        # encoding
        # --------
        conv00_out = self.cl00(x)
        conv0_out = self.cl0(conv00_out)
        conv1_out = self.cl1(conv0_out)
        conv2_out = self.cl2(conv1_out)
        conv3_out = self.cl3(conv2_out)
        conv4_out = self.cl4(conv3_out)
        conv5_out = self.cl5(conv4_out)


   
        # straightforward outs
        # --------------------
        up1_out = self.ul1(conv5_out)
        up2_out = self.ul2(torch.cat((up1_out, conv4_out), 1))
        up3_out = self.ul3(torch.cat((up2_out, conv3_out), 1))
        up4_out = self.ul4(torch.cat((up3_out, conv2_out), 1))
        up5_out = self.ul5(torch.cat((up4_out, conv1_out), 1))
        up6_out = self.ul6(torch.cat((up5_out, conv0_out), 1))
        up7_out = self.ul7(torch.cat((up6_out, conv00_out), 1))
        
        
        # final return
        # ------------
        return up7_out


# depth perceptionrelated
# -----------------------
# FCN class copied from image search notebook which worked
# generator_1_127 latent_dim, line_in_channels, design_in_channels, out_channels return_encoded_latents
# --------------------------------------------------------

# 1.
# FCN 1x1
# --------
class fcn_UNET_depthperception(nn.Module):
    def __init__(self):
        super().__init__()
        
        # AIMING FOR THIS TO BE A FCNs
        ##############################
        # generator with RELU and discrimitor with leaky relu
        
         # Initialising N/W here
        # ---------------------
        nw_activation_conv = nn.LeakyReLU(0.2) #nn.LeakyReLU(0.2) #nn.ReLU() # nn.ReLU() #nn.ReLU() #nn.LeakyReLU(0.2) # nn.Tanh() nn.Softmax2d()
        f = 3
        s = 2
        dropout_prob = 0.1
        dropout_node = nn.Dropout2d(p=dropout_prob)
        self.main_latent_dim = 128

        # 1. image encoder
        # ----------------
        # 00.
        ####
        conv00_ch = 32
        ct00 = nn.Conv2d(3,conv00_ch,f,stride = s)
        cb00 = nn.BatchNorm2d(conv00_ch)
        ca00 = nw_activation_conv
        self.cl00 = nn.Sequential(*[ct00,cb00,ca00,dropout_node])
        # 127

        # 0.
        ####
        conv0_ch = 32
        ct0 = nn.Conv2d(conv00_ch,conv0_ch,f,stride = s)
        cb0 = nn.BatchNorm2d(conv0_ch)
        ca0 = nw_activation_conv
        self.cl0 = nn.Sequential(*[ct0,cb0,ca0,dropout_node])
        # 63
        
        # 1.
        ####
        conv1_ch = 64
        ct1 = nn.Conv2d(conv0_ch,conv1_ch,f,stride = s)
        cb1 = nn.BatchNorm2d(conv1_ch)
        ca1 = nw_activation_conv
        self.cl1 = nn.Sequential(*[ct1,cb1,ca1,dropout_node])
        # 31
        
        # 2.
        ####
        conv2_ch = 128
        ct2 = nn.Conv2d(conv1_ch,conv2_ch,f,stride = s)
        cb2 = nn.BatchNorm2d(conv2_ch)
        ca2 = nw_activation_conv
        self.cl2 = nn.Sequential(*[ct2,cb2,ca2,dropout_node])
        # 15
        
        # 3.
        ####
        conv3_ch = 256
        ct3 = nn.Conv2d(conv2_ch,conv3_ch,f,stride = s)
        cb3 = nn.BatchNorm2d(conv3_ch)
        ca3 = nw_activation_conv
        self.cl3 = nn.Sequential(*[ct3,cb3,ca3,dropout_node])
        # 7
        
        # 4.
        ####
        conv4_ch = 512
        ct4 = nn.Conv2d(conv3_ch,conv4_ch,f,stride = s)
        cb4 = nn.BatchNorm2d(conv4_ch)
        ca4 = nw_activation_conv
        self.cl4 = nn.Sequential(*[ct4,cb4,ca4,dropout_node])
        # 3

        # 5.
        ####
        ct5 = nn.Conv2d(conv4_ch,self.main_latent_dim,f,stride = s)
        ca5 = nw_activation_conv
        self.cl5 = nn.Sequential(*[ct5,ca5,dropout_node])
        # 1


        #################################################
        #################################################
        #################################################

        # GETTING INTO UPCONS
        # -------------------
        # Upconv layer 1
        ###
        t1 = nn.ConvTranspose2d(self.main_latent_dim,conv4_ch,f,stride = s)
        b1 = nn.BatchNorm2d(conv4_ch)
        a1 = nw_activation_conv
        self.ul1 = nn.Sequential(*[t1,b1,a1,dropout_node])
        # 3x3
        
        # Upconv layer 2
        ###
        t2 = nn.ConvTranspose2d(conv4_ch*2,conv3_ch,f,stride = s)
        b2 = nn.BatchNorm2d(conv3_ch)
        a2 = nw_activation_conv
        self.ul2 = nn.Sequential(*[t2,b2,a2,dropout_node])
        # 7
        
        # Upconv layer 3
        ###
        t3 = nn.ConvTranspose2d(conv3_ch*2,conv2_ch,f,stride = s)
        b3 = nn.BatchNorm2d(conv2_ch)
        a3 = nw_activation_conv
        self.ul3 = nn.Sequential(*[t3,b3,a3,dropout_node])
        # 15
        
        # Upconv layer 4
        ###
        t4 = nn.ConvTranspose2d(conv2_ch*2,conv1_ch,f,stride = s)
        b4 = nn.BatchNorm2d(conv1_ch)
        a4 = nw_activation_conv
        self.ul4 = nn.Sequential(*[t4,b4,a4,dropout_node])
        # 31
        
        # Upconv layer 5
        ###
        t5 = nn.ConvTranspose2d(conv1_ch*2,conv0_ch,f,stride = s)
        b5 = nn.BatchNorm2d(conv0_ch)
        a5 = nw_activation_conv
        self.ul5 = nn.Sequential(*[t5,b5,a5,dropout_node])
        # 63
        
        
        # Upconv layer 6
        ###
        t6 = nn.ConvTranspose2d(conv0_ch*2,conv00_ch,f,stride = s)
        b6 = nn.BatchNorm2d(conv00_ch)
        a6 = nw_activation_conv
        self.ul6 = nn.Sequential(*[t6,b6,a6,dropout_node])
        # 63
        

        # Upconv layer 6
        # the outputs would be logits
        ###
        t7 = nn.ConvTranspose2d(conv00_ch*2,conv00_ch,f,stride = s)
        b7 = nn.BatchNorm2d(conv00_ch)
        a7 = nw_activation_conv
        t7_f = nn.ConvTranspose2d(conv00_ch,1,1,stride = 1)
        self.ul7 = nn.Sequential(*[t7,b7,a7,t7_f,a7])
        # 127

       

    def forward(self, x):
        
        # encoding
        # --------
        conv00_out = self.cl00(x)
        conv0_out = self.cl0(conv00_out)
        conv1_out = self.cl1(conv0_out)
        conv2_out = self.cl2(conv1_out)
        conv3_out = self.cl3(conv2_out)
        conv4_out = self.cl4(conv3_out)
        conv5_out = self.cl5(conv4_out)


   
        # straightforward outs
        # --------------------
        up1_out = self.ul1(conv5_out)
        up2_out = self.ul2(torch.cat((up1_out, conv4_out), 1))
        up3_out = self.ul3(torch.cat((up2_out, conv3_out), 1))
        up4_out = self.ul4(torch.cat((up3_out, conv2_out), 1))
        up5_out = self.ul5(torch.cat((up4_out, conv1_out), 1))
        up6_out = self.ul6(torch.cat((up5_out, conv0_out), 1))
        up7_out = self.ul7(torch.cat((up6_out, conv00_out), 1))
        
        # using torch.exp to expand model prediction
        ##
        final_out = torch.exp(up7_out)
        
        
        # final return
        # ------------
        return final_out
    
    

# 2.
# 15x15
# -----
# FCN class copied from image search notebook which worked
# generator_1_127 latent_dim, line_in_channels, design_in_channels, out_channels return_encoded_latents
# --------------------------------------------------------

class fcn_UNET_depthperception_15(nn.Module):
    def __init__(self):
        super().__init__()
        
        # AIMING FOR THIS TO BE A FCNs
        ##############################
        # generator with RELU and discrimitor with leaky relu
        
         # Initialising N/W here
        # ---------------------
        nw_activation_conv = nn.LeakyReLU(0.2) #nn.LeakyReLU(0.2) #nn.ReLU() # nn.ReLU() #nn.ReLU() #nn.LeakyReLU(0.2) # nn.Tanh() nn.Softmax2d()
        f = 3
        s = 2
        dropout_prob = 0.1
        dropout_node = nn.Dropout2d(p=dropout_prob)
        self.main_latent_dim = 128

        # 1. image encoder
        # ----------------
        # 00.
        ####
        conv00_ch = 32
        ct00 = nn.Conv2d(3,conv00_ch,f,stride = s)
        cb00 = nn.BatchNorm2d(conv00_ch)
        ca00 = nw_activation_conv
        self.cl00 = nn.Sequential(*[ct00,cb00,ca00,dropout_node])
        # 127

        # 0.
        ####
        conv0_ch = 64
        ct0 = nn.Conv2d(conv00_ch,conv0_ch,f,stride = s)
        cb0 = nn.BatchNorm2d(conv0_ch)
        ca0 = nw_activation_conv
        self.cl0 = nn.Sequential(*[ct0,cb0,ca0,dropout_node])
        # 63
        
        # 1.
        ####
        conv1_ch = 128
        ct1 = nn.Conv2d(conv0_ch,conv1_ch,f,stride = s)
        #cb1 = nn.BatchNorm2d(conv1_ch)
        ca1 = nw_activation_conv
        self.cl1 = nn.Sequential(*[ct1,ca1,dropout_node])
        # 31
        
        # 2.
        ####
        conv2_ch = 256
        ct2 = nn.Conv2d(conv1_ch,conv2_ch,f,stride = s)
        cb2 = nn.BatchNorm2d(conv2_ch)
        ca2 = nw_activation_conv
        self.cl2 = nn.Sequential(*[ct2,cb2,ca2,dropout_node])
        # 15
        
        # 3.
        ####
        #conv3_ch = 256
        #ct3 = nn.Conv2d(conv2_ch,conv3_ch,f,stride = s)
        #cb3 = nn.BatchNorm2d(conv3_ch)
        #ca3 = nw_activation_conv
        #self.cl3 = nn.Sequential(*[ct3,cb3,ca3,dropout_node])
        # 7
        
        # 4.
        ####
        #conv4_ch = 512
        #ct4 = nn.Conv2d(conv3_ch,conv4_ch,f,stride = s)
        #cb4 = nn.BatchNorm2d(conv4_ch)
        #ca4 = nw_activation_conv
        #self.cl4 = nn.Sequential(*[ct4,cb4,ca4,dropout_node])
        # 3

        # 5.
        ####
        #ct5 = nn.Conv2d(conv4_ch,self.main_latent_dim,f,stride = s)
        #ca5 = nw_activation_conv
        #self.cl5 = nn.Sequential(*[ct5,ca5,dropout_node])
        # 1


        #################################################
        #################################################
        #################################################

        # GETTING INTO UPCONS
        # -------------------
        # Upconv layer 1
        ###
        #t1 = nn.ConvTranspose2d(self.main_latent_dim,conv4_ch,f,stride = s)
        #b1 = nn.BatchNorm2d(conv4_ch)
        #a1 = nw_activation_conv
        #self.ul1 = nn.Sequential(*[t1,b1,a1,dropout_node])
        # 3x3
        
        # Upconv layer 2
        ###
        #t2 = nn.ConvTranspose2d(conv4_ch*2,conv3_ch,f,stride = s)
        #b2 = nn.BatchNorm2d(conv3_ch)
        #a2 = nw_activation_conv
        #self.ul2 = nn.Sequential(*[t2,b2,a2,dropout_node])
        # 7
        
        # Upconv layer 3
        ###
        #t3 = nn.ConvTranspose2d(conv3_ch*2,conv2_ch,f,stride = s)
        #b3 = nn.BatchNorm2d(conv2_ch)
        #a3 = nw_activation_conv
        #self.ul3 = nn.Sequential(*[t3,b3,a3,dropout_node])
        # 15
        
        # Upconv layer 4
        ###
        t4 = nn.ConvTranspose2d(conv2_ch,conv1_ch,f,stride = s)
        b4 = nn.BatchNorm2d(conv1_ch)
        a4 = nw_activation_conv
        self.ul4 = nn.Sequential(*[t4,b4,a4,dropout_node])
        # 31
        
        # Upconv layer 5
        ###
        t5 = nn.ConvTranspose2d(conv1_ch*2,conv0_ch,f,stride = s)
        b5 = nn.BatchNorm2d(conv0_ch)
        a5 = nw_activation_conv
        self.ul5 = nn.Sequential(*[t5,b5,a5,dropout_node])
        # 63
        
        
        # Upconv layer 6
        ###
        t6 = nn.ConvTranspose2d(conv0_ch*2,conv00_ch,f,stride = s)
        b6 = nn.BatchNorm2d(conv00_ch)
        a6 = nw_activation_conv
        self.ul6 = nn.Sequential(*[t6,b6,a6,dropout_node])
        # 63
        

        # Upconv layer 6
        # the outputs would be logits
        ###
        t7 = nn.ConvTranspose2d(conv00_ch*2,conv00_ch,f,stride = s)
        b7 = nn.BatchNorm2d(conv00_ch)
        a7 = nw_activation_conv
        t7_f = nn.ConvTranspose2d(conv00_ch,1,1,stride = 1)
        self.ul7 = nn.Sequential(*[t7,b7,a7,t7_f,a7])
        # 127

       

    def forward(self, x):
        
        # encoding
        # --------
        conv00_out = self.cl00(x)
        conv0_out = self.cl0(conv00_out)
        conv1_out = self.cl1(conv0_out)
        conv2_out = self.cl2(conv1_out)
        #conv3_out = self.cl3(conv2_out)
        #conv4_out = self.cl4(conv3_out)
        #conv5_out = self.cl5(conv4_out)


   
        # straightforward outs
        # --------------------
        #up1_out = self.ul1(conv5_out)
        #up2_out = self.ul2(torch.cat((up1_out, conv4_out), 1))
        #up3_out = self.ul3(torch.cat((up2_out, conv3_out), 1))
        up4_out = self.ul4(conv2_out)
        up5_out = self.ul5(torch.cat((up4_out, conv1_out), 1))
        up6_out = self.ul6(torch.cat((up5_out, conv0_out), 1))
        up7_out = self.ul7(torch.cat((up6_out, conv00_out), 1))
        
        # using torch.exp to expand model prediction
        ##
        final_out = torch.exp(up7_out)
        
        
        # final return
        # ------------
        return final_out
    
    
   
   
# end of all code
# DELETE AFTER THIS IN PY FILE
# ############################
