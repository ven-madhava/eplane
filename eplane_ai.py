# 1. imports
# ----------
import sys
import argparse
import cv2
from eplane_ai_utils import *


# 2. defining main function
# -------------------------
def main_wrapper(task, input_mode, file_url):

	# 0. inits
	# --------
	# none

	# 1. process inputs
	# -----------------
	if input_mode == 'image':

		# read image from URL and process
		# -------------------------------
		img_in = cv2.cvtColor(cv2.imread(file_url), cv2.COLOR_BGR2RGB)
		output_image = process_image(task,img_in)

		# showing output image
		# --------------------
		cv2.imshow('image',output_image)
		cv2.waitKey(0)
	
		# final return
		# ------------
		# none

	else:

		# input here is video
		# -------------------
		process_video(task, file_url)

		# DONE
		######



# 3ß. function to assert args
# -----------------------
def assert_args(args_in):

	# 0. make sure these are inline
	# -----------------------------
	assert args_in.task == 'facemask' or args_in.task == 'seg' or args_in.task == 'depth', 'Task Input Error: Task has to be facemake, seg or depth'
	assert args_in.input_mode == 'image' or args_in.input_mode == 'video', 'Input Mode Error: INput mode has to be image or video'

	# 1. all good - next steps
	# ------------------------
	main_wrapper(args_in.task,args_in.input_mode,args_in.file_url)

	# END
	########




# end of functions
# parser ops
# ----------------

# 1. init parser
# ---------------
parser = argparse.ArgumentParser()

# 2. add argumenst to parser
# --------------------------
parser.add_argument('-t', '--task', help="Enter task to peform - 'facemask': Face mask or not classifiation, 'seg': Street image segmentation, 'depth': Depth perception")
parser.add_argument('-i', '--input_mode', help="Enter input mode - 'image' or 'video'")
parser.add_argument('-f', '--file_url', help="Enter file url to perform to perform task on")
# more args

# 3. parse arguments
# The arguments are parsed with parse_args(). 
# The parsed arguments are present as object attributes. 
# In our case, there will be args.task,  args.file_url & args.input_mode attribute.
# ---------------------------------------------------------------------------------
args = parser.parse_args()

# 4. CALLING MAIN FUNCTION
###########################
assert_args(args)



# END OF ALL CODE XX
##

