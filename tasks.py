
#x# - put all the labels in the app_data structure
#x# - make the rotation editor scroll work on canvas
#x# - make the rotation click and hover labels target the canvas
#x# - make the threshold editor open up with the threshold applied
#x# - labels
  #x# - make all the click labels update their click coordonates where clicks are involved
  #x# - Do all frames have working mouse event labels?
  #x# - move the labels to the third column, such that the second canvas won't overlap them.
  #x# - Do all frames have labels for their respective improc parameters? (orth scroll!)

#x# - Image editing workflow
    #x# - warping, cropping and ROI
    #x# - rotate
    #x# - rescale
    #x# - denoise (median blur)
    #x# - grayscale and thresholding
    #x# - dilate, erode

    #x# - Link the frames together into a coherent workflow
    #x# - Bind the parameter scroll to the canvas, not the frame
    #x# - Add labels for displaying the image processing parameters
    #/# - Figure out how to properly pass parameters to the functions called by events
    #x# - Refactor the image processing code
    #-# ? Split the widget data out from the state_seed data (make a new data structure)

#x# - put a crop stage between fine rotation and rescale
#x# - bind the escape key to cancel the current action (revert to the stage initial state)
#x# - make a general FilteringEditor class from which DenoiseEditor, DilateErodeEditor...
      # and all the other filtering stage classes will inherit
#x# - make the warping stage interactive
      # (such that you can move the points afer laying them down and see the change
      # in the warp result in real time)
#x# - implement undo to revert to the previus improc stage
#x# - implement wasd tuning for the warping stage
# # - add a border adding/growing editor

#x# - OCR validation boxes
#        (raster image label with input box containing detected text underneath)
  #x# - OCR class prototype
  #x# - OCRValidation

# # - User drawn rectangle -> OCR acting on cropped section of image
# # - Word attribute augmentation via selecting boxes with shared attributes on the canvas

#x# - Implement "save pipeline data"
#x# - Make the validation entry box larger (including it's text larger)
#x# - make the validation canvas not exceed a height larger than 1/3 of the screen height
#x# - Distinguish between the types of recognized OCR boxes outputted by tesseract
  #x# - Assign different colors for plotting them
  #x# - Filter the boxes being used for ocr validation to only be text-recognition boxes, not block bounds / hierarchical bounds

# # - test

#x# - implement an automatic point placement at the image bounds for the warping\
    # editor

#x# - implement a multiple editor batching workflow

#x# - implement a batching document preprocessing workflow
  #x# - saves files to disk
  #x# - keeps track of the parameters used for each file (writes them to csv)
  # # - remembers parameters used for previous file, for each operation
 #x# - try to load all 80 files to RAM, let's see if it crashes

# # - implement a single-click text selection tool for selecting the ocr box for \
    # the header fields (document-id, supplier, total, date)

#x# - implement a box-selection tool for selecting the article frame header \
    # and another for selecting the actual table, with a dataframe assembling \
    # function (which takes into account the position of the bounding boxes)

#x# - implement roi selection for document constituents

#x# - pass controll back to the script by destroying the root widget

#x# - reimplement the document preprocessing stage in the functional style
  #x# - reparametrize the frames (functional frames)
  #x# - develop the data structures that need to be passed as parameters to the gui invoquing function
  #x# - rename the previously processed data, such that it won't get overwritten.
  #x# - implement specialized transform and filter pipeline gui deployment functions which take only an image as parameter
  #x# - implement the image loading facility and the imperative script loading the images
  #x# - execute and debug till functional
  #x# - implement save to disk for the new functional design
#x# - implement a OCR pipeline in the functional style

#x# - extract ocr recognized data from a few documents.

#x# - implement skip in all gui frames
  #x# - run a improc gui testbench
    # it was already done, i just had to call a function i wrote
#x# - implement cancel in all gui frames
#x# - implement box select for what's not being read by tessract on the whole document
#x# - install instructive labels in the ocr gui frames
#x# - install crosshairs in the ocr frames
#x# - make a validation frame for validating / editing the recognized text from the document
# # - implement selective interpretation procedures for the type of field in question: furnizor(<Capitalized String>), cantitate(<Float>, ...)
# # - implement select from historical values in the validation frame
# # - make a table processing ocr program with which to extract data from tables

# # - multiple selection line selection at the OCR Frame
  # i sometimes need to select boxes which are not in line, for example, bleeding in the next row

# # - draw boxes in the ocr validation frames for box selected values too, not just for tesseract generated bounding boxes

# # - display the date currently processed in the lower part of every frame. this will give context to the operator

# # - if the document is much taller than is wide, the canvas goes off the page in the OCR frame. Fix that
