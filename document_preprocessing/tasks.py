
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

# # - OCR validation boxes
#        (raster image label with input box containing detected text underneath)
  #x# - OCR class prototype
  # # - OCRValidation

# # - User drawn rectangle -> OCR acting on cropped section of image
# # - Word attribute augmentation via selecting boxes with shared attributes on the canvas

#x# - Implement "save pipeline data"
# # - Make the validation entry box larger (including it's text larger)
# # - Distinguish between the types of recognized OCR boxes outputted by tesseract
  # # - Assign different colors for plotting them
  # # - Filter the boxes being used for ocr validation to only be text-recognition boxes, not block bounds / hierarchical bounds
