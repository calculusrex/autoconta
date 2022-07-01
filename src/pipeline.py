



improcPipeline = Pipeline(
    sequence=[WarpingEditor,
              OrthogonalRotationEditor,
              FineRotationEditor,
              RescaleEditor,
              CropEditor,
              DenoiseEditor,
              ThresholdEditor,
              DilateErodeEditor])

ocrPipeline = Pipeline(
    sequence=[TesseractOCR,
              OCRValidation])

dataReadPipeline = Pipeline(
    sequence=[improcPipeline, ocrPipeline, Save]
