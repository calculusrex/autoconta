

def description__from_pipeline_data(pipeline_data):
    if len(pipeline_data['progress']) == 0:
        last_fulfilled_stage = 'START'
    else:
        last_fulfilled_stage = pipeline_data['progress'][-1]

    if last_fulfilled_stage == 'OCR':
        if len(pipeline_data['validation']) == 0:
            description = "OCR"
        else:
            validated_box_n = len(pipeline_data['validation'])
            current_ocr_box_text = pipeline_data['validation'][-1]['text']
            description = f'{validated_box_n}VldtBxs__currTxt_{current_ocr_box_text}'
    else:
        description = last_fulfilled_stage
    return description
