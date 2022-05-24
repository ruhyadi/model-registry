import json
import base64
from PIL import Image
import io
import torch

def init_context(context):
    context.logger.info("[INFO] Init context load model...  0%")

    # Read the DL model
    model = torch.hub.load('ruhyadi/model-registry:v1.0', 'yolov5n')
    context.user_data.model = model

    context.logger.info("[INFO] Init context load model...100%")

def handler(context, event):
    # TODO: replace MODELNAME
    context.logger.info("[INFO] Run YOLOv5 Demo")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)

    # inference image
    results = context.user_data.model(image).pandas().xyxy[0].to_dict(orient='records')

    # encoding result from inference
    encoded_results = []
    for result in results:
        encoded_results.append({
            'confidence': result['confidence'],
            'label': result['name'],
            'points': [
                result['xmin'],
                result['ymin'],
                result['xmax'],
                result['ymax']
            ],
            'type': 'rectangle'
        })

    # dump json to cvat annotations
    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)