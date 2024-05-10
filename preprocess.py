import io
import base64
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps

from clearml import StorageManager

cmap = np.array([
    [ 68.08602 ,   1.24287 ,  84.000825],
    [ 70.173705,  49.700775, 126.481275],
    [ 54.64599 ,  90.682845, 140.55192 ],
    [ 39.10782 , 126.735   , 142.21962 ],
    [ 30.9519  , 160.52046 , 135.653115],
    [ 73.674855, 193.39047 , 109.24863 ],
    [157.154715, 217.440795,  58.66326 ],
    [253.27824 , 231.070035,  36.70368 ]
]).astype(np.uint8)

label_names = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']

def base64_to_img(b64_str):
    pil_image = Image.open(io.BytesIO(base64.b64decode(b64_str))).\
                    convert('RGB').\
                    resize((416, 416))
    image = np.array(pil_image)
                                   
    return image

def img_to_base64(image):
    img = Image.fromarray(image)
    im_file = io.BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    base64_str = base64.b64encode(im_bytes)

    return base64_str

# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if isinstance(body, bytes):
            # stream of encoded image bytes
            try:
                image = np.array(Image.open(io.BytesIO(body)).convert("RGB"))/255.0
            except Exception:
                # value error would return 404, we want to return 500 so any other exception
                raise RuntimeError("Image could not be decoded")

        if isinstance(body, dict) and "base64_str" in body.keys():
            base64_str = body.get("base64_str")
            image = base64_to_img(base64_str)/255.0
            print("="*20, image.shape)
        
        return np.array([image]).astype(np.float32)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        if isinstance(data, list):
            data = data[0]
        if not isinstance(data, np.ndarray):
            # this should not happen
            return dict(output=-1)
        output_list = []
        print("="*20, data.shape)
        if len(data.shape)>3:
            for out in data:
                out = np.argmax(out, axis=-1)
                print("="*20, out.shape)
                print("="*20, cmap[out].shape)
                item={
                    'seg_mask': img_to_base64(cmap[out]),
                }
                output_list.append(item)
                
        # data is returned as probability per class (10 class/digits)
        return dict(output=output_list)