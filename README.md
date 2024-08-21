# clip-hindi
CLIP (Contrastive Language-Image Pre-training)  a vision-language model that connects images and text by learning a shared representation. The model is trained to understand and match images with their corresponding textual descriptions without relying on traditional supervised learning with labeled datasets. Instead, it leverages a large amount of image-text pairs from the web, which allows it to generalize well to new and unseen data.

In this case we used two pre-trained model to encode both images and texts:

1) l3cube-pune/hindi-bert-v2 from huggingface 
2) resnet50 from timm 

# Examples 
this model takes in a text or a prompt like `घास पर कुत्ता` and then a bunch of images pass the images and prompt to the text encoder and image encoder computes the similarities and then tell us which images is the most similar based on the prompt 


# Get Started 

First you will have to git clone this repo 

```bash
!git clone https://github.com/dame-cell/clip-hindi.git
%cd  clip-hindi
!pip install -r requirements.txt
```

Then you wil have to download the pytorch model from huggingface

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="damerajee/clip-hindi", filename="model.pt",local_dir="model")
```
You can then load the model  and the image as well we load the image from skimage 

```python 
import torch 
from clip.modeling_clip import CLIPModel
from skimage import data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import AutoTokenizer 
import torch.nn.functional as F

model = CLIPModel().to("cuda")
model.load_state_dict(torch.load("/kaggle/working/clip-hindi/model/model.pt", map_location="cuda"))
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hindi-bert-v2")

text_inputs = tokenizer(["किताब में एक पेज", "एक कॉफ़ी", "एक बिल्ली"], padding=True, truncation=True, return_tensors="pt")

# load the images 
image = data.coffee()  
cat_image = Image.fromarray(np.uint8(image))
image_np = np.array(image)  
preprocessor = model.preprocess()  
preprocessed = preprocessor(image=image_np)['image'] 

preprocessed_tensor = torch.tensor(preprocessed).float()
preprocessed_tensor = preprocessed_tensor.permute(2, 0, 1)  # Change shape to [C, H, W]
preprocessed_tensor = preprocessed_tensor.unsqueeze(0)

processed_image =preprocessed_tensor.to("cuda")
text_inputs = text_inputs.to("cuda")
```
and then you can try it out 
```python

image_features , text_features = model(processed_image,text_input_ids=text_inputs['input_ids'],text_attention_mask=text_inputs['attention_mask'])

# normalize the features 
image_embeddings_n = F.normalize(image_features, p=2, dim=-1)
text_embeddings_n = F.normalize(text_features, p=2, dim=-1)

# calculate the similarities 
dot_similarity = text_embeddings_n @ image_embeddings_n.T
print("dot_similarity",dot_similarity)

# output - > tensor([[0.0389],[0.0460],[0.0352]], device='cuda:0', grad_fn=<MmBackward0>)
```



# Reference 

```bibtex
@software{Shariatnia_Simple_CLIP_2021,
author = {Shariatnia, M. Moein},
doi = {10.5281/zenodo.6845731},
month = {4},
title = {{Simple CLIP}},
version = {1.0.0},
year = {2021}
}
```