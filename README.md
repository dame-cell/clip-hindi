# clip-hindi
CLIP (Contrastive Language-Image Pre-training)  a vision-language model that connects images and text by learning a shared representation. The model is trained to understand and match images with their corresponding textual descriptions without relying on traditional supervised learning with labeled datasets. Instead, it leverages a large amount of image-text pairs from the web, which allows it to generalize well to new and unseen data.

In this case we used two pre-trained model to encode both images and texts:

1) l3cube-pune/hindi-bert-v2 from huggingface 
2) resnet50 from timm 

# Examples 
this model takes in a text or a prompt like `घास पर कुत्ता` and then a bunch of images pass the images and prompt to the text encoder and image encoder computes the similarities and then tell us which images is the most similar based on the prompt 


# Get Started 
You can easily try this model out in this colab notebook

https://colab.research.google.com/drive/1j1wudcKZK3aRh0J_0PZoMvzucoiUUzwP?usp=sharing


# refernce 

