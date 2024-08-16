# the main code for clip 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, ViTImageProcessor, ViTModel, AutoTokenizer, AutoModelForCausalLM
import transformers

VISION_ENCODER = "google/vit-base-patch16-224-in21k"
TEXT_ENCODER = "surajp/gpt2-hindi"

class VisionEncoder(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        try:
            self.processor = ViTImageProcessor.from_pretrained(vision_encoder)
            self.model = ViTModel.from_pretrained(vision_encoder, attn_implementation="eager")
        except Exception as e:
            print(f"Error loading vision encoder: {e}")

        for params in self.model.parameters():
            params.requires_grad = True

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        return outputs.last_hidden_state[:, 0, :]  

class TextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(text_encoder, attn_implementation="eager")
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        except Exception as e:
            print(f"Error loading text encoder: {e}")

        for params in self.model.parameters():
            params.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1][:, -1, :] 

class MLP(nn.Module):
    def __init__(self, embed_dim, projected_dim=768, dropout=0.1):
        super().__init__()
        self.ln1 = nn.Linear(embed_dim, projected_dim)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(projected_dim, projected_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projected_dim)

    def forward(self, x):
        proj = self.ln1(x)
        x = self.gelu(proj)
        x = self.ln2(x)
        x = self.dropout(x)
        x = x + proj
        x = self.layer_norm(x)
        return x

class CLIP(nn.Module):
    def __init__(self, vision_embeds, vision_encoder, text_encoder, text_embeds, temperature=0.07):
        super().__init__()
        self.text_encoder = TextEncoder(text_encoder)
        self.mlp_text = MLP(embed_dim=text_embeds)
        self.vision_encoder = VisionEncoder(vision_encoder)
        self.mlp_image = MLP(embed_dim=vision_embeds)
        self.temperature = temperature

    def forward(self, batch):
        image_embs = self.vision_encoder(batch['pixel_values'])
        text_embs = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        projected_text_embeds = self.mlp_text(text_embs)
        projected_image_embeds = self.mlp_image(image_embs)

        if projected_text_embeds.size(0) != projected_image_embeds.size(0):
            projected_image_embeds = projected_image_embeds.expand(projected_text_embeds.size(0), -1)

        
        logits = (projected_text_embeds @ projected_image_embeds.T) / self.temperature
        image_sim = projected_image_embeds @ projected_image_embeds.T
        text_sim = projected_text_embeds @ projected_text_embeds.T
        targets = F.softmax((image_sim + text_sim) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')

        print("text_loss" , texts_loss,"image_loss", images_loss)
        loss = (images_loss + texts_loss) / 2.0

        return loss.mean()