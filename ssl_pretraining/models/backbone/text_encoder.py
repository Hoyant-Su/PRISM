import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer, BertConfig

class Rad_TextEncoder(nn.Module):
    def __init__(self, num_classes, frozen_text_encoder, pretrained_model_path='/TextEncoder/bert_model', type_="normal"):
        super(Rad_TextEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type_ = type_

        self.config = BertConfig.from_pretrained(pretrained_model_path)
        self.bert_model = BertModel.from_pretrained(pretrained_model_path, config=self.config)
        self.linear = nn.Linear(self.config.hidden_size, 512)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.head = nn.Linear(512, num_classes)

    def forward(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.bert_model(**encoded_input)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_token_output = self.linear(cls_output)
        output = self.head(cls_token_output)
        if self.type_ == "normal":
            return output
        else:
            return cls_token_output

if __name__ == "__main__":
    text = "Replace me by any medical text you'd like."

    rad_encoder = Rad_TextEncoder(num_classes=2, frozen_text_encoder=False, pretrained_model_path='../TextEncoder/BERT/bert_model')

    pre_logits = rad_encoder(text)

    print(f"Pre-logits shape: {pre_logits.shape}")
