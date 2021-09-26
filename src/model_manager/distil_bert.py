from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn

PAD_TOKEN = "[PAD]"


class DistilBertLinear(nn.Module):
    def __init__(self, no_class: int, pre_trained_model: str):
        super(DistilBertLinear, self).__init__()
        self.no_classes = no_class
        self.pad_ix = DistilBertTokenizer.from_pretrained(pre_trained_model).vocab[PAD_TOKEN]
        self.encoder = DistilBertModel.from_pretrained(pre_trained_model)
        self.encoder.train()
        self.act = nn.ReLU()
        self.hidden2tag = nn.Linear(768, no_class)

    def get_params(self):
        # params = self.hidden2tag.parameters()
        params = self.parameters()
        return params

    def forward(self, x, attention_mask):
        last_hidden_states = self.encoder(x, attention_mask=attention_mask)
        encoded_x = last_hidden_states[0][:, 0, :]
        z = self.hidden2tag(encoded_x)

        return z
