import torch
import torch.nn as nn
from einops import rearrange,repeat

from math import sqrt

from .attn import FullAttention
from .embed import DataEmbedding
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
from einops import reduce

class Linear_Encoder(nn.Module):
    def __init__(self, patch_size,patch_num, d_model,channel):
        super(Linear_Encoder, self).__init__()
        self.channel=channel
        self.patch_encoder=nn.Linear(d_model,d_model)
        self.point_encoder=nn.Linear(d_model,d_model)

    def forward(self, x_patch_size, x_patch_num, x_ori,  attn_mask=None):
        series=self.patch_encoder(x_patch_size)
        prior=self.point_encoder(x_patch_num)
        series=reduce(series,"(b c) m n -> b m n","mean", c=self.channel)
        prior=reduce(prior,"(b c) m n -> b m n","mean", c=self.channel)

        return series, prior

class LLMDetector(nn.Module):
    def __init__(self, win_size, enc_in, n_heads=1, d_model=256,llm_model="gpt2", patch_size=5, channel=1, dropout=0.0, output_attention=True,description=None):
        super(LLMDetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.patch_num=win_size//patch_size
        self.dataset_description=description

        self.channel_fusion = FullAttention(channel, n_heads,  attention_dropout=dropout)
        self.embedding_patch_size=DataEmbedding(self.patch_size, d_model, dropout)
        self.embedding_patch_num=DataEmbedding(self.win_size//self.patch_size, d_model, dropout)

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
         
        self.encoder=Linear_Encoder(patch_size,patch_size,d_model,channel)
        
        if llm_model=="gpt2":
            self.max_token = 1024
            self.gpt2_config = GPT2Config.from_pretrained("LLM/openai-community/gpt2")
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    "LLM/openai-community/gpt2",
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    "LLM/openai-community/gpt2",
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "LLM/openai-community/gpt2",
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "LLM/openai-community/gpt2",
                    local_files_only=False
                )
        elif llm_model=="llama2":
            self.max_token=2048
            self.llama2_config=LlamaConfig.from_pretrained("LLM/Llama/llama2")
            self.llama2_config.output_attentions = True
            self.llama2_config.output_hidden_states = True
            # self.llm_dim=1024
            try:
                self.llm_model=LlamaModel.from_pretrained(
                    "LLM/Llama/llama2",
                    local_files_only=True,
                    config=self.llama2_config
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model=LlamaModel.from_pretrained(
                    "LLM/Llama/llama2",
                    local_files_only=False,
                    config=self.llama2_config
                )
            try:
                self.tokenizer=LlamaTokenizer.from_pretrained(
                    "LLM/Llama/llama2",
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer=LlamaTokenizer.from_pretrained(
                    "LLM/Llama/llama2",
                    local_files_only=False
                )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token
        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        llm_dim = self.word_embeddings.shape[-1]
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.alignment_layer = FeatureAlignmentLayer(d_model, n_heads, d_llm=llm_dim)

    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x_ori = self.embedding_window_size(x)
        x=self.channel_fusion(x)
        x_patch, x_point = x, x
        x_patch = rearrange(x_patch, "b l m -> b m l") #Batch channel win_size
        x_point = rearrange(x_point, "b l m -> b m l") #Batch channel win_size
        x_patch = rearrange(x_patch, "b m (n p) -> (b m) n p", p = self.patch_size)
        x_patch = self.embedding_patch_size(x_patch)
        x_point = rearrange(x_point, "b m (p n) -> (b m) p n", p = self.patch_size)
        x_point = self.embedding_patch_num(x_point)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        patch, point = self.encoder(x_patch, x_point, x_ori)
        patch=self.alignment_layer(patch,source_embeddings,source_embeddings)
        point=self.alignment_layer(point,source_embeddings,source_embeddings)
        
        prompt1=[
            f"Background: We are working with a time series dataset that includes a {self.channel}-dimensional time series with a length of {self.win_size}."
            f"Dataset Description: {self.dataset_description}"
                f"Goal: Your objective is to use the patch-wise and point-wise representations of the time series to generate more generalized and robust representations."
                f"Notes: The patch-wise representation captures the global information global information and is generated based on dependencies between patches. "
                f"The point-wise representation captures the local information and is generated based on dependencies between points in a patch. "
                    f"As anomalies are rare and normal points share latent patterns, the two representations should be similar for non-anomalous time points and more different for anomalous time points. "
                f"Constrains: Generate different representations for suspected abnormal points and similar representations for normal points."
                f"Patch-wise representation:"
                ]
        prompt2=["Point-wise representation:"]
        prompt1=self.tokenizer(prompt1, return_tensors="pt", padding=True, truncation=True,
                            max_length=2048).input_ids
        prompt2=self.tokenizer(prompt2, return_tensors="pt", padding=True, truncation=True,
                                max_length=2048).input_ids
        prompt1_embeddings=self.llm_model.get_input_embeddings()(prompt1.to(x.device))
        prompt2_embeddings=self.llm_model.get_input_embeddings()(prompt2.to(x.device))
        prompt1_embeddings = repeat(prompt1_embeddings, "b n d -> (b m) n d", m=B)
        prompt2_embeddings = repeat(prompt2_embeddings, "b n d -> (b m) n d", m=B)
        llm_in=torch.cat([prompt1_embeddings[:,:(prompt1_embeddings.shape[1]-1),:],patch,prompt2_embeddings[:,1:-1,:],point],dim=1)
        llm_out=self.llm_model(inputs_embeds=llm_in).last_hidden_state
        rep_1=llm_out[:,-self.win_size:,:]
        rep_2=llm_out[:,-2*self.win_size:-self.win_size,:]
        
        return rep_1, rep_2

        


class FeatureAlignmentLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(FeatureAlignmentLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.alignment(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def alignment(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,she->blhe", A, value_embedding)

        return output