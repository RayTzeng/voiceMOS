import torch
import torch.nn as nn

from s3prl.downstream.model import AttentivePooling, MeanPooling

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Model(nn.Module):
    def __init__(self, input_size, regression_output_size, classification_output_size, pooling_name, dim, dropout, activation, num_judges=10000, **kwargs):
        super(Model, self).__init__()
        latest_size = input_size

        self.regression_linears = nn.ModuleList()
        self.classification_linears = nn.ModuleList()
        self.activation = activation
        for i in range(len(dim)):
            # Regression Head
            regression_linear_layer = nn.Sequential(
                nn.Linear(latest_size, dim[i]),
                nn.Dropout(dropout[i]),
                eval(f'nn.{self.activation}')()
            )
            self.regression_linears.append(regression_linear_layer)

            # Classification Head
            classification_linear_layer = nn.Sequential(
                nn.Linear(latest_size, dim[i]),
                nn.Dropout(dropout[i]),
                eval(f'nn.{self.activation}')()
            )
            self.classification_linears.append(classification_linear_layer)

            latest_size = dim[i]

        # Regression Head
        self.regression_output_layer = nn.Linear(latest_size, regression_output_size)
        self.regression_pooling = eval(pooling_name)(input_dim=input_size, activation=self.activation)

        # Classification Head
        self.classification_output_layer = nn.Linear(latest_size, classification_output_size)
        self.classification_pooling = eval(pooling_name)(input_dim=input_size, activation=self.activation)
        self.judge_embbeding = nn.Embedding(num_embeddings = num_judges, embedding_dim=input_size)

    def forward(self, features, features_len, judge_ids):
        time = features.shape[1]
        judge_features = self.judge_embbeding(judge_ids)
        judge_features = torch.stack([judge_features for i in range(time)], dim = 1)
        features = features + judge_features

        # Regression Head
        regression_features, _ = self.regression_pooling(features, features_len)
        for linear in self.regression_linears:
            regression_features = linear(regression_features)
        scores = self.regression_output_layer(regression_features)

        # Classification Head
        classification_features, _ = self.classification_pooling(features, features_len)
        for linear in self.classification_linears:
            classification_features = linear(classification_features)
        logits = self.classification_output_layer(classification_features)

        return scores.squeeze(-1), logits.squeeze(-1)
