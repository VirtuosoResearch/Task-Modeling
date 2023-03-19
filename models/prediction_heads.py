from turtle import forward
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, SequenceClassifierOutput, MultipleChoiceModelOutput
'''
Write prediction head logics for each task
'''
class SequenceClassificationTaskHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(SequenceClassificationTaskHead, self).__init__()
        self.num_labels = num_labels
        self.pred_head = nn.Linear(hidden_size, num_labels) 

    def forward(self, bert_outputs, labels, **kwargs):
        pooled_output = bert_outputs[1]

        logits = self.pred_head(pooled_output)
        num_labels = self.num_labels

        loss = None
        if labels is not None:

            if num_labels == 1:
                problem_type = "regression"
            elif num_labels > 1:
                problem_type = "single_label_classification"
            # else:
            #     problem_type = "multi_label_classification"

            if problem_type == "regression":
                loss_fct = MSELoss()
                if num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels)
            # elif problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)

        predictions = logits.argmax(dim=-1) if self.num_labels != 1 else logits.squeeze()
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            predictions=predictions
        )

class MultiRCTaskHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(MultiRCTaskHead, self).__init__()
        self.num_labels = num_labels
        self.pred_head = nn.Linear(hidden_size, num_labels) 

    def forward(self, bert_outputs, labels, idx, **kwargs):
        pooled_output = bert_outputs[1]

        logits = self.pred_head(pooled_output)
        num_labels = self.num_labels

        assert labels is not None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        predictions = logits.argmax(dim=-1)
        predictions = [{"idx": idx[i], "prediction": predictions[i]} for i in range(len(idx))]
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            predictions=predictions
        )

class WICTaskHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(WICTaskHead, self).__init__()
        # the hidden size times three since we concatenate the representation of two marked word
        self.pred_head = nn.Linear(hidden_size*3, num_labels) 
        self.num_labels = num_labels

    def forward(self, bert_outputs, labels, first_word_pos, second_word_pos, **kwargs):
        last_hidden_state = bert_outputs[0]
        pooled_output = bert_outputs[1]

        batch_idx = torch.arange(pooled_output.size(0))
        first_word_output = last_hidden_state[batch_idx, first_word_pos]
        second_word_output = last_hidden_state[batch_idx, second_word_pos]
        assert first_word_output.size() == pooled_output.size() and second_word_output.size() == pooled_output.size()

        wic_output = torch.cat([pooled_output, first_word_output, second_word_output], dim=1)

        logits = self.pred_head(wic_output)
        num_labels = self.num_labels

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        predictions = logits.argmax(dim=-1)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            predictions=predictions
        )

class SpanClassificationTaskHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(SpanClassificationTaskHead, self).__init__()
        self.pred_head = nn.Linear(hidden_size*3, num_labels) 
        self.num_labels = num_labels

    def forward(self, bert_outputs, labels, span1_masks, span2_masks, **kwargs):
        last_hidden_state = bert_outputs[0]
        pooled_output = bert_outputs[1]

        batch_idx = torch.arange(pooled_output.size(0))
        first_word_output = last_hidden_state * span1_masks.unsqueeze(-1).float()
        first_word_output = first_word_output.sum(dim=-2)/span1_masks.sum(dim=-1).unsqueeze(-1) # compute mean
        second_word_output = last_hidden_state * span2_masks.unsqueeze(-1).float()
        second_word_output = second_word_output.sum(dim=-2)/span2_masks.sum(dim=-1).unsqueeze(-1) # compute mean
        assert first_word_output.size() == pooled_output.size() and second_word_output.size() == pooled_output.size()

        wsc_output = torch.cat([pooled_output, first_word_output, second_word_output], dim=1)

        logits = self.pred_head(wsc_output)
        num_labels = self.num_labels

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels)
        
        predictions = logits.argmax(dim=-1)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            predictions=predictions
        )

class MultipleChoiceTaskHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(MultipleChoiceTaskHead, self).__init__()
        num_labels = 1 # force multiple choice head labels to 1 since we will reshape
        self.pred_head = nn.Linear(hidden_size, num_labels) 
        self.num_labels = num_labels

    def forward(self, bert_outputs, labels, num_choices, **kwargs):
        pooled_output = bert_outputs[1]
        logits = self.pred_head(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        predictions = reshaped_logits.argmax(dim=-1)
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            predictions=predictions
        )