import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, SequenceClassifierOutput, \
    BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING

from models import task_to_prediction_heads
from models import MultipleChoiceTaskHead

@add_start_docstrings(
    """Bert Model with multi tasks heads on top (linear layers on top of
    the hidden-states output) e.g. for Part-of-Speech (PoS) tagging and Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class MultitaskBertForClassification(BertPreTrainedModel):
    def __init__(self, config, tasks, num_labels_list, use_one_predhead=False):
        super().__init__(config)
        self.task_names = tasks
        self.task_to_labels = dict([(tasks[i], num_labels_list[i]) for i in range(len(num_labels_list))])

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.task_heads = {}
        if use_one_predhead: 
            task_name = tasks[0]
            task_head = task_to_prediction_heads[task_name](config.hidden_size, num_labels_list[0])
            for i, task in enumerate(tasks):
                self.task_heads[task] = task_head
        else:
            for i, task in enumerate(tasks):
                self.task_heads[task] =  task_to_prediction_heads[task](config.hidden_size, num_labels_list[i])
        self.task_head_list = nn.ModuleList(self.task_heads.values())

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        task_name=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
                Classification loss.
            scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
                Classification scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples::

            from transformers import BertTokenizer, BertForTokenClassification
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForTokenClassification.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)

            loss, scores = outputs[:2]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if type(self.task_heads[task_name]) == MultipleChoiceTaskHead:
            ''' Preprocessing for Multiple Choice Tasks: Flattern the features '''
            num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            inputs_embeds = (
                inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                if inputs_embeds is not None
                else None
            )
            kwargs.update({"num_choices": num_choices})

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = bert_outputs[1]
        bert_outputs.pooler_output = self.dropout(pooled_output)

        outputs = self.task_heads[task_name](bert_outputs, labels, **kwargs)
        return outputs

        