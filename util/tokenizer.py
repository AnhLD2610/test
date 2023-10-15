"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import warnings

warnings.filterwarnings("ignore")
import spacy


class Tokenizer:

    def __init__(self):
        self.spacy_en = spacy.load('en_core_web_sm')


    def tokenize(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

# from transformers import T5Model, T5Tokenizer
# # from transformers import AutoTokenizer

# model = T5Model.from_pretrained("t5-base")
# tok = T5Tokenizer.from_pretrained("t5-base",model_max_length=512)

# enc = tok("some text.", return_tensors="pt")

# input_ids=enc["input_ids"], 
# print(input_ids),
# output = model.encoder(
#     input_ids=enc["input_ids"], 
#     attention_mask=enc["attention_mask"], 
#     return_dict=True
# )
# # get the final hidden states
# emb = output.last_hidden_state
# print(emb)
# print(emb.shape)


# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# # model = T5ForConditionalGeneration.from_pretrained("t5-small")

# # input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# # input_ids = tokenizer.vocab_size("The <extra_id_0> walks in <extra_id_1> park")
# input_ids = tokenizer.get_vocab()


# # labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# print(input_ids)
# # the forward function automatically creates the correct decoder_input_ids
# # loss = model(input_ids=input_ids, labels=labels).loss
# # loss.item()