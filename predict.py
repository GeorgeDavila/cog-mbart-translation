# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import pipeline

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"
device = "cuda"

#model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#tokenizer.src_lang = "en_XX"
#encoded_hi = tokenizer(sourceText, return_tensors="pt")
#generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id[languageCode])
#translation1 = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

langList = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=TOKEN_CACHE
        )
        model = MBartForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.model = model.to(device)

    def predict(
        self,
        text2translate: str = Input(
            description="Text you want to translate.", 
            default="Hello have you seen my dog?"
            ),
        sourceLanguage: str = Input(
            description="Language of your input text.", 
            choices=["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"],
            default="en_XX"
            ),
        targetLanguage: str = Input(
            description="Language you want to translate to.", 
            choices=["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"],
            default="es_XX"
            ),
        doSentenceLevelTranslation: bool = Input(
            description="Translate at Sentence level? May be necessary for longer bodies of text.", 
            default=False,
            ),
    ) -> str:
        """Run a single prediction on the model"""

        def doTranslate(sentence2translate):
            self.tokenizer.src_lang = sourceLanguage
            encoded_hi = self.tokenizer(sentence2translate, return_tensors="pt").to(device)
            generated_tokens = self.model.generate(**encoded_hi, forced_bos_token_id= self.tokenizer.lang_code_to_id[targetLanguage])
            outputTranslate = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return outputTranslate

        if doSentenceLevelTranslation:
            text2translate = text2translate.replace("?", ".").replace("!", ".") #replace sentence delimiters
            text2translateList = text2translate.split(". ") #space after period so we dont split on abbreviations

            outputSentences = []
            for i in text2translateList:
                outputSentences.append(doTranslate(i))

            output = '. '.join(outputSentences)
        else:
            output = doTranslate(text2translate)

        return output
