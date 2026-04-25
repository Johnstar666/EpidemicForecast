from models import EpiLLM_Gpt2,EpiLLM_Gemma1,EpiLLM_DS3


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'EpiLLm_Gpt2': EpiLLM_Gpt2,
            'EpiLLm_gemma1':EpiLLM_Gemma1,
            'EpiLLm_DS3':EpiLLM_DS3
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
