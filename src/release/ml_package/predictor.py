import os, joblib
import numpy as np

# ignore warnings
import warnings
warnings.simplefilter('ignore')

class Predictor:
    ''''''
    module_path = os.path.dirname(__file__)

    def __init__(self):
        self.models_folder = os.path.join(Predictor.module_path , 'models')
        self.models = {}
        self.load_models()
        
    def load_models(self):
        '''Loads all the models in the models folder and stores them in a dictionary with their targets as their keys.'''

        models = os.listdir(self.models_folder)
        for model in models:
            model_path = os.path.join(self.models_folder , model)
            model_target = model.split('.')[0]
            model_object = joblib.load(model_path)
            self.models[model_target] = model_object
        print('--- Models loaded ---')


    def predict(self, hm_att_str : float, hm_def_str : float, aw_att_str : float, aw_def_str : float, hm_exp, aw_exp : float):
        '''
        Predicts the outcome of games using all of the models. Returns predictions, either 0 or 1 , and puts them in a dictionary.
        Args:
            - hm_att_str: float , home attack strength 
            - hm_def_str: float , home defence strength
            - aw_att_str: float , away attack strength
            - aw_def_str: float , away defence strength
            - hm_exp: float , home expected goal
            - aw_exp: float , away expected goal
        Returns:
            - predictions : dict , a dictionary where keys are targets and values are the predicted outcomes of the game
        '''
        predictions = {}
        x = np.array([hm_att_str , hm_def_str , aw_att_str , aw_def_str , hm_exp , aw_exp]).reshape(1,-1)
        for target in self.models:
            model = self.models[target]
            result = model.predict(x)
            predictions[target] = result.item()

        return predictions

        


if __name__ == '__main__':
    predictor = Predictor()
    

    # test game => Platinum vs Simba Bhora
    #              League : Zimbabwe - Premier Soccer League
    #              Date : 22/06/2024
    #              Result : 1 - 0
    test_1 = predictor.predict(hm_att_str=1.5025684931507,
                      hm_def_str=0.92889908256881,
                      aw_att_str=1.7029816513761,
                      aw_def_str=1.271404109589,
                      hm_exp=2.0660316780822,
                      aw_exp=1.2772362385321)

    print(test_1)

    # test game => Dinamo Samarqan vs Andijan
    #              League : Uzbekistan - Super League
    #              Date : 22/06/2024
    #              Result : 1 - 1
    test_2 = predictor.predict(hm_att_str=0.75238095238095,
                      hm_def_str=0.71011235955056,
                      aw_att_str=1.4794007490637,
                      aw_def_str=1.0031746031746,
                      hm_exp=1.0031746031746,
                      aw_exp=1.1835205992509)
    
    print(test_2)