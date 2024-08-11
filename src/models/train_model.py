import pandas as pd
import chardet , os , argparse , json , pickle , joblib
from typing import List

import numpy as np


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf

import mlflow
from mlflow.tensorflow import MLflowCallback


# ignore warnings
import warnings
warnings.simplefilter('ignore')


class ModelManager:
    '''
    - This is class that trains multiple learning models on a my dataset.
    - It expects a path to a folder with the following structure
        |
        |_ Dataset Folder
                |
                |_ {league_id}.csv
                |_ ...
    '''
    
    def __init__(self, 
                 data_folder : str , 
                 tracking_uri : str , 
                 predictors : List[str],
                 target_col : str = None, 
                 compile_leagues : bool = False,
                 train_from_compiled : bool = False, 
                 test_size : float = 0.1,
                 one_target : bool = True,
                 target_list : List[str] = None,
                 save_models : bool = False,
                 models_root_path : str = None,
                ):
        
        self.data_path = data_folder
        self.tracking_uri = tracking_uri
        self.predictors = predictors

        # default properties
        self.compile_leagues = compile_leagues
        self.test_size = test_size
        self.one_target = one_target
        self.target_col = target_col
        self.target_list = target_list
        self.save_models = save_models
        self.models_root_path = models_root_path
        self.train_from_compiled = train_from_compiled

        # list for holding skipped leagues
        self.skipped = set()

        # variable for holding experiment name
        self.experiment_name = None

    def read_data(self , data_path : str):
        
        # Detect encoding   
        with open(data_path, 'rb') as f:
            result = chardet.detect(f.read())

        # Read file with detected encoding
        df = pd.read_csv(data_path, encoding=result['encoding'] , index_col=False)

        return df

    def set_tracking(self):
        '''
        This function sets the mlflow tracking folder
        '''
        mlflow.set_tracking_uri(self.tracking_uri)
    
    def set_experiment_name(self , experiment_name : str):
        '''
        This sets an experiment name and returns the experiment id

        Args:
            - league_id : str , the id of the league in str for creating an experiment name
        Returns:
            - experiment_id : str , the id of the created experiment
        '''

        self.experiment_name = experiment_name
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            search_result = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
            experiment_id = search_result[0].experiment_id
        
        return experiment_id

    def drop_games_min(self , df : pd.DataFrame , min_games : int = 3):
        '''
        This function removes games with home or away teams that have below a certain amount of games home or away respectively

        Args:
            - df : pd.DataFrame , the data frame that contains the games. It expects 'num_home_games' and 'num_away_games' columns to exist.
            - min_games : int , the minimum amount of games you want to impose
        Returns:
            - pd.Dataframe : the filtered dataframe
        '''

        dropped_df = df[(df['num_home_games'] >= min_games) & (df['num_away_games'] >= min_games)]
        
        return dropped_df

    def drop_unneccessary_cols(self , df : pd.DataFrame , target_col : str  = None):
        '''
        Drops all the unneccessary columns from the data frame
        '''
        
        # drop the index column
        dropped_df = df.drop(columns=[df.columns[0]])

        # drop all columns except the target col and the selected feature columns        
        if self.one_target:
            target_col = self.target_col

            keep_columns = self.predictors
            remove_columns = [column for column in dropped_df.columns if column not in keep_columns and column != target_col ]

            dropped_df = dropped_df.drop(columns=remove_columns)
        else:
            keep_columns = self.predictors
            remove_columns = [column for column in dropped_df.columns if column not in keep_columns and column != target_col ]

            dropped_df = dropped_df.drop(columns=remove_columns)

        # assert that there aren't empty(null) values
        null_holder = dropped_df.isna().sum().to_dict()
        null_values = 0
        for key in null_holder:
            null_values += null_holder[key]
        
        assert null_values == 0 , 'There are null values in the dataframe'

        return dropped_df
    
    def split_dataset(self , df : pd.DataFrame , target_col : str = None):
        '''
        Splits the data into training and testing sets.

        Args:
            - df : pd.DataFrame
            - target_col : str , a target column  we want to separate out
        Returns:
            - an array of pd.DataFrames in the following order : X_train , X_test , y_train , y_test
        '''
        
        print(f'Splitting for target {target_col}')

        predictor_cols = df.columns.tolist()
        try:   
            if self.one_target:
                predictor_cols.remove(self.target_col)

                # predictors
                X = df.drop(columns=self.target_col)

                # target
                y = df[target_col]
                
                print(f'Splitting data done.')
                return train_test_split(X , y , test_size=0.2 , random_state=100)

            else:
                predictor_cols.remove(target_col)


                # predictors
                X = df.drop(columns=target_col)
                
                # target
                y = df[target_col]
                print(f'Splitting data done.')
                return train_test_split(X , y , test_size=0.1 , random_state=100)

        except Exception as e:
               print(e)

    def train_models(self , 
                     train_x : pd.DataFrame , 
                     train_y : pd.DataFrame , 
                     test_x : pd.DataFrame , 
                     test_y : pd.DataFrame ,
                     experiment_id : str,
                     run_name : str,
                     target_col: str = None,
                    ):
        '''
        Trains , evaluates and logs an Artificial Neurla Network with the following structure:
            - 4 hidden layers : 128 , 128 , 64 , 32 neurons each.
            - ReLU activation function for the hidden layers
            - Softmax activation function for the output layer
        
        Args:
            - train_x : pd.Dataframe , the dataframe with the training predictors
            - train_y : pd.Dataframe , the dataframe with the target class that corresponds to samples in train_x
            - test_x : pd.Dataframe , the dataframe with the testing predictors
            - test_y : pd.Dataframe , the dataframe with the target class that corresponds to samples in test_x
            - experiment_id : str , the experiment id this training run belongs to
            - run_name : str , the name to assign for the runs,
            - target_col : str , the target col we want our models to be trained on
        Returns:
            - None
        '''
        
        # define the models structure
        model = tf.keras.Sequential(
            [
            tf.keras.layers.Input(shape=(len(self.predictors),)),
            tf.keras.layers.Dense(units=128 , activation='relu'),
            tf.keras.layers.Dense(units=128 , activation='relu'),
            tf.keras.layers.Dense(units=64 , activation='relu'),
            tf.keras.layers.Dense(units=32 , activation='relu'),
            tf.keras.layers.Dense(units=len(self.target_list) , activation='softmax')
            ]
        )   

        # define the models optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01,   
            beta_1=0.9,             
            beta_2=0.999,           
            epsilon=1e-4            
        )

        # compile the model
        model.compile(optimizer=optimizer , loss='categorical_crossentropy' , metrics=['accuracy'])

        # turn off autologging for mlflow
        mlflow.tensorflow.autolog(disable=True)

        # define an earlystopping callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta = 1e-5,
            patience=5,
            mode='auto',
            verbose=1,
            restore_best_weights=True
        )

        # split the test data into validation and holdout sets
        x_validation , x_holdout , y_validation , y_holdout = train_test_split(test_x , test_y , test_size=.5 , random_state=7)

        # start the learning 
        with mlflow.start_run(run_name='Neural_Net_Training' , experiment_id=experiment_id) as run:
            history = model.fit(x=train_x , y=train_y , batch_size=128 , validation_data=(x_validation , y_validation) , callbacks=[earlystop_callback , MLflowCallback(run)] , epochs=100)

            # test the model on the holdout set and log the performance
            predictions = model.predict(x_holdout)

            max_indices = np.argmax(predictions, axis=1)
            one_hot_encodings = np.zeros_like(predictions)
            one_hot_encodings[np.arange(predictions.shape[0]), max_indices] = 1

            accuracy = accuracy_score(y_true=y_holdout , y_pred=one_hot_encodings)

            mlflow.log_metrics({'accuracy_score' : accuracy} , run_id=run)

            # log the model artifact if save model is enabled
            if self.save_models:
                if self.models_root_path != None:
                    mlflow.tensorflow.log_model(model , f'models/Neural_Net/')
                else:
                    print("Model path not defined , model not saved!")

    def pipe(self , data : pd.DataFrame , experiment_id : str , run_name : str , league_id : str = None , target_name : str = None):
        '''
        This function connects every implemented above.

        Args:
            - data : pd.DataFrame , a dataframe that contains the data
            - experiment_id : str , the id of the mlflow experiment
            - run_name : str , the run name of the mlflow run
            - target_name : str, the target column we want to train the pipeline on.
        Returns:
            - None 
        '''


        try:
            # drop games with less than 3 matches by each teams
            dropped = self.drop_games_min(df=data)

            # drop unneccessary column
            dropped = self.drop_unneccessary_cols(df=dropped , target_col=target_name)
            
            print(f'Remaining Data {dropped.shape}')


            # split the dataset
            X_train , X_test , y_train , y_test = self.split_dataset(df=dropped , target_col=target_name)

            # check if there are enough samples to teach the model
            if X_train.shape[0] >= 100:

                # train and log the models
                self.train_models(
                    train_x=X_train,
                    train_y=y_train,
                    test_x=X_test,
                    test_y=y_test,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    target_col=target_name
                )

                # find the best model and log it
                self.log_best_model(target=target_name)

            else:

                print(f'Skipped league {league_id} becuase it has low games')
                self.skipped.add(league_id)

        except Exception as e:
            print(f'Skipped league {league_id} because of {e}')
            self.skipped.add(league_id)
        
    def train_individual(self):
        '''
        This function gets called if the compile_leagues is set to False. It trains the models on individual league data.
        '''
        
        leagues = os.listdir(self.data_path)
        print('---Training on Leagues Started---')
        for league in leagues:
            print('---Load League---')            
            league_id = league.split('_')[0]
            league_path = os.path.join(self.data_path , league)
            league_df = self.read_data(data_path=league_path)

            run_name = f'{league_id}_{self.target_col}'
            experiment_id = self.set_experiment_name(experiment_name=f'{league_id}_training')

            print(f'---Training started on league {league_id} data---')
            if self.one_target:
                self.pipe(
                    data=league_df,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    target_name=self.target_col
                )
            else:
                # train on multiple targets
                for target in self.target_list:
                    run_name = f'compiled_{target}'
                    self.pipe(
                        data=league_df,
                        experiment_id=experiment_id,
                        run_name=run_name,
                        target_name=target 
                    )
                print(f'---Training finished on league {league_id} data---')

        print('---Training on Leagues Finished---')
    
    def train_compiled(self , experiment_id : str , run_name : str = None):
        '''
        This function gets called if the compile_leagues is set to True. It compiles the data of each league into one dataframe and uses that for training.
        Args:
            - experiment_id : str , the id of the mlflow experiment
            - run_name : str , the run name of the mlflow run
        Returns:
            - None
        '''

        df = pd.DataFrame()
        leagues = os.listdir(self.data_path)
        
        print('---Loading and Compiling Data---')
        for league in leagues:
            league_id = league.split('.')[0]
            league_path = os.path.join(self.data_path , league)
            
            try:
                league_df = self.read_data(data_path=league_path)

                # drop min games , skip leages with less than a 100 game
                test_df = self.drop_games_min(df=league_df)
                league_games = test_df.shape[0]
                if league_games < 100:
                    print(f'Skipped league {league_id} because it had only {league_games}')
                    self.skipped.add(league_id)
                    continue

                # check if there are columns missing by invoking the function below
                test_df = self.drop_unneccessary_cols(df=league_df , target_col=self.target_list[0])

                df = pd.concat([df , league_df])
                print(f'Added league {league_id} to the training data')
                print(f'Collected data points : {df.shape}')
        
        # check if save compiled is true and then save the compiled csv to the folder 'scraped_data/compiled'
        # with the file name 'all_leagues'

            except Exception as e:
                print(f'Skipped league {league_id} becuase of {e}')
                self.skipped.add(league_id)
                
        print(f'Collected data points : {df.shape}')
        print('---Training Started---')
        
        if self.one_target:
            self.pipe(
                data=df,
                experiment_id=experiment_id,
                run_name=run_name,
                target_name=self.target_col
            )
        else:
            # train on multiple targets
            for target in self.target_list:
                run_name = f'compiled_{target}'
                self.pipe(
                    data=df,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    target_name=target 
                )   

                print(f'--- Trained for {target} ---')
            
        print('---Training Ended---')

    def train_frm_compiled(self , experiment_id : str, run_name : str = None):
        '''
        This function gets called if train_from_compiled is set to True. It reads in the csv, persumed to be compiled, and then uses it for training.
        Args:
            - experiment_id : str , the id of the mlflow experiment
            - run_name : str , the run name of the mlflow run 
        Returns:
            - None
        '''

        df = self.read_data(self.data_path)

        print('---Training Started---')
        
        if self.one_target:
            self.pipe(
                data=df,
                experiment_id=experiment_id,
                run_name=run_name,
                target_name=self.target_col
            )
            self.log_best_model(target=self.target_col)
        else:
            # train on multiple targets
            for target in self.target_list:
                run_name = f'compiled_{target}'
                self.pipe(
                    data=df,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    target_name=target 
                )
                self.log_best_model(target=target)
                print(f'--- Training on {target} finished ---')

        print('---Training Ended---')

    def log_best_model(self , target : str):
        # find the best model
        run_name = f'compiled_{target}_%'
        runs = mlflow.search_runs(experiment_names=[self.experiment_name] , filter_string=f'''attributes.run_name LIKE "{run_name}" AND attributes.status = "FINISHED"''' , order_by=["metrics.accuracy_score DESC"])        
        best_run = runs.head(n=1)

        # get the artifacts uri
        artifact_uri = best_run["artifact_uri"][0]

        # relative path
        uri_folder_name = self.tracking_uri
        relative_path = artifact_uri.split(uri_folder_name)[1]
        relative_folder_path = f"./{uri_folder_name}{relative_path}/models"

        # model name
        name = os.listdir(relative_folder_path)[0]
        complete_model_folder_path = os.path.join(relative_folder_path , name)
        target = os.listdir(complete_model_folder_path)[0]
        model_path = os.path.join(complete_model_folder_path , target , 'model.pkl')

        # copy the model
        with open(model_path , 'rb') as model_pkl:
            loaded_model = pickle.load(model_pkl)
        
        if 'objects' not in os.listdir('./models/'):
            os.mkdir('./models/objects')
        joblib.dump(loaded_model , filename=f'./models/objects/{target}.joblib' , compress=3)

        # get model type
        run_name = best_run['tags.mlflow.runName'][0]   
        model_type = run_name.split('_')[-1]


        # save the metric and params
        metrics = dict()

        for column in best_run.columns:
            if 'metrics' in column:
                metric_name = column.split('.')[1]
                metric_value = best_run[column][0]
                if not pd.isna(metric_value):
                    metrics[metric_name] = metric_value
            


        # add the models performance
        path_to_metrics = './models/metrics.json'
        try:
            # try opening an existing file
            with open(path_to_metrics , 'r') as file:
                model_performance = json.load(file) 
            
            # update it with the new metric
            with open(path_to_metrics , 'w') as file:
                model_performance[target] = {"metrics" : metrics , "model_type" : model_type}
                json.dump(model_performance , file , indent=4)

        except Exception as e:
            with open(path_to_metrics , 'w') as file:
                model_performance = {target : {"metrics" : metrics , "model_type" : model_type}} 
                json.dump(model_performance , file , indent=4)        
 
    def fit(self , experiment_name : str = None):
        # set the local tracking folder
        self.set_tracking()
        
        if self.compile_leagues == True:
            # set the experiment name and obtain the experiment id
            experiment_id = self.set_experiment_name(experiment_name=experiment_name)
            print('---Using compiled data---')
            self.train_compiled(experiment_id=experiment_id , run_name=f'compiled_{self.target_col}')
        elif self.compile_leagues == False and self.train_from_compiled == False:
            print('---Using individual leauges---')
            self.train_individual()
        elif self.train_from_compiled == True:
            # set the experiment name and obtain the experiment id
            experiment_id = self.set_experiment_name(experiment_name=experiment_name)
            print('---Using an already compiled dataset---')
            self.train_frm_compiled(experiment_id=experiment_id , run_name=f'compiled_{self.target_col}')
        else:
            print('You should choose how you want to train the models!!')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path' , default='./data/processed/final.csv')
    args.add_argument('--mlflow_tracking_uri' , default='mlruns')
    args.add_argument('--experiment_name' , default='main experiment')

    parsed_args = args.parse_args()

    data_path = parsed_args.data_path
    tracking_uri = parsed_args.mlflow_tracking_uri
    experiment_name = parsed_args.experiment_name

    # a set for holding improper leagues , i.e leagues with to little data
    bad_leagues = set()

    target_params = ['home_win' , 'draw' , 'away_win']
    features = ['home_attack_strength', 'home_defence_strength', 'away_attack_strength',
           'away_defence_strength', 'home_expected_goal', 'away_expected_goal']

    # create the mlflow directory / if it doesn't exist
    all_folders = os.listdir(os.getcwd())
    if tracking_uri not in all_folders:
        os.mkdir(tracking_uri)
    
    # instantiate the model managet for training
    trainer = ModelManager(
            data_folder= data_path, 
            tracking_uri=tracking_uri, 
            target_list=target_params,
            predictors=features,
            one_target=False,
            train_from_compiled=True,
            test_size=0.3,
            save_models=True,
            models_root_path='./saved_models'
            )

    # invoke the fit method to start the training
    trainer.fit(experiment_name=experiment_name)
