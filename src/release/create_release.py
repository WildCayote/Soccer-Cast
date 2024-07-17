import os , shutil , argparse , joblib

MODELS_PATH = "./src/release/ml_package/models"
PACKAGE_DIRECTORY = "./src/release/ml_package"
ZIP_NAME = './src/release/ml_package'

def create_release(models_path : str):
    '''
    A script that will create a python module package which contains the trained models.
    Args:
        - models_path : str , the path to the models
    Returns:
        - Null
    '''
    
    # first clean the packages model directory
    if os.path.exists(path=MODELS_PATH) :
        shutil.rmtree(MODELS_PATH)
    
    # create a models path
    os.mkdir(MODELS_PATH)

    # copy the models from the models directory to the ./src/release/ml_package/models
    models = os.listdir(path=models_path)
    for model in models:
        model_path = os.path.join(models_path , model)
        # read the model
        loaded_model = joblib.load(model_path)

        # paste it to the new directory
        new_path = os.path.join(MODELS_PATH , model )
        joblib.dump(loaded_model , new_path)
    
    print('-- Models copied to package --')

    # then convert the ml_pacakge into a zip file
    if os.path.exists(ZIP_NAME) : os.remove(ZIP_NAME)

    shutil.make_archive(ZIP_NAME , 'zip' , PACKAGE_DIRECTORY)

    print('-- Package created --')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--models_folder' , default='./models/objects')

    parsed_args = args.parse_args()

    models_folder = parsed_args.models_folder

    create_release(models_path=models_folder)

