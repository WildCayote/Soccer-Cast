import os , shutil , argparse

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
    if os.path.exists(MODELS_PATH) :
        shutil.rmtree(MODELS_PATH)
    
    # create a models path
    os.mkdir(MODELS_PATH)

    # copy the models from the models directory to the ./src/release/ml_package/models
    models = os.listdir(path=models_path)
    for model in models:
        model_path = os.path.join(models_path , model)
        print(model_path)
        # copy the model to the new directory
        new_path = os.path.join(MODELS_PATH , model )
        shutil.copy(model_path , new_path)
    
    print('-- Models copied to package --')

    # then convert the ml_pacakge into a zip file
    if os.path.exists(ZIP_NAME + '.zip') : os.remove(ZIP_NAME + '.zip')

    shutil.make_archive(ZIP_NAME , 'zip' , PACKAGE_DIRECTORY)

    print('-- Package created --')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--models_folder' , default='./models/objects')
    args.add_argument('--test' , default='false')

    parsed_args = args.parse_args()

    models_folder = parsed_args.models_folder
    test_run = parsed_args.test

    if test_run == 'true':
        # create a package without the models copied inside it
        shutil.make_archive(ZIP_NAME , 'zip' , PACKAGE_DIRECTORY)
    elif test_run == 'false':
        # create a package with the models copied inside it
        create_release(models_path=models_folder)
    else:
        print('Please set the --test argument to either false or true!')
