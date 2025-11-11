Here is an explanation about the dirs for preprocess the data for ALTO, fine tune, evaluation and inference. 


1. data preprocess code dir:
uses the raw data in data_dir='dataset/annotated-data-raw'
hackathon-alloauto/alloauto-segmentation-training/data_preprocess
to create the dataset (train+test_val):
    a. run python hackathon-alloauto/alloauto-segmentation-training/data_preprocess/data_preprocess_for_fine_tune.py
    -> this will create the first version of data (that contains mainly switches segments):
        train_segments_updated_v1.csv
        val_segments_updated_v1.csv
        test_segments_updated_v1.csv
    b. after running data_preprocess_for_fine_tune.py run alloauto-segmentation-training/data_preprocess/data_preprocess_for_fine_tune_with_allo_auto.py
    -> this will use the created files from part a, and create the final files for training, val, testing (train+val added here more only allo/auto files each in clean segment):
        dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv
        dataset/preprocessed_augmented/val_segments_more_auto_allo.csv
        dataset/preprocessed_augmented/test_segments_original.csv (same as test_segments_updated_v1.csv)
   #TODO: perhaps create main that runs data_preprocess_for_fine_tune.py and then data_preprocess_for_fine_tune_with_allo_auto.py

2. fine tune:
    a. code for fine tune ALTO architecture: 
    hackathon-alloauto/alloauto-segmentation-training/fine_tune_ALTO_scripts
    alto mul with segmantation reward (this with the Omri's backbone was pushed to huggingface "levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_SEG_RUNI")
    other files: with/without segmentation reward, 4 or 3 ner classes, multiply loss/additive loss
    b. fine tune benchmark scripts: (not ALTO architecture, "simple ner with weight architecture):
    code:
    need only to run these 2 files:
    alloauto-segmentation-training/fine_tune_benchmark_scripts/fine_tune_all_benchmarks_standard_same_params_ALTO.py
    alloauto-segmentation-training/fine_tune_benchmark_scripts/fine_tune_all_benchmarks_standart_3_class_same_params_ALTO.py
    c. the fine tune scripts use GPU (env var for gpu defined in the top of the files)
    
3. evaluation:
    a. code for evaluation:
    the evaluation takes all the fine tuned models and test them with the test data csv file.
    code for evaluate all models with 4 classes ner:
       hackathon-alloauto/alloauto-segmentation-training/evaluation_scripts/evaluate_all_models.py
    code for evaluate all models with 3 classes ner:
       hackathon-alloauto/alloauto-segmentation-training/evaluation_scripts/evaluate_all_models_3_class_focus_copy.py
    b. the evalution use GPU (env var for gpu defined in the top of the files)
    #TODO : combine both evaluation files to one evaluation file.
    

4. inference:
    a. code for inference: 
    in the main of the following code you can give a saved 'X.docx' file, and define the output save dir, to inference with the required model.
    will return each word in the file with it's relative label.
    the code:
       alloauto-segmentation-training/inference/inference_ALTO_Orna_docx_files.py
   
   