A complete run of the project, running INSERT COMMAND will:
    - /run scripts use download_scripts and populate /data/data with /test, /train, and /val, their should be a choice for the preprocess pipeline before we populate our datasets. preprocess can use other techniques, like augmentation in different ways.
    - Pass the preprocessed data into the model, training it with given parameters. update logs with a formatted file name.
    - model saves to /models for future use, with formats to match with logs as well
    - model output can use post_pipeline to help boost answers/ performance
    - `