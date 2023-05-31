
1. Configuring your environment (Prerequisites):

    + Creating a virtual environment in terminal: `conda create -n LINet python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
    
    + (Optional: only for training) Installing [NVIDIA-Apex](https://github.com/NVIDIA/apex) 
    for accelerate training process with mixed precision. 
    [(Instructions)](https://github.com/NVIDIA/apex#linux) (Under CUDA-10.0 and Cudnn-7.4).

<!--2. Downloading Testing Sets: -->
1. Downloading Training and Testing Sets:
    + downloading _**NEW testing dataset**_ (COD10K-test + CAMO-test + CHAMELEON) and move it into `./Dataset/TestDataset/`, 
    which can be found in this [Google Drive link](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/143yHFLAabMBT7wgXA0LrMg) with the fetch code: z83z.
    
    + download **_NEW training dataset_** (COD10K-train) which can be found in this [Google Drive link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1XL6OjpDF-MVnXOY6-bdaBg) with the fetch code:djq2.  Please refer to our original paper for other training data. 
    
<!-- + download **_NEW training dataset_** (COD10K-train + CAMO-train) and move it into `./Dataset/TrainDataset/`, which can be found in this [download link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing). -->
    

<!--3. Training Configuration:

    + Assigning your customed path, like `--save_model`, `--train_img_dir`, and `--train_gt_dir` in `MyTrain.py`.
    
    + Just run it! -->

3. Testing Configuration:

    + After you download all the pre-trained model and testing data, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--model_path`) and assign your the save directory of the inferred mask (`--test_save`)
    
    + Note that we re-trained our model (marked as $\diamondsuit$ in the following figure) equipped with mixed training 
    <p align="center">
        <img src="./Images/new_score_1.png"/> <br />
    </p>

4. Evaluation your trained model:

    + One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
    please follow this the instructions in `main.m` and just run it to generate the evaluation results in 
    `./EvaluationTool/EvaluationResults/Result-CamObjDet/`.

