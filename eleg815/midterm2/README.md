### Midterm Task: Building and Training a Neural Network

#### Objective:
Your task is to build and train a Neural Network in Python using PyTorch to perform a classification task.  

#### Dataset:
The dataset for this task is available on Canvas under **Modules -> midtermii_data**.  
It includes the following files:  
- `midterm2_training_data.npy`: contains the features for training.  
- `midterm2_training_labels.npy`: contains the corresponding labels.  

You may split the dataset into training and validation sets as you see fit.  

#### Evaluation:
Your model's performance will be evaluated on a hidden testing set.  
- Full points will be awarded to the student whose model achieves the highest performance on the testing set.  
- Partial credit will be assigned based on the relative rank of your model's performance compared to other students.  

#### Submission Requirements:
You must submit the following two files:
1. A Python script named `[studentname_udid].py` with all the code.
2. A checkpoint file of your trained model saved as `[studentname_udid].pth`  

#### Python Script Requirements:
Your script, `[studentname_udid].py`, must accept the following command-line arguments:
1. Boolean value that sets the testing state
2. Path to the checkpoint file  
3. Path to the testing dataset (formatted similarly to `midterm2_training_data.npy`)  
4. Path to the output directory

The script must perform the following tasks:  
1. Load the trained model from the checkpoint.  
2. Pass the testing data through the model to generate predictions.  
3. Save the predicted labels as a `.npy` file named `[studentname_udid].npy` in the specified output directory. The predicted labels should have the same format as `midterm2_training_labels.npy`. 

#### Example Command:
```bash
python nestor_999999.py True ./model_checkpoint/nestor_999999.pth ./midterm2_testing_data.npy ./predictions/
```

---

### Additional Notes:
- **File Formats:** Ensure your `.npy` files are correctly formatted and compatible with the evaluation script.  
- **Coding Style:** Write clean, well-documented code. Use appropriate comments where necessary to explain key parts of your implementation.  
- **Testing Your Script:** Test your script thoroughly to ensure it works as intended.
