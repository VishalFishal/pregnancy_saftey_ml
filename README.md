# Introduction
The development of this machine learning model is facilitated by the need for a good method to predict if a certain drug is safe to consume for a pregnant lady, without adverse effects to her altered bodily state during pregnancy or her foetus. The idea behind this model is that chemicals of similar structures and properties carry similar characteristics in similar environments. By training the model on drugs whose structures, and chemical descriptors are well known, we can (through this ideology) also predict whether a drug is safe to consume or not. 
The dataset used should contain a sizeable portion of chemicals with 1-dimensional and 2-dimensional descriptors at least. 3-dimensional descriptors and fingerprints may be included as well but must be done carefully to avoid an over-complicated, or overfitted model. Techniques under feature selection may be used to cut down on useless or less impactful features for the model. 
The model is entirely coded in Python. Utilizing the scikit-learn library, we can easily create numerous models to test on the required dataset rapidly. The library also contains modules for the testing and validation of each model. The library also lets us apply various machine learning algorithms effortlessly, as they are but a simple module to include into the code. Both Jupyter and Spyder were used as development environments for the project.

## Data Preparation:
The dataset consists of the following columns:
1.	Name of the medication
2.	CAS Number
3.	SMILES
4.	The drug’s category
5.	Classification of the drug based on their therapeutic use or the system of the body they target.

The drugs are classified in such a way that they are annotated with safety classifications (safe or unsafe) for use during pregnancy with the help of category mapping. 
We categorized drugs from categories A and B as "safe." These categories represent drugs with no demonstrated risk to the fetus (A) or those that have not shown risk in animal studies and lack human data (B).
We categorized drugs from categories C, D, and X as "not safe." Category C includes drugs with adverse effects in animal studies and insufficient human data, while categories D and X include drugs with known risks to the fetus.

This data cannot be used directly in the model without pretreatment. Cleaning steps should carefully consider the chemicals to include, which would give a positive impact to the model’s performance in the context of accuracy. For example, salts and inorganics were removed as these could affect the descriptor values by introducing unwanted molecule weights. Any faulty smiles were dropped. Multiple datasets of drugs were tested to find the one with the best results.

## Results:
1. Test Set Performance:
2. Accuracy: 0.8205128205128205
3. ROC AUC Score: 0.8838644997889404
4. Kappa Score: 0.6384105960264901
5. MCC: 0.6400865768675789

Classification Report:
```
              precision    recall  f1-score   support

           0       0.84      0.77      0.80        92
           1       0.81      0.86      0.84       103

    accuracy                           0.82       195
   macro avg       0.82      0.82      0.82       195
weighted avg       0.82      0.82      0.82       195

Confusion Matrix:
[[71 21]
 [14 89]]
```

## Disclamers:
### This is not tested code! Please do not use the model for genuine medicinal research without reading the following:
1. The dataset was not created by me, I have no hand in the selection of the drugs.
2. Other enhancements that were tested and gave similar results were RFECV to reduce the number of features. This is not present in the current version of the code.
3. Features for the drugs can be retrieved using the PaDEL software, which has many more descriptors and fingerprints. Be warned that this software takes time to generate all of the desired data.
4. Currently the model uses all the descriptors provided in the RDKit module. No feature studies were done due to lack of domain knowledge.
5. It is highly likely that this model is severely overfitted due to the above facts.

#### Please consider improving this model!
