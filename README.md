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

This data cannot be used directly in the model without pretreatment. Cleaning steps included careful consideration of the chemicals to include, which would give a positive impact to the model’s performance in the context of accuracy. For example, salts and inorganics were removed as these could affect the descriptor values by introducing unwanted molecule weights. Any faulty smiles were dropped. Multiple datasets of drugs were tested to find the one with the best results.

## Feature Extraction:
In cheminformatics and predictive modeling, descriptor calculation is the process of characterizing a molecule's many qualities using numerical values that represent specific structural and compositional traits. These numerical values, known as descriptors, are used as input in computational models that predict chemical behavior, interactions, and biological activities.
The molecular descriptors which act as our features for our model were calculated with the RDKit library, a powerful cheminformatics toolbox. The dataset, which originally included SMILES (Simplified Molecular Input Line Entry System) representations of chemical compounds, was supplemented with a comprehensive set of molecular descriptors. These descriptors give quantitative information about the molecules' chemical and structural features, which is critical for future predictive modeling.
To construct these descriptors, the RDKit package was used to calculate numerous molecular properties in a methodical manner. A descriptor calculator was populated with all RDKit descriptors to ensure a full depiction of each molecule's attributes. The SMILES strings were processed to generate these descriptors, and the resulting data was incorporated into the current dataset. This stage involved addressing missing values by excluding entries with incomplete descriptor information, ensuring the dataset's integrity.
