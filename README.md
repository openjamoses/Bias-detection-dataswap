### Detection and Evaluation of Bias Inducing Features in Machine learning

This repository contains the replication of our work "Detection and Evaluation of Bias Inducing Features in Machine learning"

We propose an approach for systematically identifying all bias-inducing features of a model to help support the decision-making of domain experts. Our technique is based on the idea of swapping the values of the features and computing the divergences in the distribution of the model prediction using different distance functions; as follows:

- Identify features that potentially introduce bias to the model: Here we identify the features
    that directly and indirectly introduce

- The important features to the model: Given these identified bias inducing features, we wanted to assert whether or not they are important to the model. We used SHAP Values (an acronym from Shapley Additive exPlanations), a model explainable method to explain the individual prediction based on game theoretically optimal Shapley values.

We evaluated this technique using four well-known datasets to showcase how our contribution can help spearhead the standard procedure when developing, testing, maintaining, and deploying fair/equitable machine learning systems.




#### Table of Contents  

 [Detection](#Detection)  
> We describe the detection steps of bias

 [Source](#Source)  
> With this repository is included a complete working source code to reproduce the whole results reported in this Study i.e., src.
A simple configuration may be required before running the code.

In the following, you will find a brief descriptions of what is contained in the source code and how to get started.

 [Datasets](#Datasets)  
> All dataset used for this study can be found in the folder analysis. The sub-directories are organised based on the name of the dataset we considered in this analysis and are self-explainatory.
