The goal of this project is:
-
- To understand the entire cycle of development involved in training and hosting for inferencing a DL model.
- To understand and learn how to develop a robust, scalable and flexible training pipelines based on industry standard practices.


Platforms, Tools and Tech Stacks used:
-
- python
    - pytorch
    - FastAPI
    - uv for dependency management
- mlflow
- aws

Progress, difficulties and insights:
- 
1. I started the project with just writing every code in a jupyter notebook. The data ETL part, model architecture, the training loop, everything in a single jupyter notebook.
2. Dataset used: Animal-10 from Kaggle. Model used: DenseNet121 - Freezed the Convolutional Network(feature identifiers) and replaced the fully connected layers(classifiers) for retraining.
3. Then I referred to various projects in Github, blogs, articles and AI assistance(basically used cursor/claude code to look at my jupyter notebook and asked it to give suggestions on how best to refactor my code) to refactor my code into modularized components for scalability, flexibility and robustness.
4. At this point, I also created another branch called AI_version and gave cursor the full control to refactor the code and make the changes. This is just to compare my version of the refactoring with the AI version of refactoring.
5. Once I refactored the code and tested it for any bugs(ran the training pipeline in windows CPU only machine for just 1 step to make sure there is no bugs), so that I don't waste money in Cloud by debugging the scripts rather than training because it will be GPU machine and it will be costlier as well.
6. I want to utilize the free credits from the Azure but got to know that the free tiers are not applicable for GPU machines and also there were not a lot of options for the GPU machines. Mostly, I saw only high end GPU clusters(A100, H100) and all I wanted is a GPU with 15 Gigs of memory and some decent number of cuda cores(note this is the first time I am using Azure).
7. Then due to the above restrictions I decided to use AWS, g4dn.xlarge, the basic GPU enable linux machine for the training. I faced some difficulties here as well during the training because, I was using spot shared instances as opposed to dedicated instances to cut down the costs
8. At this point, I was also thinking to wrap the dependencies into a Docker image so that I don't have to waste time in setting up the environment inside the cloud GPU machine but since the dependencies weren't much, I didn't go for this option.
9. I used MLflow(note this is the first time I use mlflow) to monitor my training progress and to maintain different versions of the model and also to host the model post training. I felt this is an easy tool to implement and master. 
10. Post training, I wanted to test the model's accuracy during inferencing. This is where I faced multiple challenges. 
    - The first one is that I couldn't get MLflow to host the model, as by default, it tries to load the model in gpu. **SOLUTION:** I wrote my own simple python logic using FastAPI to host the model in GPU
    - The second problem is that my model's accuracy was very poor during inferencing even though it's test accuracy was about 89% after 1st epoch. What's more puzzling was that the dataset used for inferencing was the same as the one used for training and testing. **SOLUTION:** I checked all the possible problems as to understand why this is happening. I made sure that the preprocessing pipeline was the same. But this didn't help. I also increased the total epochs from 1 to 3 but it was also not enough. I am sure, If I improve the model architecture, I will get the good results. But I left the problem here as the aim of this project is the whole cycle of DL model development and not the model architecture itself.
    - The third one is I had the dilemma of whether to keep the preprocessing logic in the client side application or wrap it with the model hosting itself. I decided to wrap it with the model hosting itself, so that the depencies are not much of problem. Here, I had to define a custom model in MLflow to include the preprocessing step as well during inferencing. Also, remeber to define the signature of the model(input type) to mlflow while saving the model itself, so that the Mlflow model server interprets the coming input accordingly
11. So, basically, if you see, I apply the first principles thinking a lot in the projects I work. I will pace through the entire implementation of the project or product or pipeline or anything for that matter. Once I quickly implement it, I will get a good idea of the variables inside the whole project or product. I treat them as small components which make the whole and I know that improving each piece would improve my overall result ot accuracy of the project.
12. Finally, I hosted the model and created an API endpoint. Then I built a small frontend for the users to interact with the backend confirgured to access the model's API. This way, I can work on load balancing and optimize for low latency and so on. To be honest, for the front-end, I used Gemini CLI(this was totally free while I was working on this project)