Context
Showcase the champion/challenger architecture for Azure ML Workspace.
We want to use a Credit Risk simplified model (one ML model only) for semplify a possible real-world use case.
The interesting thing in a Credit Model is that it uses a delayed ground truth (simulated in this simple case) 
and we need to choose retrain data on a data window, representative of old and new (drifted) data in order to cover old and new data pattern for re-training the model.

Infrastructure:
Workspace and registry are created via Bicep (see C:\Carlo\Azure\AI-300\mlops-cdr-demo)
YAML conda Enviroment and datasets are saved in a ML registry for the training job to reference them.

Training and deployment:
The Model is trained using a Python Job or Pipeline and saved in the ML registry as well.
After that, it is deployed for inference and tested. This is our champion model.
We need also a simple client and inference data to send enough inferences requests to generate logs.
Client authenticates via a client certificate (app alraedy registered with public key in ENTRA ID)

Monitoring:
Monitoring of inference performance, model decay and data drift detections, and re-training when needed (thresholds).
This includes the Events and GitHub actions necessary.
We also need to be aware of the delayed truth for the data, and the creation of the an improved baseline for training data: 
- part of the old (previous training and inference sets where champion model perform well) 
- and part of the 

Main part: Champion/Challenger architecture pattern
Here we need to show how to correctly implement the Champion/Challenger pattern.
After a succesful re-training showing the new model (ModelB) performs better than the current model in production (ModelA)
- deploy ModelB as well in production, but only shadowing/mirroring ModelA: still all inference answers come from ModelA
- Collect and analyse ModelB and ModelA responses: if ModelB appears to perform better --> 10% of traffic to ModelB, 90% to ModelA
- If ModelB continue to perform better, increase percentage
- Until ModelB is the new Champion with 100% traffic and ModelA is only mirroring ModelA (as a roll back option)
BONUS: a simple GUI showing side-by-side the performance of the champion and challenger in the different stages

The GitHub repository should contain
- Bicep IaC code (already creted)
- creation of Yaml based conda environment in the ML registry
- Python code for training the model
- CSV training data 
- CSV inference data
- CSV re-training data
- GitHub actions for re-training
- Python code for client
- etc.
