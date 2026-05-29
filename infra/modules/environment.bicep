param workspaceName string
param environmentName string

resource env 'Microsoft.MachineLearningServices/workspaces/environments@2024-04-01' = {
  name: '${workspaceName}/${environmentName}'
  properties: {
    description: 'Default sklearn environment'
    //isAnonymous: false
    properties: {
      condaFile: loadTextContent('../envs/hdbscan_env.yml')
      image: 'mcr.microsoft.com/azureml/sklearn-1.2-ubuntu20.04-py38-cpu-inference:latest'
    }
  }
}
