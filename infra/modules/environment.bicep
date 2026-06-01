param workspaceName string
param environmentName string

//!!! Re-declare the workspace as an "existing" resource using the passed string
resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = {
  name: workspaceName
}

// ==========================================
// AzureML Environment Container (Parent)
// ==========================================
resource environmentContainer 'Microsoft.MachineLearningServices/workspaces/environments@2025-12-01' = {
  name: environmentName
  parent: workspace
  properties: {
    //description: 'Credit Risk Demo Environment Container'
    isArchived: false
  }
}

// ==========================================
// AzureML Environment Version (Child)
// ==========================================
resource environmentVersion 'Microsoft.MachineLearningServices/workspaces/environments/versions@2025-12-01' = {
  name: '1' // Explicit string assignment representing asset versioning
  parent: environmentContainer
  properties: {
    // description: 'Credit Risk Demo Environment'
    image: 'mcr.microsoft.com/azureml/sklearn-1.2-ubuntu20.04-py38-cpu-inference:latest'
    condaFile: loadTextContent('../envs/train-env.yml')
    autoRebuild: 'Disabled' // Prevents redundant continuous evaluation/build loops if configurations match
  }
}

// resource env 'Microsoft.MachineLearningServices/workspaces/environments@2024-04-01' = {
//   name: '${workspaceName}/${environmentName}'
//   properties: {
//     description: 'Credit Risk Demo Environment'
//     //isAnonymous: false
//     properties: {
//       condaFile: loadTextContent('../envs/train-env.yml')
//       image: 'mcr.microsoft.com/azureml/sklearn-1.2-ubuntu20.04-py38-cpu-inference:latest'
//     }
//   }
// }
