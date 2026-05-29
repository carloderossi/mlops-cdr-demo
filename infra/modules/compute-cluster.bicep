param workspaceName string
param computeName string
param location string

resource cluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  name: '${workspaceName}/${computeName}'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'STANDARD_DS3_V2'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 2
      }
    }
  }
}
