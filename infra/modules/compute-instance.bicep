param workspaceName string
param computeName string
param location string

resource instance 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  name: '${workspaceName}/${computeName}'
  location: location
  properties: {
    computeType: 'ComputeInstance'
    properties: {
      vmSize: 'STANDARD_DS11_V2' //'STANDARD_DS3_V2'
    }
  }
}
