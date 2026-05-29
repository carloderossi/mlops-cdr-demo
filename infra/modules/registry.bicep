// https://learn.microsoft.com/en-us/azure/templates/microsoft.machinelearningservices/registries?pivots=deployment-language-bicep
// https://github.com/Azure/bicep-registry-modules/blob/main/avm/res/machine-learning-services/registry/main.bicep

@description('Name of the Azure ML Registry')
param registryName string

@description('Location of the registry')
param location string = resourceGroup().location

resource mlRegistry 'Microsoft.MachineLearningServices/registries@2026-03-01' = {
  name: registryName
  location: location

  identity: {
    type: 'SystemAssigned'
  }

  // sku: {
  //   name: 'Standard'
  //   tier: 'Standard'
  // }

  properties: {
    publicNetworkAccess: 'Enabled'

    regionDetails: [
      {
        location: location
        acrDetails: [
          {
            systemCreatedAcrAccount: {
              acrAccountName: '${registryName}acr'
              acrAccountSku: 'Standard'
              armResourceId: {
                resourceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${registryName}-managed/providers/Microsoft.ContainerRegistry/registries/${registryName}acr'
              }
            }
          }
        ]
        storageAccountDetails: [
          {
            systemCreatedStorageAccount: {
              storageAccountName: '${registryName}sa'
              storageAccountType: 'Standard_LRS'
              allowBlobPublicAccess: false
              storageAccountHnsEnabled: true
              armResourceId: {
                resourceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${registryName}-managed/providers/Microsoft.Storage/storageAccounts/${registryName}sa'
              }
            }
          }
        ]
      }
    ]
  }
}
