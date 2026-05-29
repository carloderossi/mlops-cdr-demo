param workspaceName string
param location string

var prefixValue = replace(workspaceName, '-', '')
var randomSuffix = uniqueString(resourceGroup().id)
resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'stacc${randomSuffix}'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}

resource kv 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${workspaceName}-kv'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { 
      name: 'standard'
      family: 'A' 
    }
    accessPolicies: []
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: '${prefixValue}acreg'
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true
  }
}

resource appi 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi${randomSuffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Flow_Type: 'Bluefield'
    Request_Source: 'rest'
  }
}

resource aml 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storage.id
    containerRegistry: acr.id
    applicationInsights: appi.id
    publicNetworkAccess: 'Enabled'
    keyVault: kv.id    
  }
}

output workspaceId string = aml.id
