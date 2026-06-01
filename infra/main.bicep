targetScope = 'subscription'

@description('Name of the resource group to create')
param resourceGroupName string

@description('Location for the resource group and Azure ML resources')
param location string

@description('Azure ML workspace name')
param workspaceName string

@description('Compute cluster name')
param computeClusterName string

@description('Compute instance name')
param computeInstanceName string

@description('Environment name')
param environmentName string

// @description('Registry name')
// param registryName string

//
// 1. Create the Resource Group
//
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
}

//
// 2. Deploy Workspace + Compute + Environment into the RG
//
module workspace './modules/workspace.bicep' = {
  name: 'workspace'
  scope: rg
  params: {
    workspaceName: workspaceName
    location: location
    environmentName: environmentName
  }
}

module computeCluster './modules/compute-cluster.bicep' = {
  name: 'computeCluster'
  scope: rg
  params: {
    workspaceName: workspaceName
    computeName: computeClusterName
    location: location
  }
  dependsOn: [workspace]
}

// module environment './modules/environment.bicep' = {
//    name: 'environment'
//    scope: rg
//    params: {
//      workspaceName: workspaceName
//      environmentName: environmentName
//    }
//    dependsOn: [workspace]
//  }

 module computeInstance './modules/compute-instance.bicep' = {
  name: 'computeInstance'
  scope: rg
  params: {
    workspaceName: workspaceName
    computeName: computeInstanceName
    location: location
  }
  dependsOn: [workspace]
}

// module registry './modules/registry.bicep' = {
//    name: 'registry'
//    scope: rg
//    params: {
//      registryName: registryName
//    }
// //   dependsOn: [workspace]
// }

output resourceGroup string = rg.name
output workspaceId string = workspace.outputs.workspaceId
