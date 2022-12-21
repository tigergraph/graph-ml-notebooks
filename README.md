# TigerGraph ML Workbench: Graph ML as a Service

TigerGraph’s Machine Learning Workbench is a Python-based toolkit that accelerates the development of graph-enhanced machine learning, which leverages the added insight from connected data and graph features for better predictions. The GitHub repository of the notebooks are found [here](https://github.com/tigergraph/graph-ml-notebooks).


## Set Up Your Workbench

<!-- ### Step 0. Provisioning a DB cluster with ML workbench 

Create a TigerGraph Database Cluster with Graph Machine Learning Workbench that is enabled. 

<img src="https://tigergraph-public-data.s3.us-west-1.amazonaws.com/images/tgcloud-provisioning.png" width="800"> -->

### Step 1. Create DB credentials and set access permission

To access the graph database through pyTigerGraph, we need to create a database username and password, then put these credentials in config.json.

1. Go back to your browser tab/window for TigerGraph Cloud.
2. Click on `Cluster` on the left side menu. For the cluster containing this workbench, click `Access Management`.
3. Create a database user and grant appropriate permissions (e.g., `globaldesigner`). More details about managing database users can be found here: https://docs.tigergraph.com/cloud/security/manage-db-users. More details about access control settings can be found here:  https://docs.tigergraph.com/tigergraph-server/current/user-access/access-control-model#_built_in_roles.


<img src="https://tigergraph-public-data.s3.us-west-1.amazonaws.com/images/create-user.png" width="700">

<img src="https://tigergraph-public-data.s3.us-west-1.amazonaws.com/images/create-user-role.png" width="700">

### Step 2. Update the database credentials in config.json



1. After creating the login credentials in Step 1, go back to the ML Workbench and edit `config.json` in the root jupyter notebook folder to replace the host, username and password with your new credentials. Example: [config.json](./config.json) 
```json
{
    "host": "https://subdomain.i.tgcloud.io",
    "username": "user_1",
    "password": "MyPassword1!",
    "getToken": true 
}
```
Note: For the `host` parameter, it is the domain name of the Cluster. You can find it in Cluster’s Details page which can be found by clicking on Clusters on tgCloud's left panel, then by clicking on the cluster’s name in the list (`Details -> Network Information -> Domain`). Replace the substring `subdomain` with your actual subdomain. Make sure to keep the “https://” at the beginning of the domain in the json config.

<img src="https://tigergraph-public-data.s3.us-west-1.amazonaws.com/images/tgcloud-host.png" width="700">


2. Once the credentials are updated, all the example notebooks and demos will refer to this config for database connections via pyTigerGraph. For example, here is how the [algos/centrality.ipynb](algos/centrality.ipynb)  notebook connects to the database:

```python
from pyTigerGraph import TigerGraphConnection
conn = TigerGraphConnection(
    host=config["host"],
    username=config["username"],
    password=config["password"]
)
```

## Learn Graph ML from Example Notebooks

The ML Workbench comes with a collection of canonical Python notebooks that will introduce you to a number of features of the TigerGraph ML ecosystem.

<!-- * In the `basics` directory, you can find notebooks of how to get started with pyTigerGraph.
* In the `algos` directory, you can find notebooks for each category of algorithms within ourTigerGraph's [Graph Data Science Library](https://docs.tigergraph.com/graph-ml/current/intro/). You can run these algorithmss via the pyTigerGraph Featurizer functionality.
* In the `GNNs` directory, you can find tutorial notebooks on how to start training GNNs using data stored in a TigerGraph database.
* In the `applications` directory, you can find end to end demos of common applications such as fraud detection and recommendation. -->

- The `basics` directory contains notebooks on how to get started with pyTigerGraph.
- The `algos` directory contains notebooks for each category of algorithms within TigerGraph's [Graph Data Science Library](https://docs.tigergraph.com/graph-ml/current/intro/). You can run these algorithms via the pyTigerGraph Featurizer functionality.
- The `GNNs` directory contains tutorial notebooks on how to train GNNs using data stored in a TigerGraph database.
- The `applications` directory contains end to end demos of common applications such as fraud detection and recommendation.

We recommend starting with the tutorials in the `basics` folders if you are new to pyTigerGraph. Once you are familiar with our pyTigerGraph client, familiarize yourself with a few graph algorithms with the examples in the `algos` folder before going through the `GNNs` and end-to-end `applications` tutorials.


### 1. Getting Started with pyTigerGraph and GSQL

| folder | notebook  | intro |
| :--- | :--- | :--- |
| basics | [datasets.ipynb](./basics/datasets.ipynb)  | Load Data into TigerGraph |
| basics | [feature_engineering.ipynb](./basics/feature_engineering.ipynb)  | Util functions about building graph features from TigerGraph |
| basics | [pyTigergraph_101.ipynb](./basics/pyTigergraph_101.ipynb)  | Basic pyTigerGraph examples|
| basics | [gsql_101.ipynb](./basics/gsql_101.ipynb)  | Basic GSQL 101 using pyTigerGraph |
| basics | [gsql_102.ipynb](./basics/gsql_102.ipynb)  | Advanced GSQL 102 (pattern match) using pyTigerGraph |

### 2. Graph Algorithms 

| folder | notebook  | intro |
| :--- | :--- | :--- |
| algos | [centrality.ipynb](./algos/centrality.ipynb)  | Centrality algorithms |
| algos | [community.ipynb](./algos/community.ipynb)  | Community detection algorithms |
| algos | [similarity.ipynb](./algos/similarity.ipynb)  | Similarity algorithms |
| algos | [pathfinding.ipynb](./algos/pathfinding.ipynb)  | Pathfinding between vertices |
| algos | [embedding.ipynb](./algos/embedding.ipynb)  | Graph embedding algorithms  |
| algos | [classification.ipynb](./algos/classification.ipynb)  | Node classification algorithms |
| algos | [topologicalLinkPrediction.ipynb](./algos/topologicalLinkPrediction.ipynb)  | Topological link predictions |


### 3. Graph Neural Networks with TigerGraph

| folder | notebook  | intro |
| :--- | :--- | :--- |
| GNNs/PyG | [gcn_node_classification.ipynb](./GNNs/PyG/gcn_node_classification.ipynb)  | Node classification using PyG |
| GNNs/PyG | [gcn_link_prediction.ipynb](./GNNs/PyG/gcn_link_prediction.ipynb)  |  Link prediction using PyG |
| GNNs/PyG | [hgat_node_classification.ipynb](./GNNs/PyG/hgat_node_classification.ipynb)  | Heterogeneous Graph Attention Network using PyG |
| GNNs/DGL | [gcn_node_classification.ipynb](./GNNs/DGL/gcn_node_classification.ipynb)  | Node classification using DGL |
| GNNs/DGL | [rgcn_node_classification.ipynb](./GNNs/DGL/rgcn_node_classification.ipynb)  | Heterogeneous Graph Convolutional Network using DGL |
| GNNs/Spektral | [gcn_node_classification.ipynb](./GNNs/Spektral/gcn_node_classification.ipynb)  | Node classification using Spektral for Tensorflow |


### 4. End-to-end Applications using Graph ML

| folder | notebook  | intro |
| :--- | :--- | :--- |
| applications/fraud_detection | [fraud_detection.ipynb](./applications/fraud_detection/fraud_detection.ipynb)  | End-to-end fraud detection using Graph ML |
| applications/recommendation | [recommendation.ipynb](./applications/recommendation/recommendation.ipynb)  | End-to-end recommendation using Graph ML |
