![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)
![GitHub](https://img.shields.io/badge/Release-PROD-yellow.svg)
![GitHub](https://img.shields.io/badge/Languages-MULTI-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# Model Deployment

![logo](https://github.com/balavenkatesh3322/model_deployment/blob/master/logo.jpg)

## What is Model serving?
When it comes to deploying ML models, data scientists have to make a choice based on their use case. If they need a high volume of predictions and latency is not an issue, they typically perform inference in batch, feeding the model with large amounts of data and writing the predictions into a table. If they need predictions at low latency, e.g. in response to a user action in an app, the best practice is to deploy ML models as REST endpoints. These apps allows to send requests to an endpoint that’s always up and receive the prediction immediately.

***

| Library Name | Description |
|   :---:      |     :---:      |
| [Tensorflow Serving](https://www.tensorflow.org/serving/) | High-performant framework to serve Tensorflow models via grpc protocol able to handle 100k requests per second per core | 
|[TorchServe](https://github.com/pytorch/serve) | TorchServe is a flexible and easy to use tool for serving PyTorch models.|
| [BentoML](https://github.com/bentoml/BentoML) | BentoML is an open source framework for high performance ML model serving |
| [Clipper](https://github.com/ucbrise/clipper) |  Model server project from Berkeley's Rise Rise Lab which includes a standard RESTful API and supports TensorFlow, Scikit-learn and Caffe models| 
| [Cortex](https://github.com/cortexlabs/cortex) | Cortex is an open source platform for deploying machine learning models—trained with nearly any framework—as production web services.| 
|[Multi-Model-server](https://github.com/awslabs/multi-model-server) | Multi Model Server (MMS) is a flexible and easy to use tool for serving deep learning models trained using any ML/DL framework.|
| [DeepDetect](https://github.com/beniz/deepdetect) |  Machine Learning production server for TensorFlow, XGBoost and Cafe models written in C++ and maintained by Jolibrain| 
| [ForestFlow](https://github.com/ForestFlow/ForestFlow) |  Cloud-native machine learning model server.| 
| [Jina](https://github.com/jina-ai/jina)  |  Cloud native search framework that   supports to use deep learning/state of the art AI models for search.| 
| [KFServing](https://github.com/kubeflow/kfserving) | Serverless framework to deploy and monitor machine learning models in Kubernetes - [(Video)](https://www.youtube.com/watch?v=hGIvlFADMhU)|  
| [NVIDIA TensorRT Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) |  TensorRT Inference Server is an inference microservice that lets you serve deep learning models in production while maximizing GPU utilization.| 
| [OpenScoring](https://github.com/openscoring/openscoring) |  REST web service for scoring PMML models built and maintained by OpenScoring.io| 
| [Redis-AI](https://github.com/RedisAI/RedisAI) |  A Redis module for serving tensors and executing deep learning models. Expect changes in the API and internals.| 
| [Seldon Core](https://github.com/SeldonIO/seldon-core) |  Open source platform for deploying and monitoring machine learning models in kubernetes - [(Video)](https://www.youtube.com/watch?v=pDlapGtecbY)| 
| [model_server](https://github.com/openvinotoolkit/model_server) | OpenVINO™ Model Server is a scalable, high-performance solution for serving machine learning models optimized for Intel® architectures. The server provides an inference service via gRPC enpoint or REST API |

***
