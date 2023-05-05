# onnx 转 tensorrt 序列化 + 反序列化 (C++)

模型推理通常使用 NVIDIA 平台的 TensorRT。本文来讲述一下如何将 onnx 文件转化为 TensorRT 引擎。

## onnx 转换到 TensorRT 的 engine

使用 nvinfer1 的五个类：`IBuilder` `IBuilderConfig` `INetworkDefinition` `IParser`  `ICudaEngine` 来构建 TensorRT 引擎。这是固定格式，不用问为什么。我们要做的就是把代码中的 `model.onnx` 改成自己的 onnx 文件就行。

```cpp
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <logger.h>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

Logger logger;

int main(int argc, char** argv) {
  // 创建构建器
  IBuilder* builder = createInferBuilder(logger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilderConfig* config = builder->createBuilderConfig(); 
  
  // 创建网络模型
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

  // 解析ONNX模型
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

  // 构建引擎
  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(1 << 30);  // 1GB
  ICudaEngine* engine1 = builder->buildEngineWithConfig(*network, *config);
  
  _serialize_engine(engine1);

  // 反序列化引擎并创建执行上下文
  ICudaEngine* engine2 = _deserialize_engine();
  IExecutionContext* context = engine2->createExecutionContext();
  
  // 销毁不需要的资源
  context->destroy();
  engine2->destroy();
  engine1->destroy();
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  return 0;
}

```

## 序列化 TensorRT 引擎

如果需要转的 onnx 模型算子比较多，模型比较大，那么 onnx 转 TensorRT engine 的过程可能会非常耗时。因此需要将转好的模型进行序列化保存。上述代码中的 `_serialize_engine(engine1);` 这行代码就是序列化引擎的操作。具体实现方法如下：

```cpp
// 将引擎序列化为二进制流并保存为文件
void _serialize_engine(ICudaEngine* engine)
{
  IHostMemory* modelStream{ nullptr }; 
  modelStream = engine->serialize(); 
  ofstream f("model.engine", ios::binary); 
  // f << modelStream->data();
  f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
  f.close();
  modelStream->destroy();
}
```

上述代码中，引擎 engine 被保存到 `model.engine` 文件中，以便后续直接从 engine 文件读取。

> c++ 中文件写入有两种方法，分别是 << 和 f.write()，<< 更适合文本文件，f.write() 更适合二进制文件。在这里不做深究。

## 反序列化 TensorRT 引擎

将序列化后的 TensorRT 引擎文件，反序列化成 engine。代码如下：

```cpp
// 从文件中读取序列化后的引擎并反序列化
ICudaEngine* _deserialize_engine()
{
  IRuntime* runtime = createInferRuntime(logger);
  ICudaEngine* engine = nullptr;

  // 读取文件
  ifstream file("model.engine", std::ios::binary);
  if (file.good()) {
      // 获取文件大小
      file.seekg(0, file.end);
      size_t size = file.tellg();
      file.seekg(0, file.beg);

      // 分配内存
      vector<char> trtModelStream(size);
      assert(trtModelStream.data());

      // 读取文件内容
      file.read(trtModelStream.data(), size);
      file.close();

      // 反序列化引擎
      engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
  }
  
  // 销毁不需要的资源
  runtime->destroy();

  // 返回引擎
  return engine;
}
```

上述代码中，`model.engine` 文件被反序列化成 `ICudaEngine*` 类型，并返回给主函数。

## 模型推理

整理一下上述所有代码：

```cpp
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <logger.h>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

Logger logger;

// 将引擎序列化为二进制流并保存为文件
void _serialize_engine(ICudaEngine* engine)
{
  IHostMemory* modelStream{ nullptr }; 
  modelStream = engine->serialize(); 
  ofstream f("model.engine", ios::binary); 
  // f << modelStream->data();
  f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
  f.close();
  modelStream->destroy();
}

// 从文件中读取序列化后的引擎并反序列化
ICudaEngine* _deserialize_engine()
{
  IRuntime* runtime = createInferRuntime(logger);
  ICudaEngine* engine = nullptr;

  // 读取文件
  ifstream file("model.engine", std::ios::binary);
  if (file.good()) {
      // 获取文件大小
      file.seekg(0, file.end);
      size_t size = file.tellg();
      file.seekg(0, file.beg);

      // 分配内存
      vector<char> trtModelStream(size);
      assert(trtModelStream.data());

      // 读取文件内容
      file.read(trtModelStream.data(), size);
      file.close();

      // 反序列化引擎
      engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
  }
  
  // 销毁不需要的资源
  runtime->destroy();

  // 返回引擎
  return engine;
}

int main(int argc, char** argv) {
  // 创建构建器
  IBuilder* builder = createInferBuilder(logger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilderConfig* config = builder->createBuilderConfig(); 
  
  // 创建网络模型
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

  // 解析ONNX模型
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

  // 构建引擎
  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(1 << 30);  // 1GB
  ICudaEngine* engine1 = builder->buildEngineWithConfig(*network, *config);
  
  _serialize_engine(engine1);

  // 反序列化引擎并创建执行上下文
  ICudaEngine* engine2 = _deserialize_engine();
  IExecutionContext* context = engine2->createExecutionContext();
  
  // 销毁不需要的资源
  context->destroy();
  engine2->destroy();
  engine1->destroy();
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  return 0;
}

```

模型推理的时候使用上下文 `context` 这个变量。

## 总结

通过本文学习，了解如何将 onnx 模型转换成 TensorRT 引擎，并序列化以及反序列化。另外。nvinfer1 的五个类：`IBuilder` `IBuilderConfig` `INetworkDefinition` `IParser`  `ICudaEngine` 在创建的时候，可以使用智能指针，来避免内存泄漏。

# 日期

2023/05/05：本文创作
