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
