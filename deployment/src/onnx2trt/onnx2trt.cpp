#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <logger.h>

#include <cstring>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

bool _serialize_engine(ICudaEngine* engine)
{

  IHostMemory *modelStream = engine->serialize();
  string serialize_str;
  ofstream p;
  serialize_str.resize(modelStream->size());
  memcpy((void *)serialize_str.data(), modelStream->data(), modelStream->size());

  p.open("model.engine");
  p << serialize_str;
  p.close();
  return true;

}

int main(int argc, char** argv) {
  // Create builder 
  Logger logger;
  IBuilder* builder = createInferBuilder(logger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilderConfig* config = builder->createBuilderConfig(); 
  
  // Create model to populate the network 
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

  // Parse ONNX file 
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

  // Build engine
  builder->setMaxBatchSize(1);
  // builder->setMaxWorkspaceSize(1 << 30);  // 1GB
  // ICudaEngine* engine = builder->buildCudaEngine(*network);
  config->setMaxWorkspaceSize(1 << 30);  // 1GB
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  _serialize_engine(engine);
  
  engine->destroy();
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  return 0;
}
